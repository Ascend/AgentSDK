#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# AgentSDK publishes this standalone src-layout package across independent Parts; the complete
# ClawCodex source and focused tests validate imports and dynamic patterns during migration.
# The target hook also enables legacy default diagnostics beyond its declared high-value set.
# pylint: disable=E0402,W0621,W0706,W2301

"""LkbRepository Protocol and JSON-file-backed implementation.

Spec appendix B — public repository interface.
Spec §7.6 — execute_atomic two-phase optimistic commit (delegates to json_store).
Spec §7.8 – §7.10 — lifecycle state machine (delegates to lifecycle.py).
Spec §7.12 — doctor / repair (delegates to doctor.py).
Spec §5.2 — board identity resolution (delegates to board_resolver.py).

The repository is the top-level entry point used by the application service
and CLI commands.  It hides the filesystem layout, lock management, and
store internals behind a clean Protocol so future storage backends (e.g.
SQLite, remote) can be swapped in without changing callers.

This module imports nothing from ToolContext or Task-v2 (spec §11.4 inv 12).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .audit_compaction import AuditCompactionResult
from .board_resolver import (
    LKB_BOARDS_SUBDIR,
    board_dir,
    resolve_board as br_resolve_board,
    resolve_home,
)
from .commands import CommandResult
from .doctor import DoctorReport, doctor as run_doctor
from .file_lock import BoardFileLock, BoardStoreBusyError
from .graph_types import Board, BoardPolicy, GraphSnapshot, RevisionVector
from .json_store import (
    BoardEnvelope,
    BoardNotFoundError,
    BoardStoreCorruptError,
    BoardTombstonedError,
    JsonBoardStore,
    set_payload_hash,
    validate_board_envelope,
    validate_envelope_schema,
    verify_payload_hash,
)
from .lifecycle import (
    LifecycleData,
    LifecycleTransitionDenied,
    archive_board,
    board_lifecycle_state,
    close_board,
    purge_board,
    read_archive,
    read_tombstone,
    restore_board,
    tombstone_path,
    trash_board,
)
from .ir_hash import canonical_hash


def _make_command_id(prefix: str = "cmd") -> str:
    """Generate a unique command ID (UUID-based)."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


__all__ = [
    "ArchiveRef",
    "BoardHeader",
    "BoardNotFoundError",
    "BoardStoreBusyError",
    "BoardStoreCorruptError",
    "JsonFileLkbRepository",
    "LkbRepository",
    "LifecycleTransitionDenied",
    "get_repository",
]


# ── BoardHeader (list_boards lightweight record) ──────────────────────


@dataclass(frozen=True)
class BoardHeader:
    """Lightweight header for board listings.

    Contains enough information for ``list_boards`` to display a summary
    of every known board without loading the full envelope of each one.
    Spec §7.3 mentions Board Header as part of the envelope.
    """

    board_id: str
    display_name: str = ""
    schema_version: int = 0
    store_revision: int = 0
    lifecycle_state: str = "active"
    board_dir: Path = field(default_factory=Path)


# ── ArchiveRef ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchiveRef:
    """Reference to an archived board.

    Returned by :meth:`LkbRepository.archive` and accepted by
    :meth:`LkbRepository.restore`.  Contains enough information to
    locate and verify the archive, plus metadata for display.
    """

    board_id: str
    archive_path: Path
    store_revision: int = 0
    payload_hash: str = ""
    archived_at: str = ""
    archived_by: str = ""
    reason: str = ""


# ── LkbRepository Protocol (spec appendix B) ──────────────────────────


class LkbRepository(Protocol):
    """Protocol for board repository operations (spec appendix B).

    The repository is the primary entry point for loading, mutating, and
    managing boards.  It is intentionally storage-agnostic so callers
    do not depend on the JSON-on-disk implementation.
    """

    # ── core read / write ────────────────────────────────────────────

    def resolve_board(
        self,
        workspace_root: str | Path | None = None,
        *,
        explicit_id: str | None = None,
        session_id: str | None = None,
    ) -> Board:
        """Resolve a Board identity using the 5-tier priority from spec §5.2.

        If the board already exists on disk, its stored metadata is used.
        Otherwise a fresh Board object is returned (not yet persisted).
        """
        ...

    def load_snapshot(self, board_id: str) -> GraphSnapshot:
        """Load the current graph snapshot for *board_id*.

        Raises :class:`BoardNotFoundError` if the board does not exist.
        Raises :class:`BoardStoreCorruptError` if the board is unreadable
        and unrecoverable.
        """
        ...

    def load_envelope(self, board_id: str) -> BoardEnvelope:
        """Load the validated persisted envelope for *board_id*."""
        ...

    def execute_atomic(
        self,
        board_id: str,
        command_id: str,
        request_hash: str,
        expected_revision_vector: RevisionVector | None,
        mutate: Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]],
        *,
        expected_store_revision: int | None = None,
        actor: str = "repository",
        reason: str | None = None,
        audit_context: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Execute *mutate* atomically against *board_id*.

        See :meth:`JsonBoardStore.execute_atomic` for the full two-phase
        protocol (spec §7.6).
        """
        ...

    # ── lifecycle (spec §7.8 – §7.10) ───────────────────────────────

    def archive(self, board_id: str, reason: str) -> ArchiveRef:
        """Archive *board_id*.  Returns an :class:`ArchiveRef`.

        The board must be in ``closed`` state (or this will close it
        first).  See spec §7.10.
        """
        ...

    def compact_audit(self, board_id: str, *, actor: str = "operator") -> AuditCompactionResult:
        """Manually compact old Audit records into immutable history."""
        ...

    def restore(self, archive_ref: ArchiveRef) -> Board:
        """Restore a board from an archive.

        Returns the restored Board.  Does not overwrite an existing
        active board with the same ID (raises an error instead).
        See spec §7.10.
        """
        ...

    def doctor(self, board_id: str, *, repair: bool = False) -> DoctorReport:
        """Run integrity diagnostics on *board_id*.

        When *repair* is True, performs only safe automatic repairs
        (restore from .bak, clean orphan temp files, etc.).  Never
        deletes project boards or archives.
        """
        ...

    # ── enumeration ──────────────────────────────────────────────────

    def list_boards(
        self,
        *,
        include_archived: bool = False,
    ) -> list[BoardHeader]:
        """List all known boards.

        Scans the boards directory and reads just the header from each
        ``board.json``.  Filters out trashed/purged boards.  When
        *include_archived* is True, archived boards are also returned.
        """
        ...

    # ── additional lifecycle operations ──────────────────────────────

    def close(
        self,
        board_id: str,
        reason: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Close a board (active -> closed)."""
        ...

    def trash(
        self,
        board_id: str,
        reason: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Move a board to trash (closed/archived -> trashed)."""
        ...

    def purge(
        self,
        board_id: str,
        reason: str,
        confirm: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Irreversibly purge a board.  Requires *confirm* == board_id."""
        ...


# ── JsonFileLkbRepository ─────────────────────────────────────────────


class JsonFileLkbRepository:
    """JSON-file-backed implementation of :class:`LkbRepository`.

    Holds a home root, uses ``board_resolver`` for paths, and caches
    per-board ``(BoardFileLock, JsonBoardStore)`` instances so repeated
    operations on the same board don't rebuild them.

    Lifecycle operations delegate to ``lifecycle.py``.  Doctor
    operations delegate to ``doctor.py``.

    Parameters
    ----------
    home:
        Optional home directory override.  If None, follows the standard
        precedence (explicit > ``CLAWCODEX_HOME`` env > ``Path.home()``).
    """

    def __init__(self, *, home: Path | None = None) -> None:
        self._home = resolve_home(home=home)
        self._boards_root = self._home / LKB_BOARDS_SUBDIR
        # Per-board cache: board_id -> (lock, store)
        self._board_cache: dict[str, tuple[BoardFileLock, JsonBoardStore]] = {}
        self._cache_lock = threading.Lock()

    # ── public: core read / write ────────────────────────────────────

    @property
    def home(self) -> Path:
        return self._home

    def resolve_board(
        self,
        workspace_root: str | Path | None = None,
        *,
        explicit_id: str | None = None,
        session_id: str | None = None,
    ) -> Board:
        """Resolve board identity and ensure the board exists on disk.

        If a board with the resolved ID already exists, its stored
        metadata is returned.  Otherwise a new board is created with
        the resolved identity.
        """
        board_identity = br_resolve_board(
            workspace_root,
            explicit_id=explicit_id,
            session_id=session_id,
        )
        board_id = board_identity.board_id
        d = board_dir(board_id, home=self._home)
        board_json = d / "board.json"

        if board_json.is_file():
            # Board exists — load stored metadata
            store = self._get_store(board_id)
            try:
                env = store.load()
                b = env.board
                policy_dict = b.get("policy", {})
                if not isinstance(policy_dict, dict):
                    policy_dict = {}
                return Board(
                    board_id=str(b.get("board_id", board_id)),
                    project_uri=str(b.get("project_uri", "")),
                    display_name=str(b.get("display_name", "")),
                    schema_version=int(b.get("schema_version", 1)),
                    store_revision=env.store_revision,
                    created_at=str(b.get("created_at", "")),
                    updated_at=str(b.get("updated_at", "")),
                    policy=BoardPolicy.from_dict(policy_dict),
                )
            except BoardStoreCorruptError:
                raise
        else:
            # Board doesn't exist yet — create it
            try:
                self._create_board(board_identity)
            except FileExistsError:
                return self.resolve_board(
                    workspace_root,
                    explicit_id=board_id,
                    session_id=session_id,
                )
            return board_identity

    def load_snapshot(self, board_id: str) -> GraphSnapshot:
        """Load the current graph snapshot for *board_id*."""
        store = self._get_store(board_id)
        if not store.exists():
            raise BoardNotFoundError(board_id, self._board_path(board_id))
        return store.read_snapshot()

    def load_envelope(self, board_id: str) -> BoardEnvelope:
        """Load the authoritative validated envelope for *board_id*."""
        store = self._get_store(board_id)
        if not store.exists():
            raise BoardNotFoundError(board_id, self._board_path(board_id))
        return store.load()

    def execute_atomic(
        self,
        board_id: str,
        command_id: str,
        request_hash: str,
        expected_revision_vector: RevisionVector | None,
        mutate: Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]],
        *,
        expected_store_revision: int | None = None,
        actor: str = "repository",
        reason: str | None = None,
        audit_context: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Execute *mutate* atomically against *board_id*.

        Ordinary mutations do not create or resurrect boards.
        """
        store = self._get_store(board_id)
        if not store.exists():
            raise BoardNotFoundError(board_id, self._board_path(board_id))

        return store.execute_atomic(
            board_id,
            command_id,
            request_hash,
            expected_revision_vector,
            mutate,
            expected_store_revision=expected_store_revision,
            actor=actor,
            reason=reason,
            audit_context=audit_context,
        )

    # ── public: lifecycle ────────────────────────────────────────────

    def close(
        self,
        board_id: str,
        reason: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Close a board (active -> closed)."""
        store = self._get_store(board_id)
        # using _make_command_id helper

        cid = _make_command_id("close")
        rh = canonical_hash({"kind": "close_board", "board_id": board_id, "reason": reason})
        return close_board(
            store,
            board_id,
            actor=actor,
            command_id=cid,
            request_hash=rh,
            reason=reason,
        )

    def archive(self, board_id: str, reason: str) -> ArchiveRef:
        """Archive *board_id*.  Returns an :class:`ArchiveRef`."""
        store = self._get_store(board_id)

        # Step 1: close first if still active (best-effort; if already
        # closed or archived this is a no-op transition that will be
        # handled by the state machine).
        # using _make_command_id helper

        # Check current state
        env = store.load()
        state = board_lifecycle_state(env)

        if state == "active":
            cid = _make_command_id("close-for-archive")
            rh = canonical_hash({"kind": "close_board", "board_id": board_id, "reason": reason})
            close_result = close_board(
                store,
                board_id,
                actor="system",
                command_id=cid,
                request_hash=rh,
                reason=reason,
            )
            if not close_result.committed:
                # If close was denied, we can't archive
                raise LifecycleTransitionDenied(
                    board_id=board_id,
                    from_state=state,
                    to_state="archived",
                    reason=f"Cannot archive board {board_id!r}: {close_result.reason}",
                )

        # Step 2: archive
        cid = _make_command_id("archive")
        rh = canonical_hash({"kind": "archive_board", "board_id": board_id, "reason": reason})
        result = archive_board(
            store,
            board_id,
            actor="system",
            command_id=cid,
            request_hash=rh,
            reason=reason,
        )

        if not result.committed:
            raise LifecycleTransitionDenied(
                board_id=board_id,
                from_state=state,
                to_state="archived",
                reason=f"Archive of board {board_id!r} failed: {result.reason}",
            )

        # Build ArchiveRef from current state
        env = store.load()
        archive_info = env.lifecycle.get("archive_info", {}) if isinstance(env.lifecycle, dict) else {}
        archive_path = Path(str(archive_info.get("archive_path", "")))
        archive_document = read_archive(archive_path, expected_board_id=board_id)

        return ArchiveRef(
            board_id=board_id,
            archive_path=archive_path,
            store_revision=int(archive_document["sourceStoreRevision"]),
            payload_hash=str(archive_document["payloadHash"]),
            archived_at=str(archive_info.get("archived_at", "")),
            archived_by=str(archive_info.get("archived_by", "system")),
            reason=str(archive_info.get("reason", reason)),
        )

    def restore(self, archive_ref: ArchiveRef) -> Board:
        """Restore a board from an archive ref.

        Verifies the archive schema and hash, then creates a new active
        revision of the board.  Does not overwrite an existing active
        board with the same ID.
        """
        board_id = archive_ref.board_id
        marker = tombstone_path(self._home / "lkb", board_id)
        if marker.is_file():
            read_tombstone(marker, expected_board_id=board_id)
            raise LifecycleTransitionDenied(
                board_id=board_id,
                from_state="purged",
                to_state="active",
                reason=f"Cannot restore tombstoned board {board_id!r}",
            )

        # Verify archive file exists and is readable
        if not archive_ref.archive_path.is_file():
            raise BoardNotFoundError(board_id, archive_ref.archive_path)
        archive_document = read_archive(archive_ref.archive_path, expected_board_id=board_id)
        if archive_ref.store_revision and archive_ref.store_revision != archive_document["sourceStoreRevision"]:
            raise LifecycleTransitionDenied(board_id, "archive", "active", "ArchiveRef revision mismatch")
        if archive_ref.payload_hash and archive_ref.payload_hash != archive_document["payloadHash"]:
            raise LifecycleTransitionDenied(board_id, "archive", "active", "ArchiveRef hash mismatch")

        # Check for existing active board
        d = board_dir(board_id, home=self._home)
        board_json = d / "board.json"
        if board_json.is_file():
            # Check lifecycle state
            try:
                store = self._get_store(board_id)
                env = store.load()
                state = board_lifecycle_state(env)
                if env.lifecycle.get("tombstone") or state in {"purging", "purged"}:
                    raise LifecycleTransitionDenied(
                        board_id=board_id,
                        from_state=state,
                        to_state="active",
                        reason=f"Cannot restore tombstoned board {board_id!r}",
                    )
                if state == "active":
                    raise LifecycleTransitionDenied(
                        board_id=board_id,
                        from_state="active",
                        to_state="active",
                        reason=(f"Cannot restore board {board_id!r}: an active board with this ID already exists"),
                    )
            except BoardStoreCorruptError:
                # Corrupt but exists — let's try restore anyway
                pass

        # If the board directory exists (e.g. trashed or archived),
        # use restore_board on the existing store.  Otherwise, we need
        # to create the board from the archive.
        # using _make_command_id helper

        if board_json.is_file():
            store = self._get_store(board_id)
            cid = _make_command_id("restore")
            rh = canonical_hash(
                {
                    "kind": "restore_board",
                    "board_id": board_id,
                    "archive": str(archive_ref.archive_path),
                }
            )
            result = restore_board(
                store,
                board_id,
                archive_ref=archive_ref,
                actor="system",
                command_id=cid,
                request_hash=rh,
                reason=f"restored from archive {archive_ref.archive_path}",
            )
            if not result.committed:
                raise LifecycleTransitionDenied(
                    board_id=board_id,
                    from_state="archived",
                    to_state="active",
                    reason=f"Restore of board {board_id!r} failed: {result.reason}",
                )
        else:
            # Board doesn't exist — create it from archive
            self._restore_from_archive_file(archive_ref)

        # Return the restored board
        return self.resolve_board(explicit_id=board_id)

    def trash(
        self,
        board_id: str,
        reason: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Move a board to trash."""
        store = self._get_store(board_id)
        # using _make_command_id helper

        cid = _make_command_id("trash")
        rh = canonical_hash({"kind": "trash_board", "board_id": board_id, "reason": reason})
        return trash_board(
            store,
            board_id,
            actor=actor,
            command_id=cid,
            request_hash=rh,
            reason=reason,
        )

    def purge(
        self,
        board_id: str,
        reason: str,
        confirm: str,
        *,
        actor: str = "system",
    ) -> CommandResult:
        """Irreversibly purge a board.  Requires *confirm* == board_id."""
        if confirm != board_id:
            raise ValueError("purge confirmation must exactly match board_id")
        if not reason.strip():
            raise ValueError("purge reason is required")
        marker = tombstone_path(self._home / "lkb", board_id)
        if marker.is_file():
            read_tombstone(marker, expected_board_id=board_id)
            directory = board_dir(board_id, home=self._home)
            lock = BoardFileLock(directory)
            store = JsonBoardStore(
                directory,
                board_id=board_id,
                lock=lock,
                lkb_root=self._home / "lkb",
            )
        else:
            store = self._get_store(board_id)
        # using _make_command_id helper

        cid = _make_command_id("purge")
        rh = canonical_hash({"kind": "purge_board", "board_id": board_id, "reason": reason, "confirm": confirm})
        result = purge_board(
            store,
            board_id,
            actor=actor,
            command_id=cid,
            request_hash=rh,
            reason=reason,
            confirm=confirm,
            authorized=True,
        )

        if result.committed:
            # Remove from cache
            with self._cache_lock:
                self._board_cache.pop(board_id, None)

        return result

    def doctor(self, board_id: str, *, repair: bool = False) -> DoctorReport:
        """Run integrity diagnostics on *board_id*."""
        return run_doctor(
            board_id,
            repair=repair,
            home=self._home,
        )

    def compact_audit(self, board_id: str, *, actor: str = "operator") -> AuditCompactionResult:
        """Manually compact old Audit records into immutable history."""
        store = self._get_store(board_id)
        if not store.exists():
            raise BoardNotFoundError(board_id, self._board_path(board_id))
        return store.compact_audit(board_id, actor=actor)

    # ── public: enumeration ──────────────────────────────────────────

    def list_boards(
        self,
        *,
        include_archived: bool = False,
    ) -> list[BoardHeader]:
        """List all known boards.

        Scans ``<home>/lkb/boards/`` and reads the header (board_id,
        display_name, store_revision, lifecycle state) from each
        ``board.json``.  Boards that fail validation are skipped
        (call :meth:`doctor` to inspect them).
        """
        boards_root = self._boards_root
        if not boards_root.is_dir():
            return []

        results: list[BoardHeader] = []

        for entry in sorted(boards_root.iterdir()):
            if not entry.is_dir():
                continue
            board_json = entry / "board.json"
            if not board_json.is_file():
                continue

            try:
                header = self._read_header(board_json, entry)
            except (OSError, ValueError, BoardStoreCorruptError):
                # Enumeration is best-effort by contract: one damaged Board
                # must not hide unrelated healthy Boards.  ``doctor`` remains
                # the explicit diagnostic/repair entry point for a known ID.
                continue

            # Filter by lifecycle state
            state = header.lifecycle_state
            if state in ("trashed", "purging", "purged"):
                continue
            if state == "archived" and not include_archived:
                continue

            results.append(header)

        return results

    # ── internal: board cache ────────────────────────────────────────

    def _get_store(self, board_id: str) -> JsonBoardStore:
        """Get (or create) the JsonBoardStore for *board_id*."""
        with self._cache_lock:
            cached = self._board_cache.get(board_id)
            if cached is not None:
                return cached[1]

        d = board_dir(board_id, home=self._home)
        marker = tombstone_path(self._home / "lkb", board_id)
        if marker.is_file():
            read_tombstone(marker, expected_board_id=board_id)
            raise BoardTombstonedError(board_id, marker)
        d.mkdir(parents=True, exist_ok=True)

        lock = BoardFileLock(d)
        store = JsonBoardStore(
            d,
            board_id=board_id,
            lock=lock,
            lkb_root=self._home / "lkb",
        )

        with self._cache_lock:
            # Double-check — another thread might have populated it
            cached = self._board_cache.get(board_id)
            if cached is not None:
                return cached[1]
            self._board_cache[board_id] = (lock, store)
            return store

    def _board_path(self, board_id: str) -> Path:
        return board_dir(board_id, home=self._home) / "board.json"

    # ── internal: board creation ─────────────────────────────────────

    def _create_board(self, board: Board) -> JsonBoardStore:
        """Create a new board on disk from a Board identity."""
        tombstone = tombstone_path(self._home / "lkb", board.board_id)
        if tombstone.is_file():
            raise LifecycleTransitionDenied(
                board_id=board.board_id,
                from_state="purged",
                to_state="active",
                reason=f"Board {board.board_id!r} has a tombstone and cannot be recreated",
            )
        d = board_dir(board.board_id, home=self._home)
        lock = BoardFileLock(d)
        store = JsonBoardStore.create_board(
            d,
            board=board,
            lock=lock,
            lkb_root=self._home / "lkb",
        )
        with self._cache_lock:
            self._board_cache[board.board_id] = (lock, store)
        return store

    def _create_board_from_id(self, board_id: str) -> JsonBoardStore:
        """Create a minimal board with just a board_id."""
        now = datetime.now(timezone.utc).isoformat()
        board = Board(
            board_id=board_id,
            project_uri=f"board:{board_id}",
            display_name=board_id,
            schema_version=1,
            store_revision=0,
            created_at=now,
            updated_at=now,
            policy=BoardPolicy(),
        )
        return self._create_board(board)

    def _read_header(self, board_json: Path, board_dir_path: Path) -> BoardHeader:
        """Read a lightweight header from a board.json file."""
        with open(board_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        validate_envelope_schema(data)
        if not verify_payload_hash(data):
            raise BoardStoreCorruptError(f"{board_json} has an invalid payload hash")

        board = data.get("board", {}) if isinstance(data, dict) else {}
        lc = data.get("lifecycle", {}) if isinstance(data, dict) else {}

        return BoardHeader(
            board_id=str(board.get("board_id", "")),
            display_name=str(board.get("display_name", "")),
            schema_version=int(data.get("schemaVersion", 0)) if isinstance(data, dict) else 0,
            store_revision=int(data.get("storeRevision", 0)) if isinstance(data, dict) else 0,
            lifecycle_state=str(lc.get("state", "active")) if isinstance(lc, dict) else "active",
            board_dir=board_dir_path,
        )

    # ── internal: restore from archive file ──────────────────────────

    def _restore_from_archive_file(self, archive_ref: ArchiveRef) -> None:
        """Atomically restore a verified wrapper when the Board is absent."""
        archive = read_archive(archive_ref.archive_path, expected_board_id=archive_ref.board_id)
        if archive_ref.store_revision and archive_ref.store_revision != archive["sourceStoreRevision"]:
            raise LifecycleTransitionDenied(archive_ref.board_id, "archive", "active", "ArchiveRef revision mismatch")
        if archive_ref.payload_hash and archive_ref.payload_hash != archive["payloadHash"]:
            raise LifecycleTransitionDenied(archive_ref.board_id, "archive", "active", "ArchiveRef hash mismatch")
        restored = BoardEnvelope.from_dict(archive["envelope"])
        lifecycle = LifecycleData.from_dict(restored.lifecycle)
        lifecycle.state = "active"
        lifecycle.updated_at = datetime.now(timezone.utc).isoformat()
        lifecycle.closed_at = ""
        lifecycle.archived_at = ""
        restored.lifecycle = lifecycle.to_dict()
        restored.lifecycle["archive_info"] = {
            "archive_id": archive["archiveId"],
            "archive_path": str(archive_ref.archive_path),
            "archive_hash": archive["payloadHash"],
            "source_store_revision": archive["sourceStoreRevision"],
            "source_payload_hash": archive["sourcePayloadHash"],
        }
        restored.lifecycle["restore_info"] = {
            "source_archive_id": archive["archiveId"],
            "source_archive_hash": archive["payloadHash"],
            "source_store_revision": archive["sourceStoreRevision"],
            "source_archive_path": str(archive_ref.archive_path),
            "restored_by": "system",
            "restored_at": lifecycle.updated_at,
        }
        restored.store_revision = int(archive["sourceStoreRevision"]) + 1
        restored.board["store_revision"] = restored.store_revision
        cid = _make_command_id("restore-from-archive")
        request_hash = canonical_hash(
            {
                "kind": "restore_from_archive",
                "board_id": archive_ref.board_id,
                "archive": str(archive_ref.archive_path),
            }
        )
        restored.processed_commands[cid] = {
            "command_id": cid,
            "request_hash": request_hash,
            "decision": "committed",
            "actor": "system",
            "store_revision": restored.store_revision,
            "revision_vector": restored.current_revision_vector().to_dict(),
            "reason": f"restored from archive {archive_ref.archive_path}",
        }
        restored.events.append(
            {
                "type": "archive_restored",
                "command_id": cid,
                "actor": "system",
                "store_revision": restored.store_revision,
                "source_archive_id": archive["archiveId"],
                "source_archive_hash": archive["payloadHash"],
                "source_store_revision": archive["sourceStoreRevision"],
            }
        )
        set_payload_hash(restored, previous_hash=str(archive["sourcePayloadHash"]))
        validate_board_envelope(restored, board_id=archive_ref.board_id)

        board_directory = board_dir(archive_ref.board_id, home=self._home)
        lock = BoardFileLock(board_directory)
        store = JsonBoardStore(
            board_directory,
            board_id=archive_ref.board_id,
            lock=lock,
            lkb_root=self._home / "lkb",
        )
        with lock:
            marker = tombstone_path(self._home / "lkb", archive_ref.board_id)
            if marker.is_file():
                raise BoardTombstonedError(archive_ref.board_id, marker)
            board_directory.mkdir(parents=True, exist_ok=True)
            if (board_directory / "board.json").exists():
                raise FileExistsError(f"Board already exists: {board_directory / 'board.json'}")
            store._write_atomic(restored)
        with self._cache_lock:
            self._board_cache[archive_ref.board_id] = (lock, store)


_repository_singleton: JsonFileLkbRepository | None = None
_repository_lock = threading.Lock()


def get_repository(*, home: Path | None = None) -> JsonFileLkbRepository:
    """Get or create the repository singleton.

    Parameters
    ----------
    home:
        Optional home directory override.  When provided, returns a
        repository rooted at *home* (may not be the global singleton).
        When None, returns the global singleton (or creates it).

    The global singleton uses the standard home resolution precedence
    (explicit param > ``CLAWCODEX_HOME`` env > ``Path.home()``).
    """
    if home is not None:
        # Specific home requested — return a fresh instance
        return JsonFileLkbRepository(home=home)

    global _repository_singleton
    if _repository_singleton is None:
        with _repository_lock:
            if _repository_singleton is None:
                _repository_singleton = JsonFileLkbRepository()
    return _repository_singleton
