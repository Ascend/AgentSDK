#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# -------------------------------------------------------------------------
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""Session storage — JSONL transcript recording matching TypeScript session/storage.ts.

Provides:
- SessionStorage: write/read JSONL transcripts with metadata
- Atomic writes, content replacement for large tool results
- Flush control, session listing, cleanup
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.types.messages import Message, message_from_dict, message_to_dict
from src.utils.clawcodex_dirs import get_sessions_dir

logger = logging.getLogger(__name__)


# F-125 C13: cross-process file lock for JSONL append writes.
# ``fcntl.flock`` is available on POSIX (Linux/macOS/WSL). On Windows
# the import fails and we degrade to unlocked writes — the same
# behaviour as before this change. The lock is held for the duration
# of a single append batch so concurrent processes serialise their
# writes rather than interleaving lines.
try:
    import fcntl as _fcntl  # type: ignore[import-not-found]

    _HAS_FCNTL = True
except ImportError:
    _fcntl = None
    _HAS_FCNTL = False


@contextmanager
def _locked_append(file_path: Path) -> Iterator[Any]:
    """Context manager that acquires an exclusive ``flock`` on *file_path*.

    Opens the file in append mode, acquires ``LOCK_EX`` (blocking —
    waits for other holders to release), and yields the file object.
    On POSIX the lock is released automatically when the file object
    is closed by the ``with`` block. On Windows (no ``fcntl``) the
    file is opened without a lock — callers get the same append
    semantics but without cross-process serialisation.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(file_path, "a", encoding="utf-8")  # noqa: SIM115
    try:
        if _HAS_FCNTL and _fcntl is not None:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            except OSError:
                # Locking can fail on some filesystems (e.g. network
                # mounts). Degrade to unlocked append rather than
                # failing the write.
                pass
        yield fh
    finally:
        try:
            if _HAS_FCNTL and _fcntl is not None:
                try:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            fh.close()


# Canonical default honors CLAWCODEX_CONFIG_DIR. The downstream-specific
# CLAWCODEX_SESSIONS_DIR override remains available via resolve_sessions_dir.
SESSIONS_DIR = get_sessions_dir()
CONTENT_DIR_NAME = "content"

# Thresholds
LARGE_CONTENT_THRESHOLD = 10_000  # 10KB — store separately
DEFAULT_RETENTION_DAYS = 30
MAX_FLUSH_BATCH = 50

# F-11: sessionStorage 容量限制
# Prevents unbounded memory growth from tracking too many session
# directories in long-running daemon/swarm processes.
MAX_CACHED_SESSION_FILES = 1000

# Module-level LRU cache: session_id → session_dir path.
# Entries are registered when a SessionStorage is initialized, and
# evicted in FIFO order when the cap is reached.
_session_file_cache: OrderedDict[str, Path] = OrderedDict()


def resolve_sessions_dir(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("CLAWCODEX_SESSIONS_DIR", "")).strip()
    if override:
        return Path(override).expanduser()
    return SESSIONS_DIR


def register_session_file(session_id: str, dir_path: Path) -> None:
    """Register a session directory in the LRU cache.

    When the cache exceeds MAX_CACHED_SESSION_FILES, the oldest
    entry is evicted.  This is a no-op if *session_id* is already
    registered (its insertion order is not promoted).
    """
    if session_id in _session_file_cache:
        return
    if len(_session_file_cache) >= MAX_CACHED_SESSION_FILES:
        _session_file_cache.popitem(last=False)
    _session_file_cache[session_id] = dir_path


def get_cached_session_dirs() -> dict[str, Path]:
    """Return a copy of the current cache (session_id → dir_path)."""
    return dict(_session_file_cache)


def clear_session_cache() -> None:
    """Clear the entire cache (e.g. during testing or config reload)."""
    _session_file_cache.clear()


@dataclass
class SessionMetadata:
    """Metadata for a session.

    F-49 P5-F: ``cwd``, ``total_cost``, ``last_user_input``, ``agent_name``,
    and ``cost`` are kept as in-memory attributes (for backward compatibility
    with callers that read them) but are no longer **written** to
    ``metadata.json``. The on-disk shape is now limited to list-summary
    fields — full conversation lives in ``transcript.jsonl`` and the cost
    snapshot lives in the last ``session_snapshot`` line of that file.

    ``from_dict`` still tolerates the old extra fields so legacy
    ``metadata.json`` files keep loading without data loss; those fields
    simply round-trip through memory and are dropped on the next save.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    model: str = ""
    title: str = ""
    message_count: int = 0
    last_updated: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    # ---- Legacy fields kept in memory only (not serialized by ``to_dict``) ----
    # These were previously written to ``metadata.json`` and are still
    # tolerated on read via ``from_dict`` for backward compatibility.
    cwd: str = ""
    total_cost: float = 0.0
    last_user_input: str = ""
    agent_name: str = ""  # S-R4-A: agent type used for the session
    cost: dict[str, Any] | None = None  # Full cost block (matches _snapshot_cost_block shape)

    # Fields that ``to_dict`` will serialize. Anything not in this set is
    # kept in memory only — see P5-F in docs/FEATURE_PLAN.md.
    _SERIALIZED_FIELDS: tuple[str, ...] = field(
        default=(
            "session_id",
            "start_time",
            "model",
            "title",
            "message_count",
            "last_updated",
            "tags",
            "last_user_input",
        ),
        init=False,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize session metadata for disk persistence.

        ``last_user_input`` is included here (despite F-49 P5-F's original
        intent) because the session browser (``/resume``) relies on it for
        display — reading it from the transcript JSONL for every session in
        the listing would be O(n) file reads with unacceptable latency.
        Other legacy fields (``cwd``, ``total_cost``, ``agent_name``,
        ``cost``) remain excluded as they are not needed by the UI.
        """
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "model": self.model,
            "title": self.title,
            "message_count": self.message_count,
            "last_updated": self.last_updated,
            "tags": list(self.tags),
            "last_user_input": self.last_user_input,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        # Legacy fields are tolerated on read so old metadata.json
        # files keep loading. New writes (via to_dict) drop them.
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            start_time=data.get("start_time", time.time()),
            model=data.get("model", ""),
            title=data.get("title", ""),
            message_count=data.get("message_count", 0),
            last_updated=data.get("last_updated", time.time()),
            tags=data.get("tags", []),
            # Legacy fields — kept in memory only.
            cwd=data.get("cwd", ""),
            total_cost=data.get("total_cost", 0.0),
            last_user_input=data.get("last_user_input", ""),
            agent_name=data.get("agent_name", ""),
            cost=data.get("cost"),
        )


class SessionStorage:
    """JSONL-based session transcript storage."""

    def __init__(
        self,
        session_id: str | None = None,
        sessions_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.sessions_dir = sessions_dir or resolve_sessions_dir()
        self._session_dir = self.sessions_dir / self.session_id
        self._transcript_path = self._session_dir / "transcript.jsonl"
        self._metadata_path = self._session_dir / "metadata.json"
        self._content_dir = self._session_dir / CONTENT_DIR_NAME
        self._write_buffer: list[dict[str, Any]] = []
        self._metadata: SessionMetadata | None = None
        # Set of message uuids already flushed to disk — guards against
        # duplicate transcript lines when ``save_to_session_storage``
        # is called multiple times for the same session (e.g. resume
        # paths, agent loop restarts). Lazy-initialised on first flush
        # from the on-disk transcript so a resumed session inherits
        # the de-dup baseline instead of re-appending everything.
        self._flushed_uuids: set[str] | None = None

        # F-11: Register in the session file LRU cache.
        register_session_file(self.session_id, self._session_dir)

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    # --- Metadata ---

    # F-49 P5-F: fields that are kept in memory but no longer written
    # to ``metadata.json``. ``cwd`` is preserved here for caller
    # compatibility (extensions may still pass it); it is simply not
    # persisted. The same applies to total_cost / last_user_input /
    # agent_name / cost — those move to the transcript's
    # ``session_init`` / ``session_snapshot`` lines.
    _LEGACY_METADATA_FIELDS: tuple[str, ...] = (
        "cwd",
        "total_cost",
        "last_user_input",
        "agent_name",
        "cost",
    )

    def init_metadata(
        self,
        *,
        model: str = "",
        cwd: str = "",
        title: str = "",
        tags: list[str] | None = None,
        **legacy: Any,
    ) -> SessionMetadata:
        """Initialize session metadata.

        Idempotent: when ``metadata.json`` already exists on disk
        (e.g. a resumed session or a second ``save_to_session_storage``
        call for the same session_id), the existing ``start_time`` is
        preserved instead of being overwritten with ``time.time()``.
        The same applies to the on-disk ``message_count`` and
        ``last_updated`` — only the caller-supplied fields are written.
        This prevents the clock-skew class of bugs where re-saving a
        session moves its start_time later than the messages it
        contains (see visualizer screenshot repro for
        ``02cba64e-…``).

        F-49 P5-F: ``cwd`` and other legacy metadata fields
        (``total_cost`` / ``last_user_input`` / ``agent_name`` /
        ``cost``) are accepted for caller compatibility but are NOT
        written to ``metadata.json``. They live in the transcript
        JSONL instead (``session_init`` carries ``cwd``, ``session_snapshot``
        carries the cost block). Use ``**legacy`` to silently absorb
        any other deprecated kwarg without TypeError-ing existing
        callers; the values are kept on the in-memory instance but
        not persisted.
        """
        existing = self._load_metadata()
        if existing is not None:
            # Preserve wall-clock anchors from disk; refresh the
            # caller-supplied identity fields only.
            if model:
                existing.model = model
            # Legacy fields are still updated in memory so callers
            # that read ``meta.cwd`` / ``meta.cost`` see the latest
            # value — they're just not written back to disk.
            if cwd:
                existing.cwd = cwd
            for key, value in legacy.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            if title:
                existing.title = title
            if tags:
                existing.tags = tags
            self._metadata = existing
            self._save_metadata()
            return existing
        self._metadata = SessionMetadata(
            session_id=self.session_id,
            model=model,
            cwd=cwd,
            title=title,
            tags=tags or [],
        )
        # Absorb any legacy kwargs onto the in-memory instance
        # without persisting them.
        for key, value in legacy.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
        self._save_metadata()
        return self._metadata

    def get_metadata(self) -> SessionMetadata | None:
        """Get cached or loaded metadata."""
        if self._metadata is not None:
            return self._metadata
        return self._load_metadata()

    def update_metadata(self, **kwargs: Any) -> None:
        """Update metadata fields.

        F-49 P5-F: legacy fields (``cwd``, ``total_cost``,
        ``last_user_input``, ``agent_name``, ``cost``) are still
        updated on the in-memory ``SessionMetadata`` so callers that
        read them keep working, but they are NOT written to
        ``metadata.json``. The next ``_save_metadata`` will use
        ``to_dict()`` which excludes them, so they evaporate on the
        next save. This keeps callers backward compatible while
        preventing the legacy fields from re-entering the on-disk
        format.
        """
        meta = self.get_metadata()
        if meta is None:
            return
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        # ``last_updated`` is still written — it's one of the
        # serialized list-summary fields. Always bump it so
        # list_sessions() ordering reflects the actual save.
        meta.last_updated = time.time()
        self._save_metadata()

    def _save_metadata(self) -> None:
        if self._metadata is None:
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._metadata_path, json.dumps(self._metadata.to_dict(), indent=2))

    def _load_metadata(self) -> SessionMetadata | None:
        if not self._metadata_path.exists():
            return None
        try:
            data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            self._metadata = SessionMetadata.from_dict(data)
            return self._metadata
        except Exception as e:  # noqa: BLE001:
            logger.debug("Failed to load metadata: %s", e)
            return None

    # --- Transcript writing ---

    def write_message(self, message: Message) -> None:
        """Write a message to the transcript buffer."""
        msg_dict = message_to_dict(message)
        msg_dict = self._replace_large_content(msg_dict)
        self._write_buffer.append(msg_dict)
        if len(self._write_buffer) >= MAX_FLUSH_BATCH:
            self.flush()

    def write_raw(self, data: dict[str, Any]) -> None:
        """Write raw dict to transcript buffer."""
        self._write_buffer.append(data)
        if len(self._write_buffer) >= MAX_FLUSH_BATCH:
            self.flush()

    def flush(self) -> None:
        """Flush buffered messages to disk.

        Dedupes by message ``uuid``: if a buffered entry's uuid was
        already flushed (either this session's buffer or the on-disk
        transcript), it is skipped. This prevents the transcript
        doubling seen when ``save_to_session_storage`` is invoked
        twice for the same session — the second call would otherwise
        re-append the entire conversation, producing the duplicate
        lines visible in screenshot repro
        (transcript.jsonl lines 1-6 == 7-12, identical uuids).
        """
        if not self._write_buffer:
            return
        # Lazy-init the seen-uuid set from the on-disk transcript so
        # resumed sessions don't restart from an empty baseline.
        if self._flushed_uuids is None:
            self._flushed_uuids = self._scan_flushed_uuids()
        # Filter buffer, preserving order, and track newly-flushed ids.
        to_write: list[dict[str, Any]] = []
        for entry in self._write_buffer:
            uuid = entry.get("uuid") if isinstance(entry, dict) else None  # pylint: disable=redefined-outer-name
            if uuid and uuid in self._flushed_uuids:
                continue
            to_write.append(entry)
            if uuid:
                self._flushed_uuids.add(uuid)
        if not to_write:
            self._write_buffer.clear()
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        # F-125 C13: hold an exclusive flock for the append batch so
        # two processes resuming the same session don't interleave
        # their JSONL lines. The lock is per-call (acquired+released
        # inside _locked_append); long-held locks would block the
        # agent loop, so we keep the critical section to the file
        # write only.
        with _locked_append(self._transcript_path) as f:
            for entry in to_write:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count = len(to_write)
        self._write_buffer.clear()

        # Update metadata message count
        if self._metadata:
            self._metadata.message_count += count
            self._metadata.last_updated = time.time()
            self._save_metadata()

    def _scan_flushed_uuids(self) -> set[str]:
        """Walk the on-disk transcript and collect already-flushed uuids.

        Tolerant of malformed lines (mirrors ``read_transcript``) so
        a corrupt trailing write doesn't poison the dedup baseline.
        """
        seen: set[str] = set()
        if not self._transcript_path.exists():
            return seen
        try:
            with open(self._transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        uuid = entry.get("uuid")  # pylint: disable=redefined-outer-name
                        if isinstance(uuid, str) and uuid:
                            seen.add(uuid)
        except OSError:
            return seen
        return seen

    # --- Content replacement ---

    def _replace_large_content(self, msg_dict: dict[str, Any]) -> dict[str, Any]:
        """Replace large tool result content with file references."""
        content = msg_dict.get("content")
        if not isinstance(content, list):
            return msg_dict

        replaced = False
        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > LARGE_CONTENT_THRESHOLD:
                    ref_id = str(uuid.uuid4())
                    self._store_content(ref_id, inner)
                    block = dict(block)
                    block["content"] = f"[content stored: {ref_id}]"
                    block["_content_ref"] = ref_id
                    replaced = True

            new_content.append(block)

        if replaced:
            msg_dict = dict(msg_dict)
            msg_dict["content"] = new_content
        return msg_dict

    def _store_content(self, ref_id: str, content: str) -> None:
        """Store large content to a separate file."""
        self._content_dir.mkdir(parents=True, exist_ok=True)
        path = self._content_dir / f"{ref_id}.txt"
        path.write_text(content, encoding="utf-8")

    def load_content(self, ref_id: str) -> str | None:
        """Load stored content by reference ID."""
        path = self._content_dir / f"{ref_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    # --- Transcript reading ---

    def read_transcript(self) -> list[dict[str, Any]]:
        """Read all entries from the JSONL transcript."""
        if not self._transcript_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self._transcript_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Malformed JSONL line %d: %s", line_num, e)
        return entries

    def read_messages(self) -> list[Message]:
        """Read transcript as typed Message objects."""
        entries = self.read_transcript()
        messages: list[Message] = []
        for entry in entries:
            try:
                messages.append(message_from_dict(entry))
            except Exception as e:  # noqa: BLE001:
                logger.warning("Failed to parse message: %s", e)
        return messages

    # --- Session listing & cleanup ---

    @classmethod
    def list_sessions(
        cls,
        sessions_dir: Path | None = None,
        *,
        limit: int = 50,
        tag_filter: str | None = None,
    ) -> list[SessionMetadata]:
        """List recent sessions sorted by last_updated descending.

        When tag_filter is set, only sessions whose tags list contains
        any entry starting with tag_filter are returned.
        """
        base = sessions_dir or resolve_sessions_dir()
        if not base.exists():
            return []

        sessions: list[SessionMetadata] = []
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = SessionMetadata.from_dict(data)
                if tag_filter is not None and not any(t.startswith(tag_filter) for t in meta.tags):
                    continue
                sessions.append(meta)
            except Exception:  # noqa: BLE001, S112
                continue

        sessions.sort(key=lambda s: s.last_updated, reverse=True)
        return sessions[:limit]

    @classmethod
    def cleanup_sessions(
        cls,
        sessions_dir: Path | None = None,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> int:
        """Delete sessions older than retention_days. Returns count deleted."""
        base = sessions_dir or resolve_sessions_dir()
        if not base.exists():
            return 0

        cutoff = time.time() - (retention_days * 86400)
        deleted = 0

        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    last_updated = data.get("last_updated", 0)
                    if last_updated < cutoff:
                        shutil.rmtree(entry, ignore_errors=True)
                        deleted += 1
                        continue
                except Exception:  # noqa: BLE001, S110  # nosec B110
                    pass
            else:
                # No metadata — check directory mtime
                try:
                    if entry.stat().st_mtime < cutoff:
                        shutil.rmtree(entry, ignore_errors=True)
                        deleted += 1
                except Exception:  # noqa: BLE001, S110  # nosec B110
                    pass

        return deleted

    def delete(self) -> None:
        """Delete this session's directory."""
        if self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
        # F-11: Evict from the LRU cache.
        _session_file_cache.pop(self.session_id, None)


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
