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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402
"""Crash-recoverable purge and tombstone lifecycle operations."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from .atomic_file import atomic_write_json
from .commands import CommandResult
from .ir_hash import canonical_hash
from .json_store import BoardEnvelope, JsonBoardStore
from .lifecycle_core import (
    LifecycleError,
    LifecycleTransitionDenied,
    _execute_lifecycle,
    _has_active_claims,
    _hit,
    _now_iso,
    board_lifecycle_state,
    read_archive,
    transition,
)
from .lifecycle_paths import _lkb_root_for_store, _safe_chain


def tombstone_path(root: Path | str, board_id: str) -> Path:
    from .board_resolver import safe_board_id

    return Path(root) / "tombstones" / f"{safe_board_id(board_id)}.json"


def read_tombstone(path: Path | str, *, expected_board_id: str) -> dict[str, Any]:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"tombstone is unreadable: {target}: {exc}") from exc
    if not isinstance(data, dict) or data.get("tombstoneFormat") != "lkb-tombstone-v1":
        raise LifecycleError("unsupported tombstone format")
    if data.get("schemaVersion") != 1:
        raise LifecycleError(f"unsupported tombstone schema {data.get('schemaVersion')!r}")
    if data.get("boardId") != expected_board_id:
        raise LifecycleError("tombstone Board ID mismatch")
    expected = data.get("payloadHash")
    payload = {key: value for key, value in data.items() if key != "payloadHash"}
    if not expected or canonical_hash(payload) != expected:
        raise LifecycleError("tombstone payload hash mismatch")
    return data


def _tombstone_document(
    board_id: str,
    purging: BoardEnvelope,
    operation_id: str,
    *,
    actor: str,
    reason: str,
    purged_at: str,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "tombstoneFormat": "lkb-tombstone-v1",
        "schemaVersion": 1,
        "boardId": board_id,
        "purgeId": operation_id,
        "sourceStoreRevision": purging.store_revision,
        "sourcePayloadHash": purging.integrity["payloadHash"],
        "purgedBy": actor,
        "purgedAt": purged_at,
        "reason": reason,
    }
    document["payloadHash"] = canonical_hash(document)
    return document


def _purge_pending_path(root: Path, board_id: str) -> Path:
    from .board_resolver import safe_board_id

    return root / "tombstones" / f".{safe_board_id(board_id)}.purge-pending"


def _read_purge_pending(path: Path, *, expected_board_id: str) -> dict[str, Any]:
    try:
        pending = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"purge journal is unreadable: {path}: {exc}") from exc
    if (
        not isinstance(pending, dict)
        or pending.get("pendingFormat") != "lkb-purge-pending-v1"
        or pending.get("boardId") != expected_board_id
    ):
        raise LifecycleError("invalid purge journal")
    expected = pending.get("payloadHash")
    payload = {key: value for key, value in pending.items() if key != "payloadHash"}
    if not expected or canonical_hash(payload) != expected:
        raise LifecycleError("purge journal payload hash mismatch")
    tombstone = pending.get("tombstone")
    if not isinstance(tombstone, dict):
        raise LifecycleError("purge journal has no tombstone candidate")
    tombstone_payload = {key: value for key, value in tombstone.items() if key != "payloadHash"}
    if (
        tombstone.get("boardId") != expected_board_id
        or tombstone.get("tombstoneFormat") != "lkb-tombstone-v1"
        or canonical_hash(tombstone_payload) != tombstone.get("payloadHash")
    ):
        raise LifecycleError("purge journal tombstone candidate is invalid")
    return pending


def _write_purge_pending(
    store: JsonBoardStore,
    board_id: str,
    tombstone: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = _lkb_root_for_store(store)
    path = _purge_pending_path(root, board_id)
    pending: dict[str, Any] = {
        "pendingFormat": "lkb-purge-pending-v1",
        "schemaVersion": 1,
        "boardId": board_id,
        "purgeId": tombstone["purgeId"],
        "tombstone": tombstone,
    }
    pending["payloadHash"] = canonical_hash(pending)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _safe_chain(root, path.parent) or (path.exists() and not _safe_chain(root, path)):
        raise LifecycleError(f"unsafe purge journal path: {path}")
    if path.exists():
        existing = _read_purge_pending(path, expected_board_id=board_id)
        if existing != pending:
            raise LifecycleError("another purge journal already owns this Board ID")
        return path, existing
    atomic_write_json(
        path,
        pending,
        fsync_dir=True,
        failpoint=getattr(store, "_failpoint", None),
        payload_hash_key="payloadHash",
    )
    return path, _read_purge_pending(path, expected_board_id=board_id)


def _publish_tombstone(
    store: JsonBoardStore,
    board_id: str,
    document: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = _lkb_root_for_store(store)
    path = tombstone_path(root, board_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _safe_chain(root, path.parent) or (path.exists() and not _safe_chain(root, path)):
        raise LifecycleError(f"unsafe tombstone publication path: {path}")
    if path.exists():
        existing = read_tombstone(path, expected_board_id=board_id)
        expected_links = {
            key: document[key]
            for key in (
                "purgeId",
                "sourceStoreRevision",
                "sourcePayloadHash",
                "purgedBy",
                "reason",
            )
        }
        if any(existing.get(key) != value for key, value in expected_links.items()):
            raise LifecycleError("another purge tombstone already owns this Board ID")
        return path, existing
    _hit(store, "purge_before_tombstone")
    atomic_write_json(
        path,
        document,
        fsync_dir=True,
        failpoint=getattr(store, "_failpoint", None),
        payload_hash_key="payloadHash",
    )
    verified = read_tombstone(path, expected_board_id=board_id)
    _hit(store, "purge_after_tombstone")
    return path, verified


def _remove_managed_purge_data(
    store: JsonBoardStore,
    board_id: str,
    operation_id: str,
    pending: dict[str, Any],
    *,
    active_watchers: int,
) -> None:
    board_dir = Path(getattr(store, "_board_dir"))
    root = _lkb_root_for_store(store)
    if active_watchers or int(getattr(store, "_active_watchers", 0)):
        raise LifecycleTransitionDenied(board_id, "purging", "purged", "active watcher prevents purge")
    with getattr(store, "_lock"):
        envelope = getattr(store, "_load_locked")()
        operation = envelope.lifecycle.get("purge_operation")
        if (
            board_lifecycle_state(envelope) != "purging"
            or not isinstance(operation, dict)
            or operation.get("purge_id") != operation_id
            or _has_active_claims(envelope)
        ):
            raise LifecycleError("purge state changed before managed deletion")
        marker = pending["tombstone"]
        if (
            marker.get("purgeId") != operation_id
            or marker.get("sourceStoreRevision") != envelope.store_revision
            or marker.get("sourcePayloadHash") != envelope.integrity.get("payloadHash")
        ):
            raise LifecycleError("purge journal no longer matches managed data")
        if not _safe_chain(root, board_dir, descendants=True):
            raise LifecycleError("unsafe board path prevents purge")
        archive_dir = root / "archives"
        from .board_resolver import safe_board_id

        board_archives = archive_dir / safe_board_id(board_id)
        if board_archives.exists():
            if not _safe_chain(root, board_archives, descendants=True):
                raise LifecycleError("unsafe archive path prevents purge")
            _verify_archive_tree(board_archives, board_id=board_id)
        _hit(store, "purge_before_delete")
        entries = sorted(board_dir.iterdir(), key=lambda item: (item.name == "board.json", item.name))
        for entry in entries:
            if entry.name in {".lock", ".lock.owner.json"}:
                continue
            if not _safe_chain(root, entry, descendants=entry.is_dir()):
                raise LifecycleError(f"unsafe managed purge target: {entry}")
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        if board_archives.is_dir():
            shutil.rmtree(board_archives)
        _hit(store, "purge_after_delete")
    # The permanent .lock anchor is deliberately retained.  The owner file
    # is removed by BoardFileLock while it still owns the same anchor inode.


def _verify_archive_tree(path: Path, *, board_id: str) -> None:
    for entry in path.iterdir():
        if entry.name == ".tmp" and entry.is_dir():
            if any(entry.iterdir()):
                raise LifecycleError("unverified archive temporary files prevent purge")
            continue
        if not entry.is_file() or entry.suffix != ".json":
            raise LifecycleError(f"unverified archive entry prevents purge: {entry}")
        read_archive(entry, expected_board_id=board_id)


def _finish_tombstoned_cleanup(
    store: JsonBoardStore,
    board_id: str,
    marker: Path | None,
    *,
    active_watchers: int,
) -> None:
    """Resume cleanup when a crash removed board.json before other data."""
    if active_watchers or int(getattr(store, "_active_watchers", 0)):
        raise LifecycleTransitionDenied(board_id, "purged", "purged", "active watcher prevents purge recovery")
    board_dir = Path(getattr(store, "_board_dir"))
    root = _lkb_root_for_store(store)
    if marker is not None:
        read_tombstone(marker, expected_board_id=board_id)
    with getattr(store, "_lock"):
        if not _safe_chain(root, board_dir, descendants=True):
            raise LifecycleError("unsafe board path prevents purge recovery")
        for entry in list(board_dir.iterdir()):
            if entry.name in {".lock", ".lock.owner.json"}:
                continue
            if not _safe_chain(root, entry, descendants=entry.is_dir()):
                raise LifecycleError(f"unsafe purge recovery target: {entry}")
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        from .board_resolver import safe_board_id

        archive_dir = root / "archives" / safe_board_id(board_id)
        if archive_dir.is_dir():
            if not _safe_chain(root, archive_dir, descendants=True):
                raise LifecycleError("unsafe archive path prevents purge recovery")
            _verify_archive_tree(archive_dir, board_id=board_id)
            shutil.rmtree(archive_dir)


def purge_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str,
    confirm: str,
    authorized: bool = False,
    active_watchers: int = 0,
) -> CommandResult:
    """Two-phase purge ending only after tombstone publication and data removal."""
    if confirm != board_id:
        raise ValueError("purge confirmation must exactly match board_id")
    if not reason.strip():
        raise ValueError("purge reason is required")
    if not authorized:
        raise PermissionError("purge requires independent administrative permission")
    root = _lkb_root_for_store(store)
    board_directory = Path(getattr(store, "_board_dir"))
    if not _safe_chain(root, board_directory, descendants=True):
        raise LifecycleTransitionDenied(
            board_id,
            "unknown",
            "purging",
            "unsafe board path prevents purge",
        )
    marker = tombstone_path(root, board_id)
    pending_path = _purge_pending_path(root, board_id)
    if not store.exists():
        if marker.is_file():
            read_tombstone(marker, expected_board_id=board_id)
            _finish_tombstoned_cleanup(
                store,
                board_id,
                marker,
                active_watchers=active_watchers,
            )
            pending_path.unlink(missing_ok=True)
            return CommandResult(decision="committed", command_id=command_id, reason=reason)
        pending = _read_purge_pending(pending_path, expected_board_id=board_id)
        _finish_tombstoned_cleanup(
            store,
            board_id,
            None,
            active_watchers=active_watchers,
        )
        marker, tombstone = _publish_tombstone(
            store,
            board_id,
            pending["tombstone"],
        )
        pending_path.unlink(missing_ok=True)
        return CommandResult(
            decision="committed",
            command_id=command_id,
            reason=str(tombstone["reason"]),
            derived_facts=(f"tombstone:{marker}",),
        )
    if marker.is_file():
        read_tombstone(marker, expected_board_id=board_id)
        with getattr(store, "_lock"):
            current = getattr(store, "_load_locked")()
    else:
        current = store.load()
    state = board_lifecycle_state(current)
    if active_watchers or int(getattr(store, "_active_watchers", 0)):
        raise LifecycleTransitionDenied(board_id, state, "purging", "active watcher prevents purge")
    if _has_active_claims(current):
        raise LifecycleTransitionDenied(board_id, state, "purging", "active claims exist")
    archive_info = current.lifecycle.get("archive_info")
    if archive_info is not None:
        if not isinstance(archive_info, dict) or not archive_info.get("archive_path"):
            raise LifecycleTransitionDenied(board_id, state, "purging", "archive has not been verified")
        archive = read_archive(Path(str(archive_info["archive_path"])), expected_board_id=board_id)
        if archive.get("payloadHash") != archive_info.get("archive_hash"):
            raise LifecycleTransitionDenied(board_id, state, "purging", "archive provenance does not verify")
    from .board_resolver import safe_board_id

    archive_tree = root / "archives" / safe_board_id(board_id)
    if archive_tree.exists():
        if not _safe_chain(root, archive_tree, descendants=True):
            raise LifecycleTransitionDenied(board_id, state, "purging", "unsafe archive path prevents purge")
        try:
            _verify_archive_tree(archive_tree, board_id=board_id)
        except LifecycleError as exc:
            raise LifecycleTransitionDenied(
                board_id, state, "purging", f"unverified archive prevents purge: {exc}"
            ) from exc

    if state == "trashed":
        operation_id = uuid.uuid4().hex

        def prepare(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            candidate = transition(envelope, "purging", actor=actor, reason=reason)
            candidate.lifecycle["purge_operation"] = {
                "purge_id": operation_id,
                "actor": actor,
                "reason": reason,
                "started_at": _now_iso(),
            }
            return candidate, CommandResult(decision="committed", command_id=f"{command_id}:prepare", reason=reason)

        _execute_lifecycle(
            store,
            board_id,
            f"{command_id}:prepare",
            f"{request_hash}:prepare",
            prepare,
            actor=actor,
            reason=reason,
        )
        _hit(store, "purge_after_prepare")
    elif state != "purging":
        raise LifecycleTransitionDenied(board_id, state, "purging", "board must be trashed")

    if marker.is_file():
        with getattr(store, "_lock"):
            purging = getattr(store, "_load_locked")()
    else:
        purging = store.load()
    operation = purging.lifecycle.get("purge_operation")
    if not isinstance(operation, dict) or not operation.get("purge_id"):
        raise LifecycleError("purging board has no resumable purge operation")
    operation_id = str(operation["purge_id"])
    operation_actor = str(operation.get("actor", actor))
    operation_reason = str(operation.get("reason", reason))
    with getattr(store, "_lock"):
        latest = getattr(store, "_load_locked")()
        latest_operation = latest.lifecycle.get("purge_operation")
        if (
            board_lifecycle_state(latest) != "purging"
            or not isinstance(latest_operation, dict)
            or latest_operation.get("purge_id") != operation_id
        ):
            raise LifecycleError("purge state changed before managed deletion")
        tombstone_document = _tombstone_document(
            board_id,
            latest,
            operation_id,
            actor=operation_actor,
            reason=operation_reason,
            purged_at=str(latest_operation.get("started_at", "")),
        )
        pending_path, pending = _write_purge_pending(
            store,
            board_id,
            tombstone_document,
        )
    _hit(store, "purge_after_pending")
    _remove_managed_purge_data(
        store,
        board_id,
        operation_id,
        pending,
        active_watchers=active_watchers,
    )
    marker, _ = _publish_tombstone(
        store,
        board_id,
        tombstone_document,
    )
    pending_path.unlink(missing_ok=True)
    return CommandResult(
        decision="committed",
        command_id=command_id,
        reason=operation_reason,
        derived_facts=(f"tombstone:{marker}",),
    )
