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

"""Crash-recoverable Board lifecycle, immutable archives, purge, and GC."""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic_file import atomic_write_json
from .commands import CommandResult
from .error_codes import LkbErrorCode
from .ir_hash import canonical_hash
from .json_store import BoardEnvelope, JsonBoardStore, validate_board_envelope
from .lifecycle_paths import lkb_root_for_store, safe_chain

VALID_STATES = ("active", "closed", "archiving", "archived", "trashed", "purging")
_TRANSITIONS: dict[str, frozenset[str]] = {
    "active": frozenset({"closed"}),
    "closed": frozenset({"active", "archiving", "trashed"}),
    "archiving": frozenset({"archived", "closed"}),
    "archived": frozenset({"active", "trashed"}),
    "trashed": frozenset({"active", "purging"}),
    "purging": frozenset({"trashed"}),
}

GC_TEMP_AGE_SECONDS = 24 * 3600
GC_SESSION_ORPHAN_AGE_SECONDS = 7 * 24 * 3600
GC_QUARANTINE_AGE_SECONDS = 30 * 24 * 3600
GC_TOMBSTONE_AGE_SECONDS = 90 * 24 * 3600
ARCHIVE_DIGEST_PREFIX_LEN = 16
_ACTIVE_CLAIM_STATUS = "active"
_ACTIVE_CLAIMS_GUARDED_TARGET_STATES = frozenset({"closed", "archiving", "trashed", "purging"})


class LifecycleError(Exception):
    """Base error for lifecycle protocols."""


class LifecycleTransitionDenied(LifecycleError):
    code = LkbErrorCode.LIFECYCLE_TRANSITION_DENIED

    def __init__(self, board_id: str, from_state: str, to_state: str, reason: str) -> None:
        self.board_id = board_id
        self.from_state = from_state
        self.to_state = to_state
        self.transition_reason = reason
        super().__init__(f"Lifecycle transition denied for board {board_id!r}: {from_state} -> {to_state} ({reason})")


@dataclass
class LifecycleData:
    state: str = "active"
    scope: str = "project"
    created_at: str = ""
    updated_at: str = ""
    closed_at: str = ""
    archived_at: str = ""
    retention_policy: str = "default"
    origin_project_uri: str = ""

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(f"invalid lifecycle state: {self.state!r}")
        if self.scope not in {"project", "session"}:
            raise ValueError(f"invalid lifecycle scope: {self.scope!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "scope": self.scope,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "archived_at": self.archived_at,
            "retention_policy": self.retention_policy,
            "origin_project_uri": self.origin_project_uri,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LifecycleData:
        return cls(
            state=str(value.get("state", "active")),
            scope=str(value.get("scope", "project")),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            closed_at=str(value.get("closed_at", "")),
            archived_at=str(value.get("archived_at", "")),
            retention_policy=str(value.get("retention_policy", "default")),
            origin_project_uri=str(value.get("origin_project_uri", "")),
        )


def genesis_lifecycle(
    *,
    scope: str,
    created_at: str,
    origin_project_uri: str,
    retention_policy: str = "default",
) -> dict[str, Any]:
    return LifecycleData(
        scope=scope,
        created_at=created_at,
        updated_at=created_at,
        retention_policy=retention_policy,
        origin_project_uri=origin_project_uri,
    ).to_dict()


def board_lifecycle_state(envelope: BoardEnvelope) -> str:
    return str((envelope.lifecycle or {}).get("state", "active"))


def ordinary_write_denial_reason(envelope: BoardEnvelope) -> str | None:
    state = board_lifecycle_state(envelope)
    if state == "active":
        return None
    if state in {"closed", "archived", "trashed"}:
        return f"board is {state}; ordinary writes are disabled"
    if state in {"archiving", "purging"}:
        return f"board lifecycle transition is in progress ({state})"
    return f"unknown lifecycle state {state!r}; refusing write"


def ordinary_write_allowed(envelope: BoardEnvelope) -> bool:
    return ordinary_write_denial_reason(envelope) is None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_active_claims(envelope: BoardEnvelope) -> bool:
    return any(claim.get("status") == _ACTIVE_CLAIM_STATUS for claim in envelope.claims.values())


def transition(
    envelope: BoardEnvelope,
    to_state: str,
    *,
    actor: str,
    reason: str | None = None,
    override_active_claims: bool = False,
) -> BoardEnvelope:
    """Pure state transition; persistence is owned by the caller."""
    from_state = board_lifecycle_state(envelope)
    if to_state not in VALID_STATES:
        raise LifecycleTransitionDenied(envelope.board_id(), from_state, to_state, f"invalid target state {to_state!r}")
    if from_state == to_state:
        return envelope
    if to_state not in _TRANSITIONS.get(from_state, frozenset()):
        raise LifecycleTransitionDenied(envelope.board_id(), from_state, to_state, "transition is not allowed")
    if to_state in _ACTIVE_CLAIMS_GUARDED_TARGET_STATES and _has_active_claims(envelope):
        if not override_active_claims or not reason:
            raise LifecycleTransitionDenied(
                envelope.board_id(),
                from_state,
                to_state,
                "active claims require an authorized override and non-empty reason",
            )

    result = envelope.clone()
    lifecycle = LifecycleData.from_dict(result.lifecycle)
    now = _now_iso()
    lifecycle.state = to_state
    lifecycle.updated_at = now
    if to_state == "closed":
        lifecycle.closed_at = now
    elif to_state == "archived":
        lifecycle.archived_at = now
    elif to_state == "active":
        lifecycle.closed_at = ""
        lifecycle.archived_at = ""
    extras = {key: copy.deepcopy(value) for key, value in result.lifecycle.items() if key not in lifecycle.to_dict()}
    result.lifecycle = lifecycle.to_dict()
    result.lifecycle.update(extras)
    event: dict[str, Any] = {
        "type": "lifecycle_transition",
        "from_state": from_state,
        "to_state": to_state,
        "actor": actor,
        "timestamp": now,
    }
    if reason:
        event["reason"] = reason
    if override_active_claims:
        event["override_active_claims"] = True
    result.events.append(event)
    return result


def _execute_lifecycle(
    store: JsonBoardStore,
    board_id: str,
    command_id: str,
    request_hash: str,
    mutate: Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]],
    *,
    actor: str,
    reason: str | None,
) -> CommandResult:
    return store.execute_atomic(
        board_id,
        command_id,
        request_hash,
        None,
        mutate,
        actor=actor,
        reason=reason,
        lifecycle_operation=True,
    )


def _simple_mutator(
    state: str,
    *,
    actor: str,
    command_id: str,
    reason: str | None,
    override_active_claims: bool = False,
) -> Callable[[BoardEnvelope], tuple[BoardEnvelope, CommandResult]]:
    def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        candidate = transition(
            envelope,
            state,
            actor=actor,
            reason=reason,
            override_active_claims=override_active_claims,
        )
        return candidate, CommandResult(decision="committed", command_id=command_id, reason=reason)

    return mutate


def close_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str | None = None,
    override_active_claims: bool = False,
) -> CommandResult:
    return _execute_lifecycle(
        store,
        board_id,
        command_id,
        request_hash,
        _simple_mutator(
            "closed",
            actor=actor,
            command_id=command_id,
            reason=reason,
            override_active_claims=override_active_claims,
        ),
        actor=actor,
        reason=reason,
    )


def reopen_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str | None = None,
) -> CommandResult:
    return _execute_lifecycle(
        store,
        board_id,
        command_id,
        request_hash,
        _simple_mutator("active", actor=actor, command_id=command_id, reason=reason),
        actor=actor,
        reason=reason,
    )


def trash_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str | None = None,
) -> CommandResult:
    return _execute_lifecycle(
        store,
        board_id,
        command_id,
        request_hash,
        _simple_mutator("trashed", actor=actor, command_id=command_id, reason=reason),
        actor=actor,
        reason=reason,
    )


def _verify_envelope_hash(data: dict[str, Any]) -> bool:
    integrity = data.get("integrity")
    if not isinstance(integrity, dict) or not integrity.get("payloadHash"):
        return False
    payload = {key: value for key, value in data.items() if key != "integrity"}
    return canonical_hash(payload) == integrity["payloadHash"]


def read_archive(path: Path | str, *, expected_board_id: str) -> dict[str, Any]:
    """Read one immutable ``lkb-archive-v1`` wrapper and verify all links."""
    archive_path = Path(path)
    try:
        data = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"archive is unreadable: {archive_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("archiveFormat") != "lkb-archive-v1":
        raise LifecycleError("unsupported archive format")
    if data.get("schemaVersion") != 1:
        raise LifecycleError(f"unsupported archive schema {data.get('schemaVersion')!r}")
    if data.get("boardId") != expected_board_id:
        raise LifecycleError("archive Board ID mismatch")
    expected_hash = data.get("payloadHash")
    payload = {key: value for key, value in data.items() if key != "payloadHash"}
    if not expected_hash or canonical_hash(payload) != expected_hash:
        raise LifecycleError("archive payload hash mismatch")
    envelope = data.get("envelope")
    if not isinstance(envelope, dict) or not _verify_envelope_hash(envelope):
        raise LifecycleError("archive envelope hash mismatch")
    if envelope.get("board", {}).get("board_id") != expected_board_id:
        raise LifecycleError("archive envelope Board ID mismatch")
    if envelope.get("storeRevision") != data.get("sourceStoreRevision"):
        raise LifecycleError("archive source revision mismatch")
    if envelope.get("schemaVersion") != data.get("boardSchemaVersion"):
        raise LifecycleError("archive board schema mismatch")
    if envelope.get("integrity", {}).get("payloadHash") != data.get("sourcePayloadHash"):
        raise LifecycleError("archive source payload hash mismatch")
    try:
        validate_board_envelope(envelope, board_id=expected_board_id, verify_hash=True)
    except Exception as exc:
        raise LifecycleError(f"archive envelope is invalid: {exc}") from exc
    return data


def _hit(store: JsonBoardStore, name: str) -> None:
    failpoint = getattr(store, "_failpoint", None)
    if failpoint is not None:
        failpoint.hit(name)


def _archive_path(store: JsonBoardStore, board_id: str, source: BoardEnvelope, operation_id: str) -> Path:
    from .board_resolver import safe_board_id

    board_dir = store.board_dir
    digest = str(source.integrity["payloadHash"]).split(":", 1)[-1][:ARCHIVE_DIGEST_PREFIX_LEN]
    filename = f"r{source.store_revision:020d}-{digest}-{operation_id}.json"
    return board_dir.parent.parent / "archives" / safe_board_id(board_id) / filename


def _publish_archive(
    store: JsonBoardStore,
    board_id: str,
    source: BoardEnvelope,
    operation_id: str,
    *,
    actor: str,
    reason: str | None,
    created_at: str,
) -> tuple[Path, str]:
    source_data = source.to_dict()
    if not _verify_envelope_hash(source_data):
        raise LifecycleError("source envelope hash is invalid")
    path = _archive_path(store, board_id, source, operation_id)
    document: dict[str, Any] = {
        "archiveFormat": "lkb-archive-v1",
        "schemaVersion": 1,
        "boardSchemaVersion": source.schema_version,
        "boardId": board_id,
        "sourceStoreRevision": source.store_revision,
        "sourcePayloadHash": source.integrity["payloadHash"],
        "archiveId": operation_id,
        "createdAt": created_at,
        "createdBy": actor,
        "reason": reason or "",
        "envelope": source_data,
    }
    archive_hash = canonical_hash(document)
    document["payloadHash"] = archive_hash
    root = lkb_root_for_store(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not safe_chain(root, path.parent) or (path.exists() and not safe_chain(root, path)):
        raise LifecycleError(f"unsafe archive publication path: {path}")
    if path.exists():
        existing = read_archive(path, expected_board_id=board_id)
        if existing.get("payloadHash") != archive_hash:
            raise LifecycleError("immutable archive path already contains different content")
        return path, archive_hash
    _hit(store, "archive_before_publish")
    atomic_write_json(
        path,
        document,
        fsync_dir=True,
        failpoint=getattr(store, "_failpoint", None),
        payload_hash_key="payloadHash",
    )
    read_archive(path, expected_board_id=board_id)
    _hit(store, "archive_after_publish")
    return path, archive_hash


def archive_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str | None = None,
) -> CommandResult:
    """Persist ``archiving``, publish an immutable snapshot, then CAS archived."""
    current = store.load()
    state = board_lifecycle_state(current)
    if state == "archived":
        info = current.lifecycle.get("archive_info")
        if not isinstance(info, dict):
            raise LifecycleError("archived board has no archive provenance")
        archive_path = info.get("archive_path")
        if not isinstance(archive_path, str) or not archive_path.strip():
            raise LifecycleError("archived board has invalid archive provenance path")
        archive = read_archive(Path(archive_path.strip()), expected_board_id=board_id)
        if archive["payloadHash"] != info.get("archive_hash"):
            raise LifecycleError("archived board provenance hash mismatch")
        return CommandResult(
            decision="committed",
            command_id=command_id,
            reason=str(info.get("reason", reason or "")),
        )
    if state not in {"closed", "archiving"}:
        raise LifecycleTransitionDenied(board_id, state, "archiving", "board must be closed")

    if state == "closed":
        operation_id = uuid.uuid4().hex

        def prepare(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            candidate = transition(envelope, "archiving", actor=actor, reason=reason)
            candidate.lifecycle["archive_operation"] = {
                "archive_id": operation_id,
                "actor": actor,
                "reason": reason or "",
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
        _hit(store, "archive_after_prepare")

    source = store.load()
    if board_lifecycle_state(source) != "archiving" or _has_active_claims(source):
        raise LifecycleTransitionDenied(board_id, board_lifecycle_state(source), "archived", "archive is not quiescent")
    operation = source.lifecycle.get("archive_operation")
    if not isinstance(operation, dict) or not operation.get("archive_id"):
        raise LifecycleError("archiving board has no resumable archive operation")
    operation_id = str(operation["archive_id"])
    operation_actor = str(operation.get("actor", actor))
    operation_reason = str(operation.get("reason", reason or ""))
    archive_path, archive_hash = _publish_archive(
        store,
        board_id,
        source,
        operation_id,
        actor=operation_actor,
        reason=operation_reason,
        created_at=str(operation.get("started_at", "")),
    )
    source_revision = source.store_revision
    source_hash = str(source.integrity["payloadHash"])

    def commit(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        actual_operation = envelope.lifecycle.get("archive_operation")
        if (
            board_lifecycle_state(envelope) != "archiving"
            or envelope.store_revision != source_revision
            or envelope.integrity.get("payloadHash") != source_hash
            or not isinstance(actual_operation, dict)
            or actual_operation.get("archive_id") != operation_id
        ):
            raise LifecycleError("archive CAS failed; verified immutable archive retained")
        candidate = transition(envelope, "archived", actor=operation_actor, reason=operation_reason)
        candidate.lifecycle.pop("archive_operation", None)
        candidate.lifecycle["archive_info"] = {
            "archive_id": operation_id,
            "archive_path": str(archive_path),
            "archive_hash": archive_hash,
            "source_store_revision": source_revision,
            "source_payload_hash": source_hash,
            "archived_by": operation_actor,
            "archived_at": candidate.lifecycle["archived_at"],
            "reason": operation_reason,
        }
        return candidate, CommandResult(decision="committed", command_id=command_id, reason=operation_reason)

    return _execute_lifecycle(
        store,
        board_id,
        f"{command_id}:commit",
        f"{request_hash}:commit:{archive_hash}",
        commit,
        actor=operation_actor,
        reason=operation_reason,
    )


def _archive_ref_values(archive_ref: Any) -> tuple[str, Path, int, str]:
    try:
        return (
            str(archive_ref.board_id),
            Path(archive_ref.archive_path),
            int(archive_ref.store_revision),
            str(archive_ref.payload_hash),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise LifecycleError("restore requires a valid ArchiveRef") from exc


def restore_board(
    store: JsonBoardStore,
    board_id: str,
    *,
    archive_ref: Any,
    actor: str,
    command_id: str,
    request_hash: str,
    reason: str | None = None,
) -> CommandResult:
    """Restore the explicitly selected immutable ArchiveRef into this Board."""
    ref_board_id, path, ref_revision, ref_hash = _archive_ref_values(archive_ref)
    if ref_board_id != board_id:
        raise LifecycleError("ArchiveRef Board ID mismatch")
    archive = read_archive(path, expected_board_id=board_id)
    if ref_revision and ref_revision != int(archive["sourceStoreRevision"]):
        raise LifecycleError("ArchiveRef revision mismatch")
    if ref_hash and ref_hash != str(archive["payloadHash"]):
        raise LifecycleError("ArchiveRef payload hash mismatch")
    source = BoardEnvelope.from_dict(archive["envelope"])

    def mutate(envelope: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
        state = board_lifecycle_state(envelope)
        if state not in {"archived", "trashed", "closed"}:
            raise LifecycleTransitionDenied(board_id, state, "active", "restore target is not idle")
        candidate = transition(envelope, "active", actor=actor, reason=reason)
        for name in (
            "graphs",
            "nodes",
            "edges",
            "claims",
            "validation_runs",
            "history_segments",
        ):
            setattr(candidate, name, copy.deepcopy(getattr(source, name)))
        candidate.lifecycle["restore_info"] = {
            "source_archive_id": archive["archiveId"],
            "source_archive_hash": archive["payloadHash"],
            "source_store_revision": archive["sourceStoreRevision"],
            "source_archive_path": str(path),
            "restored_by": actor,
            "restored_at": _now_iso(),
        }
        return candidate, CommandResult(decision="committed", command_id=command_id, reason=reason)

    return _execute_lifecycle(
        store,
        board_id,
        command_id,
        request_hash,
        mutate,
        actor=actor,
        reason=reason,
    )
