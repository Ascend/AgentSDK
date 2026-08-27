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

"""Immutable Audit history segments and active-envelope size policy."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_file import dir_fsync
from .error_codes import LkbErrorCode
from .json_store_models import BoardEnvelope, _now_iso

AUDIT_SOFT_EVENT_COUNT = 2_000
AUDIT_SOFT_MAX_BYTES = 8 * 1024 * 1024
AUDIT_HARD_MAX_BYTES = 64 * 1024 * 1024
AUDIT_TAIL_EVENT_COUNT = 500
AUDIT_TAIL_COMMAND_COUNT = 500
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600

logger = logging.getLogger(__name__)
_windows_permission_warning_lock = threading.Lock()
_windows_permission_warning_emitted = False


class AuditSizeWarning(UserWarning):
    """The active Audit payload should be manually compacted."""


class AuditSizeLimitError(RuntimeError):
    """A write would exceed the active Audit hard limit."""

    code = LkbErrorCode.AUDIT_SIZE_LIMIT


@dataclass(frozen=True, slots=True)
class AuditCompactionPlan:
    events: tuple[dict[str, Any], ...]
    processed_commands: tuple[tuple[str, dict[str, Any]], ...]
    validation_runs: tuple[tuple[str, dict[str, Any]], ...]

    @property
    def empty(self) -> bool:
        return not (self.events or self.processed_commands or self.validation_runs)


@dataclass(frozen=True, slots=True)
class AuditCompactionResult:
    segment_id: str | None
    file: str | None
    event_count: int
    processed_command_count: int
    validation_run_count: int
    bytes_before: int
    bytes_after: int


def active_audit_size(envelope: BoardEnvelope) -> int:
    """Return canonical UTF-8 bytes used by active audit collections."""
    payload = {
        "events": envelope.events,
        "processedCommands": envelope.processed_commands,
        "validationRuns": envelope.validation_runs,
        "historySegments": envelope.history_segments,
    }
    return len(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def needs_compaction(envelope: BoardEnvelope) -> bool:
    return len(envelope.events) >= AUDIT_SOFT_EVENT_COUNT or active_audit_size(envelope) >= AUDIT_SOFT_MAX_BYTES


def build_compaction_plan(envelope: BoardEnvelope) -> AuditCompactionPlan:
    """Select old audit material while retaining live read/idempotency state."""
    event_cut = max(0, len(envelope.events) - AUDIT_TAIL_EVENT_COUNT)
    events = tuple(envelope.events[:event_cut])

    command_items = list(envelope.processed_commands.items())
    command_cut = max(0, len(command_items) - AUDIT_TAIL_COMMAND_COUNT)
    processed = tuple(
        (command_id, record)
        for command_id, record in command_items[:command_cut]
        if not record.get("history_segment_id")
    )

    selected_validation_ids = {
        str(record.get("validation_run_id")) for _, record in processed if record.get("validation_run_id")
    }
    latest_by_subject: dict[str, str] = {}
    for run_id, run in envelope.validation_runs.items():
        subject = run.get("subjectRef") or run.get("subject_ref")
        if isinstance(subject, dict):
            subject = f"{subject.get('graph', '')}:{subject.get('kind', '')}:{subject.get('id', '')}"
        if isinstance(subject, str) and subject:
            latest_by_subject[subject] = run_id
    protected_validation_ids = set(latest_by_subject.values())
    validations = tuple(
        (run_id, envelope.validation_runs[run_id])
        for run_id in envelope.validation_runs
        if run_id in selected_validation_ids and run_id not in protected_validation_ids
    )
    return AuditCompactionPlan(events, processed, validations)


def _record_lines(plan: AuditCompactionPlan) -> bytes:
    records: list[dict[str, Any]] = []
    records.extend({"kind": "event", "record": event} for event in plan.events)
    records.extend(
        {"kind": "processed_command", "commandId": command_id, "record": record}
        for command_id, record in plan.processed_commands
    )
    records.extend(
        {"kind": "validation_run", "validationRunId": run_id, "record": record}
        for run_id, record in plan.validation_runs
    )
    return b"".join(
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _revision_bounds(plan: AuditCompactionPlan) -> tuple[int, int]:
    revisions: list[int] = []
    for event in plan.events:
        revision = event.get("store_revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            revisions.append(revision)
    for _, record in plan.processed_commands:
        revision = record.get("store_revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            revisions.append(revision)
    return (min(revisions, default=0), max(revisions, default=0))


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    if os.name == "posix":
        path.chmod(_PRIVATE_DIRECTORY_MODE)


def _ensure_private_file(path: Path) -> None:
    if os.name == "posix":
        path.chmod(_PRIVATE_FILE_MODE)


def _warn_windows_permission_fallback() -> None:
    """Record that Windows history protection relies on inherited ACLs."""
    global _windows_permission_warning_emitted
    if os.name != "nt" or _windows_permission_warning_emitted:
        return
    with _windows_permission_warning_lock:
        if _windows_permission_warning_emitted:
            return
        logger.warning(
            "LKB audit history permissions rely on inherited Windows ACLs; POSIX permission modes are not enforced."
        )
        _windows_permission_warning_emitted = True


def write_history_segment(
    board_dir: Path,
    plan: AuditCompactionPlan,
    *,
    previous_segment_hash: str = "",
) -> dict[str, Any]:
    """Publish a deterministic immutable gzip JSONL segment and manifest."""
    _warn_windows_permission_fallback()
    raw = _record_lines(plan)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    start_revision, end_revision = _revision_bounds(plan)
    segment_id = f"audit-{start_revision}-{end_revision}-{digest[:16]}"
    filename = f"{segment_id}.jsonl.gz"
    history_dir = board_dir / "history"
    _ensure_private_directory(history_dir)
    target = history_dir / filename

    if target.exists():
        if target.read_bytes() != compressed:
            raise RuntimeError(f"immutable history segment collision: {target}")
    else:
        tmp_dir = history_dir / ".tmp"
        _ensure_private_directory(tmp_dir)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=tmp_dir)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            _ensure_private_file(target)
            dir_fsync(history_dir)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    _ensure_private_file(target)

    manifest: dict[str, Any] = {
        "segmentId": segment_id,
        "file": filename,
        "sha256": f"sha256:{digest}",
        "previousSegmentHash": previous_segment_hash,
        "startStoreRevision": start_revision,
        "endStoreRevision": end_revision,
        "eventCount": len(plan.events),
        "processedCommandCount": len(plan.processed_commands),
        "validationRunCount": len(plan.validation_runs),
        "uncompressedBytes": len(raw),
        "compressedBytes": len(compressed),
        "createdAt": _now_iso(),
    }
    validate_history_segment(history_dir, manifest)
    return manifest


def validate_history_segment(history_dir: Path, manifest: dict[str, Any]) -> None:
    """Validate path safety, bytes/hash, gzip stream, JSONL and counts."""
    for key in (
        "startStoreRevision",
        "endStoreRevision",
        "eventCount",
        "processedCommandCount",
        "validationRunCount",
        "uncompressedBytes",
        "compressedBytes",
    ):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"history segment {key} must be a non-negative integer")
    if manifest["endStoreRevision"] < manifest["startStoreRevision"]:
        raise ValueError("history segment revision bounds are invalid")
    filename = manifest.get("file")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("history segment file must be a safe basename")
    path = history_dir / filename
    compressed = path.read_bytes()
    expected_hash = manifest.get("sha256")
    actual_hash = f"sha256:{hashlib.sha256(compressed).hexdigest()}"
    if expected_hash != actual_hash:
        raise ValueError(f"history segment hash mismatch: {filename}")
    if manifest.get("compressedBytes") != len(compressed):
        raise ValueError(f"history segment compressed size mismatch: {filename}")
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ValueError(f"history segment gzip is invalid: {filename}") from exc
    if manifest.get("uncompressedBytes") != len(raw):
        raise ValueError(f"history segment uncompressed size mismatch: {filename}")

    counts = {"event": 0, "processed_command": 0, "validation_run": 0}
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"history segment JSONL is invalid: {filename}") from exc
        if not isinstance(record, dict) or record.get("kind") not in counts:
            raise ValueError(f"history segment record is invalid: {filename}")
        counts[str(record["kind"])] += 1
    expected_counts = {
        "event": manifest.get("eventCount"),
        "processed_command": manifest.get("processedCommandCount"),
        "validation_run": manifest.get("validationRunCount"),
    }
    if counts != expected_counts:
        raise ValueError(f"history segment record counts mismatch: {filename}")


def thin_processed_command(record: dict[str, Any], segment_id: str) -> dict[str, Any]:
    """Retain the complete idempotency result summary plus segment locator."""
    keys = (
        "command_id",
        "request_hash",
        "decision",
        "store_revision",
        "revision_vector",
        "validation_run_id",
        "error_code",
        "reason",
        "derived_facts",
        "claim_id",
        "affected_refs",
    )
    result = {key: record[key] for key in keys if key in record}
    result["history_segment_id"] = segment_id
    return result


__all__ = [
    "AUDIT_HARD_MAX_BYTES",
    "AUDIT_SOFT_EVENT_COUNT",
    "AUDIT_SOFT_MAX_BYTES",
    "AuditCompactionPlan",
    "AuditCompactionResult",
    "AuditSizeLimitError",
    "AuditSizeWarning",
    "active_audit_size",
    "build_compaction_plan",
    "needs_compaction",
    "thin_processed_command",
    "validate_history_segment",
    "write_history_segment",
]
