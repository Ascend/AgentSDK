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

"""Audit soft/hard limits and immutable manual history compaction."""

from __future__ import annotations

import gzip
import json
import os
import stat
from pathlib import Path

import pytest

from lkb import audit_compaction as audit
from lkb.board_resolver import board_dir
from lkb.commands import CommandResult
from lkb.doctor import FindingArea, FindingSeverity, doctor
from lkb.file_lock import BoardFileLock
from lkb.graph_types import Board, BoardPolicy
from lkb.json_store import JsonBoardStore


def _store(root: Path, board_id: str = "audit-board") -> JsonBoardStore:
    directory = board_dir(board_id, home=root)
    board = Board(
        board_id=board_id,
        project_uri=f"project:{board_id}",
        display_name=board_id,
        schema_version=1,
        store_revision=0,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        policy=BoardPolicy(),
    )
    return JsonBoardStore.create_board(
        directory,
        board=board,
        lock=BoardFileLock(directory),
        home=root,
    )


def _write_noop(store: JsonBoardStore, board_id: str, index: int) -> None:
    command_id = f"cmd-{index}"

    def mutate(envelope):
        return envelope, CommandResult(decision="committed", command_id=command_id)

    store.execute_atomic(
        board_id,
        command_id,
        f"hash-{index}",
        None,
        mutate,
        actor="test",
    )


def test_manual_compaction_preserves_idempotency_and_writes_valid_segment(
    tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_lkb_root)
    for index in range(6):
        _write_noop(store, "audit-board", index)
    monkeypatch.setattr(audit, "AUDIT_TAIL_EVENT_COUNT", 2)
    monkeypatch.setattr(audit, "AUDIT_TAIL_COMMAND_COUNT", 2)

    result = store.compact_audit("audit-board", actor="operator")

    assert result.segment_id
    assert result.event_count == 10
    assert result.processed_command_count == 4
    assert result.bytes_after < result.bytes_before
    envelope = store.load()
    assert len(envelope.history_segments) == 1
    manifest = envelope.history_segments[0]
    segment_path = Path(store._board_dir) / "history" / manifest["file"]
    records = [json.loads(line) for line in gzip.decompress(segment_path.read_bytes()).splitlines()]
    assert sum(record["kind"] == "event" for record in records) == result.event_count
    assert all(f"cmd-{index}" in envelope.processed_commands for index in range(6))
    assert envelope.processed_commands["cmd-0"]["history_segment_id"] == result.segment_id

    revision = envelope.store_revision

    def must_not_run(_envelope):
        raise AssertionError("idempotent compacted command was re-applied")

    replay = store.execute_atomic(
        "audit-board",
        "cmd-0",
        "hash-0",
        None,
        must_not_run,
        actor="test",
    )
    assert replay.committed
    assert store.load().store_revision == revision
    report = doctor("audit-board", home=tmp_lkb_root)
    assert not any(
        finding.area == FindingArea.HISTORY and finding.severity in {FindingSeverity.ERROR, FindingSeverity.CRITICAL}
        for finding in report.findings
    )


def test_doctor_reports_corrupt_referenced_segment(tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_lkb_root, "audit-corrupt")
    for index in range(4):
        _write_noop(store, "audit-corrupt", index)
    monkeypatch.setattr(audit, "AUDIT_TAIL_EVENT_COUNT", 1)
    monkeypatch.setattr(audit, "AUDIT_TAIL_COMMAND_COUNT", 1)
    result = store.compact_audit("audit-corrupt", actor="operator")
    segment = Path(store._board_dir) / "history" / str(result.file)
    segment.write_bytes(segment.read_bytes() + b"corrupt")

    report = doctor("audit-corrupt", home=tmp_lkb_root)

    assert any(
        finding.area == FindingArea.HISTORY and finding.severity == FindingSeverity.ERROR for finding in report.findings
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not portable")
def test_compaction_enforces_private_history_permissions(tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_lkb_root, "audit-permissions")
    history_dir = Path(store._board_dir) / "history"
    tmp_dir = history_dir / ".tmp"
    tmp_dir.mkdir(parents=True)
    history_dir.chmod(0o755)
    tmp_dir.chmod(0o755)
    for index in range(4):
        _write_noop(store, "audit-permissions", index)
    monkeypatch.setattr(audit, "AUDIT_TAIL_EVENT_COUNT", 1)
    monkeypatch.setattr(audit, "AUDIT_TAIL_COMMAND_COUNT", 1)

    result = store.compact_audit("audit-permissions", actor="operator")

    segment = history_dir / str(result.file)
    assert stat.S_IMODE(history_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(tmp_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(segment.stat().st_mode) == 0o600


def test_windows_permission_fallback_logs_one_safe_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(audit.os, "name", "nt")
    monkeypatch.setattr(audit, "_windows_permission_warning_emitted", False)

    with caplog.at_level("WARNING", logger=audit.__name__):
        audit._warn_windows_permission_fallback()
        audit._warn_windows_permission_fallback()

    expected_warning = (
        "LKB audit history permissions rely on inherited Windows ACLs; POSIX permission modes are not enforced."
    )
    assert caplog.messages == [expected_warning]


@pytest.mark.parametrize(
    "field",
    [
        "startStoreRevision",
        "endStoreRevision",
        "eventCount",
        "processedCommandCount",
        "validationRunCount",
        "uncompressedBytes",
        "compressedBytes",
    ],
)
def test_history_manifest_rejects_boolean_integer_fields(
    tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    store = _store(tmp_lkb_root, f"audit-bool-{field.lower()}")
    for index in range(3):
        _write_noop(store, store.load().board_id(), index)
    monkeypatch.setattr(audit, "AUDIT_TAIL_EVENT_COUNT", 1)
    monkeypatch.setattr(audit, "AUDIT_TAIL_COMMAND_COUNT", 1)
    store.compact_audit(store.load().board_id(), actor="operator")
    manifest = dict(store.load().history_segments[0])
    manifest[field] = True

    with pytest.raises(ValueError, match=field):
        audit.validate_history_segment(Path(store._board_dir) / "history", manifest)


def test_soft_warning_and_hard_limit_are_enforced(tmp_lkb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_lkb_root, "audit-limits")
    _write_noop(store, "audit-limits", 0)
    monkeypatch.setattr(audit, "AUDIT_SOFT_EVENT_COUNT", 1)
    with pytest.warns(audit.AuditSizeWarning, match="board compact"):
        store.load()

    monkeypatch.setattr(audit, "AUDIT_SOFT_EVENT_COUNT", 10_000)
    monkeypatch.setattr(audit, "AUDIT_HARD_MAX_BYTES", 1)
    with pytest.raises(audit.AuditSizeLimitError, match="hard limit"):
        _write_noop(store, "audit-limits", 1)
