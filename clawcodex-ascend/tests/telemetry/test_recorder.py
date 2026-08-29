#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for the recorder singleton + NullRecorder zero-cost path."""

from __future__ import annotations

from pathlib import Path

import pytest

from telemetry import recorder as recorder_mod
from telemetry.config import ReportingConfig, TelemetryConfig
from telemetry.events import EventType, TelemetryEvent
from telemetry.recorder import (
    _NullRecorder,
    _TelemetryRecorderImpl,
    get_recorder,
    override_recorder,
    reset_recorder_for_tests,
)
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.storage import utc_date, utc_now


@pytest.fixture(autouse=True)
def _reset_recorder():
    reset_recorder_for_tests()
    yield
    reset_recorder_for_tests()


def test_default_is_null_recorder():
    assert TelemetryConfig().enabled is True  # dev-default
    r = get_recorder()
    # All methods are no-ops and must not raise.
    r.record_session_start(session_id="x", entrypoint="cli")
    r.record_session_end(session_id="x", duration_s=0.0, exit_status=0)
    r.record_command_run(session_id="x", command_name="repl")
    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        r.record_error(session_id="x", exc=exc)
    r.record_tool_summary(session_id="x", tool_name="bash")
    r.record_event(TelemetryEvent(type=EventType.TOOL_SUMMARY, session_id="x", fields={"tool_name": "x"}))
    r.flush()
    r.close()


def test_override_recorder_runs_user_instance(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )
    override_recorder(impl)
    r = get_recorder()
    assert r is impl
    r.record_session_start(session_id="s1", entrypoint="cli", platform="Linux")
    today = utc_date(utc_now())
    rows = storage.read_day("events", today)
    assert any(row["type"] == "session_start" for row in rows)
    override_recorder(None)
    assert isinstance(get_recorder(), _TelemetryRecorderImpl)  # dev-default


def test_recorder_writes_crash_event(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )
    try:
        raise ValueError("boom")
    except ValueError as exc:
        impl.record_error(session_id="s1", exc=exc)
    today = utc_date(utc_now())
    crashes = storage.read_day("crashes", today)
    assert crashes and crashes[0]["fields"]["error_class"] == "ValueError"
    assert len(crashes[0]["fields"]["fingerprint"]) == 16


def test_record_event_writes_to_storage_via_public_api(tmp_path):
    """``record_event`` is the public chokepoint the bridge uses.

    It must accept a pre-built :class:`TelemetryEvent`, run it through
    the same redaction + storage + aggregation pipeline as the typed
    ``record_*()`` helpers, and produce the same JSONL row.
    """
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )

    event = TelemetryEvent(
        type=EventType.TOOL_SUMMARY,
        session_id="s1",
        fields={"tool_name": "image_processing", "subtype": "resize"},
    )
    impl.record_event(event)

    today = utc_date(utc_now())
    rows = storage.read_day("events", today)
    assert any(row["type"] == "tool_summary" and row["fields"]["tool_name"] == "image_processing" for row in rows)


def test_record_event_runs_through_redactor(tmp_path):
    """Sensitive fields on a raw TelemetryEvent must be scrubbed by
    ``Redactor.redact_event`` inside ``record_event``, not only by the
    typed helpers.
    """
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )

    impl.record_event(
        TelemetryEvent(
            type=EventType.SESSION_START,
            session_id="s1",
            fields={
                "entrypoint": "image_pipeline",
                "client_type": "cli",
                "is_non_interactive": True,
                "platform": "linux",
                "python_version": "3.12",
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "app_version": "0.0.0",
                "prompt": "private user prompt",
                "output": "private assistant output",
            },
        )
    )

    today = utc_date(utc_now())
    rows = storage.read_day("events", today)
    fields = rows[0]["fields"]
    assert "prompt" not in fields
    assert "output" not in fields


def test_record_event_is_noop_after_close(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )
    impl.close()
    impl.record_event(
        TelemetryEvent(
            type=EventType.TOOL_SUMMARY,
            session_id="s1",
            fields={"tool_name": "x"},
        )
    )
    today = utc_date(utc_now())
    assert storage.read_day("events", today) == []


def test_configure_reporters_wires_issue_reporter(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    reporters = recorder_mod.CompositeReporter()
    cfg = TelemetryConfig(
        enabled=True,
        storage_dir=tmp_path / "telemetry",
        reporting=ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            owner="acme",
            repo="widget",
            api_key="token",
        ),
    )

    recorder_mod._configure_reporters(
        cfg,
        storage,
        Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters,
    )

    names = [type(reporter).__name__ for reporter in reporters]
    assert names == ["IssueReporter"]


def test_configure_reporters_routes_local_file_mode_to_local_reporter(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    reporters = recorder_mod.CompositeReporter()
    cfg = TelemetryConfig(
        enabled=True,
        storage_dir=tmp_path / "telemetry",
        reporting=ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            mode="local_file",
        ),
    )

    recorder_mod._configure_reporters(
        cfg,
        storage,
        Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters,
    )

    names = [type(reporter).__name__ for reporter in reporters]
    assert names == ["LocalFileReporter"]


def test_flush_forces_fresh_aggregation_before_emit(tmp_path):
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    reporter = recorder_mod.DryRunReporter()
    reporters = recorder_mod.CompositeReporter([reporter])
    impl = _TelemetryRecorderImpl(
        cfg=TelemetryConfig(
            enabled=True,
            storage_dir=tmp_path / "telemetry",
            reporting=ReportingConfig(reporting_enabled=True, kind="dry_run"),
        ),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=reporters,
    )

    impl.record_session_start(session_id="s1", entrypoint="cli", platform="Linux")
    impl.record_command_run(session_id="s1", command_name="repl", duration_s=1.0)
    impl.flush()

    assert "Sessions: 1" in reporter.last_rendered
    assert "Command runs: 1" in reporter.last_rendered


# ---------------------------------------------------------------------------
# J: SessionAnalyticsMetadata → record_session_start bridge
# ---------------------------------------------------------------------------


def _build_recorder_for_j(tmp_path: Path) -> _TelemetryRecorderImpl:
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    return _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )


def test_record_session_start_includes_analytics_metadata_fields(tmp_path):
    """explicit analytics metadata kwargs land in the JSONL row."""
    impl = _build_recorder_for_j(tmp_path)

    impl.record_session_start(
        session_id="meta-explicit",
        entrypoint="tui",
        client_type="tui",
        is_non_interactive=False,
        platform="Linux",
        os_version="6.6.87.2-microsoft-standard-WSL2",
        ide_type="vscode",
        ide_version="1.102.0",
        is_resume=True,
        start_time=1718400000.0,
        extra={"region": "us-east-1", "channel": "internal"},
    )

    rows = impl._storage.read_day("events", utc_date(utc_now()))
    session = next(r for r in rows if r["type"] == "session_start")
    fields = session["fields"]
    assert fields["os_version"] == "6.6.87.2-microsoft-standard-WSL2"
    assert fields["ide_type"] == "vscode"
    assert fields["ide_version"] == "1.102.0"
    assert fields["is_resume"] is True
    assert fields["start_time"] == 1718400000.0
    assert fields["extra"] == {"region": "us-east-1", "channel": "internal"}


def test_record_session_start_autocollects_analytics_when_kwargs_omitted(tmp_path):
    """omitting the new kwargs still fills them in via
    ``collect_session_metadata``. The bridge must be transparent so old
    call sites that pass only the 8-kw signature keep working.
    """
    impl = _build_recorder_for_j(tmp_path)

    impl.record_session_start(
        session_id="meta-autocollect",
        entrypoint="cli",
        client_type="cli",
        is_non_interactive=True,
        platform="Linux",
        model="claude-opus-4-7",
        ide_type="vscode",
        ide_version="1.102.0",
        is_resume=False,
    )

    rows = impl._storage.read_day("events", utc_date(utc_now()))
    session = next(r for r in rows if r["type"] == "session_start")
    fields = session["fields"]
    # platform-supplied fields are kept verbatim
    assert fields["ide_type"] == "vscode"
    assert fields["ide_version"] == "1.102.0"
    assert fields["is_resume"] is False
    # os_version / start_time / extra are auto-derived (non-empty)
    assert isinstance(fields["os_version"], str) and fields["os_version"]
    assert isinstance(fields["start_time"], float) and fields["start_time"] > 0
    assert fields["extra"] == {}


def test_record_session_start_redacts_extra_dict(tmp_path):
    """the ``extra`` dict must never smuggle prompt/output/
    transcript/messages into the payload, even if a caller passes them
    directly. The redactor's ``_BLOCKED_EXTRA_KEYS`` set is the single
    gate.
    """
    impl = _build_recorder_for_j(tmp_path)

    impl.record_session_start(
        session_id="meta-redact",
        entrypoint="cli",
        client_type="cli",
        is_non_interactive=True,
        platform="Linux",
        extra={
            "PROMPT": "private user prompt",
            "output": "private assistant output",
            "transcript": "session transcript",
            "MESSAGES": "all messages",
            "region": "us-east-1",
        },
    )

    rows = impl._storage.read_day("events", utc_date(utc_now()))
    session = next(r for r in rows if r["type"] == "session_start")
    extra = session["fields"]["extra"]
    # Blocked keys are dropped, case-insensitively
    assert "PROMPT" not in extra
    assert "prompt" not in extra
    assert "output" not in extra
    assert "transcript" not in extra
    assert "MESSAGES" not in extra
    # Non-blocked keys survive
    assert extra == {"region": "us-east-1"}


def test_record_session_start_noop_when_disabled_even_with_metadata(tmp_path):
    """the NullRecorder swallows all kwargs without raising so
    direct call sites in CLI dispatch can pass the analytics fields
    unconditionally — no need to gate by enabled-state.
    """
    null = _NullRecorder()
    null.record_session_start(
        session_id="disabled",
        entrypoint="cli",
        client_type="cli",
        is_non_interactive=True,
        platform="Linux",
        os_version="6.6",
        ide_type="vscode",
        ide_version="1.102.0",
        is_resume=True,
        start_time=1718400000.0,
        extra={"k": "v"},
    )
    assert null.enabled is False
    # No I/O: no storage was attached
    assert not hasattr(null, "_storage")
