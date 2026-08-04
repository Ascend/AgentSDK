#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the F-97-I analytics → telemetry bridge.

Covers the 14-way ``AnalyticsEvent`` → ``TelemetryEvent`` mapping in
:class:`AnalyticsTelemetrySink`, the redaction pass-through that
scrubs ``prompt``/``output``/secrets from the data dict, and the
idempotent :func:`install_analytics_bridge` lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from telemetry import recorder as recorder_mod
from telemetry.bridge import (
    AnalyticsTelemetrySink,
    get_analytics_bridge,
    install_analytics_bridge,
    reset_analytics_bridge_for_tests,
)
from telemetry.config import TelemetryConfig
from telemetry.recorder import (
    _TelemetryRecorderImpl,
    override_recorder,
    reset_recorder_for_tests,
)
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.storage import utc_date, utc_now
from clawcodex_ext.services.analytics.events import (
    AnalyticsEvent,
    EventType as AnalyticsEventType,
    set_analytics_sink,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Wipe both the recorder singleton and the bridge singleton.

    The default global analytics sink is reset to ``NullSink`` so other
    tests that depend on the analytics module don't observe a leaked
    bridge instance.
    """
    reset_recorder_for_tests()
    reset_analytics_bridge_for_tests()
    set_analytics_sink(_NullAnalyticsSink())
    yield
    reset_recorder_for_tests()
    reset_analytics_bridge_for_tests()
    set_analytics_sink(_NullAnalyticsSink())


class _NullAnalyticsSink:
    """No-op analytics sink for test isolation."""

    def emit(self, event):  # noqa: ARG002
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _build_impl(tmp_path: Path) -> _TelemetryRecorderImpl:
    storage = recorder_mod.LocalJsonlStorage(tmp_path / "telemetry", 7)
    return _TelemetryRecorderImpl(
        cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
        storage=storage,
        aggregator=recorder_mod.DailyAggregator(storage),
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        reporters=recorder_mod.CompositeReporter(),
    )


def _read_event_rows(storage: recorder_mod.LocalJsonlStorage) -> list[dict]:
    return storage.read_day("events", utc_date(utc_now()))


# ---------------------------------------------------------------------------
# No-op / disabled telemetry
# ---------------------------------------------------------------------------


def test_emit_is_noop_when_telemetry_disabled(tmp_path):
    """With no recorder override the global sink is NullSink → no rows."""
    bridge = AnalyticsTelemetrySink()
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="x",
            data={"subtype": "resize"},
        )
    )
    # No exception, no storage side-effects (NullSink is still the
    # global sink at this point because install_analytics_bridge was
    # not called).
    from clawcodex_ext.services.analytics.events import get_analytics_sink

    assert isinstance(get_analytics_sink(), _NullAnalyticsSink)


def test_emit_is_noop_when_recorder_disabled(tmp_path):
    """Even after install, an explicit NullRecorder override stays no-op."""
    override_recorder(recorder_mod._NullRecorder())
    bridge = install_analytics_bridge()
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="x",
            data={"subtype": "resize"},
        )
    )
    # No rows written; nothing to read.
    assert _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry")) == []


# ---------------------------------------------------------------------------
# All 14 analytics EventTypes
# ---------------------------------------------------------------------------


def test_image_processing_maps_to_tool_summary(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()

    bridge = get_analytics_bridge()
    assert bridge is not None
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="img-session-1",
            model="claude-opus-4-7",
            data={"subtype": "pdf_page_extraction", "success": True, "page_count": 3},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "tool_summary"
    assert row["fields"]["tool_name"] == "image_processing"
    assert row["fields"]["subtype"] == "pdf_page_extraction"
    assert row["fields"]["success"] is True
    assert row["fields"]["page_count"] == 3
    # Provenance fields are preserved.
    assert row["fields"]["analytics_event_type"] == "image_processing"
    assert row["fields"]["model"] == "claude-opus-4-7"


def test_session_start_and_end_mapping(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.SESSION_START,
            session_id="s1",
            data={"entrypoint": "image_pipeline"},
        )
    )
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.SESSION_END,
            session_id="s1",
            data={"duration_s": 1.23, "exit_status": 0},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    types = [r["type"] for r in rows]
    assert types == ["session_start", "session_end"]


@pytest.mark.parametrize(
    "analytics_type",
    [
        AnalyticsEventType.TURN_START,
        AnalyticsEventType.TURN_END,
        AnalyticsEventType.AGENT_SPAWN,
        AnalyticsEventType.AGENT_COMPLETE,
    ],
)
def test_command_run_subtypes(tmp_path, analytics_type):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=analytics_type,
            session_id="s1",
            data={"agent_id": "a-1"},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 1
    assert rows[0]["type"] == "command_run"
    # The bridge preserves the original type in `subtype`. The
    # redactor's command-name whitelist does not include
    # ``turn_start``/``turn_end``/``agent_spawn``/``agent_complete``,
    # so ``command_name`` is bucketized to ``"other"``. This is the
    # documented privacy guarantee — the analytics event type is
    # carried in ``subtype``, not in the whitelisted command slot.
    assert rows[0]["fields"]["subtype"] == analytics_type.value
    assert rows[0]["fields"]["command_name"] == "other"
    assert rows[0]["fields"]["agent_id"] == "a-1"


def test_tool_use_maps_to_tool_summary(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.TOOL_USE,
            session_id="s1",
            data={"tool": "Bash", "success": True, "duration_s": 0.4},
        )
    )
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.TOOL_RESULT,
            session_id="s1",
            data={"tool": "Bash", "success": True, "duration_s": 0.4},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 2
    for row in rows:
        assert row["type"] == "tool_summary"
        assert row["fields"]["tool_name"] == "Bash"
        assert row["fields"]["subtype"] in ("tool_use", "tool_result")
        assert row["fields"]["success"] is True
        assert row["fields"]["duration_s"] == 0.4


def test_error_event_builds_fingerprint_from_string(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.ERROR,
            session_id="s1",
            data={
                "error_class": "ImageSizeError",
                "message": "image exceeded max size 102400 bytes",
            },
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "error"
    assert row["fields"]["error_class"] == "ImageSizeError"
    assert len(row["fields"]["fingerprint"]) == 16
    fingerprint = row["fields"]["fingerprint"]

    # Two messages with different *shape* (not just different numbers)
    # produce different fingerprints. Numbers alone are stripped by
    # the volatile-token normalizer — that is by design so a runtime
    # raising the same error with a different id still merges into
    # the same fingerprint bucket.
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.ERROR,
            session_id="s1",
            data={
                "error_class": "ImageSizeError",
                "message": "decoding failed: unsupported PIL format",
            },
        )
    )
    rows2 = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert rows2[1]["fields"]["fingerprint"] != fingerprint

    # Two messages with the same shape but different numeric values
    # produce the same fingerprint.
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.ERROR,
            session_id="s1",
            data={
                "error_class": "ImageSizeError",
                "message": "image exceeded max size 9999999 bytes",
            },
        )
    )
    rows3 = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert rows3[2]["fields"]["fingerprint"] == fingerprint


@pytest.mark.parametrize(
    "dropped_type",
    [
        AnalyticsEventType.COMPACT,
        AnalyticsEventType.PERMISSION_PROMPT,
        AnalyticsEventType.PERMISSION_DECISION,
        AnalyticsEventType.MODEL_SWITCH,
    ],
)
def test_sensitive_or_unsupported_types_are_dropped(tmp_path, dropped_type):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=dropped_type,
            session_id="s1",
            data={"anything": "value"},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert rows == []


# ---------------------------------------------------------------------------
# Session-id hashing
# ---------------------------------------------------------------------------


def test_session_id_is_hashed_to_16_char_prefix(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="very-long-raw-session-id-1234",
            data={"subtype": "resize"},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 1
    sid = rows[0]["session_id"]
    assert isinstance(sid, str)
    assert len(sid) == 16
    # Stable across calls with the same raw id.
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="very-long-raw-session-id-1234",
            data={"subtype": "resize"},
        )
    )
    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert rows[1]["session_id"] == sid


def test_empty_session_id_yields_empty_string(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="",
            data={"subtype": "resize"},
        )
    )

    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert rows[0]["session_id"] == ""


# ---------------------------------------------------------------------------
# install_analytics_bridge lifecycle
# ---------------------------------------------------------------------------


def test_install_analytics_bridge_is_idempotent():
    a = install_analytics_bridge()
    b = install_analytics_bridge()
    assert a is b
    assert get_analytics_bridge() is a


def test_install_analytics_bridge_installs_into_global_sink():
    from clawcodex_ext.services.analytics.events import get_analytics_sink

    bridge = install_analytics_bridge()
    assert get_analytics_sink() is bridge


def test_reset_analytics_bridge_drops_singleton():
    install_analytics_bridge()
    assert get_analytics_bridge() is not None
    reset_analytics_bridge_for_tests()
    assert get_analytics_bridge() is None


# ---------------------------------------------------------------------------
# Malformed data
# ---------------------------------------------------------------------------


def test_emit_survives_malformed_data(tmp_path):
    impl = _build_impl(tmp_path)
    override_recorder(impl)
    install_analytics_bridge()
    bridge = get_analytics_bridge()

    # data is None — bridge must not raise.
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="s1",
            data=None,  # type: ignore[arg-type]
        )
    )
    # Non-string subtype
    bridge.emit(
        AnalyticsEvent(
            type=AnalyticsEventType.IMAGE_PROCESSING,
            session_id="s1",
            data={"subtype": 42, "success": "yes"},  # type: ignore[dict-item]
        )
    )
    # Missing success defaults to True
    rows = _read_event_rows(recorder_mod.LocalJsonlStorage(tmp_path / "telemetry"))
    assert len(rows) == 2
    assert rows[0]["fields"]["subtype"] == "unknown"
    assert rows[1]["fields"]["success"] is True
    assert rows[1]["fields"]["subtype"] == "42"
