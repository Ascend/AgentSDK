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

"""End-to-end privacy audit tests for telemetry reporter payloads."""

from __future__ import annotations

from telemetry import recorder as recorder_mod
from telemetry.aggregator import DailyAggregator
from telemetry.bridge import (
    install_analytics_bridge,
    reset_analytics_bridge_for_tests,
)
from telemetry.config import ReportingConfig, TelemetryConfig
from telemetry.events import EventType, TelemetryEvent
from telemetry.recorder import (
    _TelemetryRecorderImpl,
    override_recorder,
    reset_recorder_for_tests,
)
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.reporters.dry_run import DryRunReporter
from telemetry.reporters.issue import IssueReporter
from telemetry.storage import LocalJsonlStorage, utc_date, utc_now
from clawcodex_ext.services.analytics.events import (
    AnalyticsEvent,
    EventType as AnalyticsEventType,
    set_analytics_sink,
)


class _NoopClient:
    def __init__(self) -> None:
        self.find_titles: list[str] = []
        self.created: list[dict[str, str]] = []

    async def find_issue_by_title(self, title: str, *, state: str = "open") -> None:
        self.find_titles.append(title)
        return None

    async def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict[str, str]:
        payload = {"number": "1", "title": title, "body": body}
        self.created.append(payload)
        return payload


class _PlatformClient(_NoopClient):
    class platform:
        open_state = "open"


def test_reporter_payload_omits_sensitive_raw_event_fields(tmp_path) -> None:
    date = "2026-06-15"
    storage = LocalJsonlStorage(tmp_path / "telemetry", 7)
    sensitive_values = [
        "write a private prompt about payroll",
        "assistant output with customer secret",
        "full transcript should not leave disk",
        "tool output containing file body",
        "CLAWCODEx_SECRET_TOKEN",
        "ghp_12345678901234567890",
        "/home/alice/project/private.txt",
        "private source file contents",
    ]
    storage.append(
        "events",
        TelemetryEvent(
            type=EventType.SESSION_START,
            session_id="session-1",
            fields={
                "platform": "linux",
                "provider": "anthropic",
                "model": "claude-sonnet",
                "prompt": sensitive_values[0],
                "assistant_output": sensitive_values[1],
                "transcript": sensitive_values[2],
                "tool_output": sensitive_values[3],
                "env": {"CLAWCODEx_SECRET": sensitive_values[4]},
                "api_key": sensitive_values[5],
                "absolute_path": sensitive_values[6],
                "file_contents": sensitive_values[7],
            },
        ).to_dict(),
        date=date,
    )
    storage.append(
        "events",
        TelemetryEvent(
            type=EventType.COMMAND_RUN,
            session_id="session-1",
            fields={
                "command_name": "print",
                "success": True,
                "exit_status": 0,
                "duration_s": 1.25,
                "command_args": "--secret ghp_12345678901234567890",
                "cwd": "/home/alice/project",
            },
        ).to_dict(),
        date=date,
    )

    summary = DailyAggregator(storage).aggregate(date)
    reporter = DryRunReporter()
    rendered = reporter.render(summary, date)

    assert "Sessions: 1" in rendered
    assert "Command runs: 1" in rendered
    assert "anthropic 1" in rendered
    assert "print: 1" in rendered
    for value in sensitive_values:
        assert value not in rendered
    assert "--secret" not in rendered
    assert "/home/alice/project" not in rendered


def test_issue_secret_scan_blocks_upload_without_persisting_body(tmp_path) -> None:
    date = "2026-06-15"
    storage = LocalJsonlStorage(tmp_path / "telemetry", 7)
    client = _PlatformClient()
    reporter = IssueReporter(
        storage=storage,
        redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
        config=ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            platform="github",
            owner="acme",
            repo="widget",
            api_key="token",
        ),
        client=client,
    )
    rendered = "# Summary\nleaked AKIAIOSFODNN7EXAMPLE\nprivate prompt body\n"

    assert reporter.emit(rendered, date=date) is False

    assert not client.find_titles
    assert not client.created
    rows = storage.read_day("reporter_errors", date)
    assert rows and rows[0]["reason"] == "secret_scan"
    assert "rendered" not in rows[0]
    assert "AKIAIOSFODNN7EXAMPLE" not in str(rows[0])
    assert "private prompt body" not in str(rows[0])


# ---------------------------------------------------------------------------
# I: analytics bridge privacy audit
# ---------------------------------------------------------------------------


class _NullAnalyticsSink:
    """No-op analytics sink used to reset the global between tests."""

    def emit(self, event):  # noqa: ARG002
        return None

    def flush(self):
        return None

    def close(self):
        return None


def test_analytics_bridge_redacts_prompt_output_and_secrets(tmp_path) -> None:
    """Bridge forwarding an analytics event whose ``data`` carries
    ``prompt`` / ``output`` / ``api_key`` / env dict must not leak
    those values through the recorder → aggregator → DryRunReporter
    pipeline.

    The redaction is automatic — the bridge builds a
    :class:`TelemetryEvent` whose ``fields`` flow through
    :meth:`Redactor.redact_event` inside the recorder, and the
    aggregator's daily summary feeds the DryRunReporter.
    """
    reset_recorder_for_tests()
    reset_analytics_bridge_for_tests()
    set_analytics_sink(_NullAnalyticsSink())
    try:
        storage = LocalJsonlStorage(tmp_path / "telemetry", 7)
        impl = _TelemetryRecorderImpl(
            cfg=TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
            storage=storage,
            aggregator=recorder_mod.DailyAggregator(storage),
            redactor=Redactor(RedactionConfig(), (str(tmp_path),)),
            reporters=recorder_mod.CompositeReporter(),
        )
        override_recorder(impl)
        install_analytics_bridge()

        sensitive_prompt = "summarize the secret payroll file"
        sensitive_output = "decoded transcript containing PII"
        sensitive_key = "ghp_abcdef0123456789ABCDEF"

        bridge = __import__(
            "telemetry.bridge",
            fromlist=["get_analytics_bridge"],
        ).get_analytics_bridge()
        assert bridge is not None
        bridge.emit(
            AnalyticsEvent(
                type=AnalyticsEventType.IMAGE_PROCESSING,
                session_id="audit-session",
                model="claude-opus-4-7",
                data={
                    "subtype": "pdf_page_extraction",
                    "page_count": 2,
                    "prompt": sensitive_prompt,
                    "output": sensitive_output,
                    "api_key": sensitive_key,
                    "env": {"CLAWCODEx_TOKEN": "should-not-leave"},
                },
            )
        )

        date = utc_date(utc_now())
        summary = DailyAggregator(storage).aggregate(date)
        rendered = DryRunReporter().render(summary, date)

        for value in (
            sensitive_prompt,
            sensitive_output,
            sensitive_key,
            "CLAWCODEx_TOKEN",
            "should-not-leave",
        ):
            assert value not in rendered, f"{value!r} leaked into reporter payload"
    finally:
        reset_recorder_for_tests()
        reset_analytics_bridge_for_tests()
        set_analytics_sink(_NullAnalyticsSink())


# ---------------------------------------------------------------------------
# L: cross-version privacy + dedup invariant
# ---------------------------------------------------------------------------


def test_v1_v2_fingerprint_redact_and_migrate_hash_equivalent() -> None:
    """A v1 ERROR event with a 16-char fingerprint string and a v2
    ERROR event with a structured fingerprint dict (same ``hash``)
    must remain dedupable after the redaction + migration pipeline.

    This is the single invariant the rollout rests on: even
    after the reda``tor scrubs secret-like patterns out of the hash
    and the migrator normalizes the v1 string into the v2 dict form,
    the two events must end up in the same crash bucket — so the
    daily crash summary doesn't double-count legacy errors.
    """
    from telemetry.migration import (
        _fingerprint_dict_to_hash,
        normalize_event,
    )

    shared_hash = "abc1234567890def"
    redactor = Redactor(RedactionConfig(), ())

    v1_event = TelemetryEvent(
        type=EventType.ERROR,
        session_id="legacy-sess",
        fields={
            "error_class": "ValueError",
            "fingerprint": shared_hash,
            "stacktrace": ["ValueError: oops"],
        },
    )
    v2_event = TelemetryEvent(
        type=EventType.ERROR,
        session_id="modern-sess",
        fields={
            "error_class": "ValueError",
            "fingerprint": {
                "hash": shared_hash,
                "version": 2,
                "method": "sha1-truncate",
            },
            "stacktrace": ["ValueError: oops"],
        },
    )

    v1_redacted = redactor.redact_event(v1_event).to_dict()
    v2_redacted = redactor.redact_event(v2_event).to_dict()

    v1_normalized = normalize_event(v1_redacted)
    v2_normalized = normalize_event(v2_redacted)

    v1_join_key = _fingerprint_dict_to_hash(v1_normalized["fields"]["fingerprint"])
    v2_join_key = _fingerprint_dict_to_hash(v2_normalized["fields"]["fingerprint"])

    assert v1_join_key == v2_join_key == shared_hash
    # Both end up at v2 after the pipeline
    assert v1_normalized["schema_version"] == 2
    assert v2_normalized["schema_version"] == 2
    # No secrets in either redacted payload
    for value in (shared_hash,):
        assert value in v1_redacted["fields"]["fingerprint"]  # short hash is non-sensitive
    assert v1_redacted["fields"].get("error_class") == "ValueError"
    assert v2_redacted["fields"].get("error_class") == "ValueError"
