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

"""Tests for DryRunReporter and LocalFileReporter."""

from __future__ import annotations

import time

from telemetry.aggregator import DailyAggregator
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.reporters import DryRunReporter, LocalFileReporter
from telemetry.storage import LocalJsonlStorage, utc_date, utc_now


def _summary_dict(tmp_path):
    storage = LocalJsonlStorage(tmp_path / "telemetry", 7)
    storage.append(
        "events",
        {
            "type": "session_start",
            "timestamp": time.time(),
            "session_id": "abc",
            "fields": {
                "entrypoint": "cli",
                "platform": "Linux",
                "python_version": "3.11",
                "provider": "anthropic",
                "model": "claude",
            },
        },
    )
    storage.append(
        "events",
        {
            "type": "command_run",
            "timestamp": time.time(),
            "session_id": "abc",
            "fields": {
                "command_name": "repl",
                "mode": "interactive",
                "success": True,
                "duration_s": 1.0,
                "exit_status": 0,
            },
        },
    )
    storage.append(
        "crashes",
        {
            "type": "error",
            "timestamp": time.time(),
            "session_id": "abc",
            "fields": {
                "error_class": "ValueError",
                "fingerprint": "abc123",
                "stacktrace": [],
            },
        },
    )
    agg = DailyAggregator(storage)
    today = utc_date(utc_now())
    summary = agg.aggregate(today)
    return storage, today, summary


def test_dry_run_reporter_renders(tmp_path):
    storage, today, summary = _summary_dict(tmp_path)
    reporter = DryRunReporter()
    rendered = reporter.render(summary, today)
    assert "ClawCodex Telemetry" in rendered and "⚠️" in rendered
    assert "Sessions: 1" in rendered
    assert "Error report" in rendered
    assert reporter.emit(rendered, date=today) is True
    assert reporter.last_rendered == rendered
    assert reporter.last_date == today


def test_local_file_reporter_writes_file(tmp_path):
    storage, today, summary = _summary_dict(tmp_path)
    redactor = Redactor(RedactionConfig(), (str(tmp_path),))
    reporter = LocalFileReporter(storage, redactor)
    rendered = reporter.render(summary, today)
    assert reporter.emit(rendered, date=today) is True
    out = storage.base_dir / "reports" / f"{today}.md"
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "Sessions: 1" in body


def test_local_file_reporter_blocks_on_secret(tmp_path):
    storage, today, summary = _summary_dict(tmp_path)
    redactor = Redactor(RedactionConfig(), (str(tmp_path),))
    reporter = LocalFileReporter(storage, redactor)
    bad = "leaked: AKIAIOSFODNN7EXAMPLE\n"
    assert reporter.emit(bad, date=today) is False
    out = storage.base_dir / "reports" / f"{today}.md"
    assert not out.exists()
    blocked = storage.read_day("reporter_blocked", today)
    assert blocked and any(row.get("reason") == "secret_scan" for row in blocked)
