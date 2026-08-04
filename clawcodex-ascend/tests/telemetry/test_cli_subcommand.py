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

"""Tests for the telemetry CLI subcommands."""

from __future__ import annotations

import json

from telemetry import cli
from telemetry.config import ReportingConfig, TelemetryConfig
from telemetry.recorder import reset_recorder_for_tests


class _PreviewRecorder:
    enabled = True

    def __init__(self) -> None:
        self.dates: list[str] = []

    def build_report_for(self, date: str) -> str:
        self.dates.append(date)
        return f"safe report for {date}"


class _SecretPreviewRecorder:
    enabled = True

    def build_report_for(self, date: str) -> str:
        return "rendered body contains leaked AKIAIOSFODNN7EXAMPLE"


def test_status_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    reset_recorder_for_tests()
    rc = cli.run_status([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Telemetry status" in out
    assert "enabled        : True" in out  # dev-default


def test_enable_prints_snippet(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = cli.run_enable([])
    out = capsys.readouterr().out
    assert rc == 0
    # The snippet should be parseable JSON.
    assert "telemetry" in out
    # Find the JSON blob in the output.
    start = out.find("{")
    assert start != -1
    blob = out[start:]
    parsed = json.loads(blob)
    assert "telemetry" in parsed
    assert "storage_dir" in parsed["telemetry"]


def test_disable_prints_snippet(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = cli.run_disable([])
    out = capsys.readouterr().out
    assert rc == 0
    start = out.find("{")
    parsed = json.loads(out[start:])
    assert parsed["telemetry"]["enabled"] is False


def test_preview_when_disabled(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    reset_recorder_for_tests()
    rc = cli.run_preview([])
    out = capsys.readouterr().out
    assert rc == 0  # dev-default — enabled=True now
    assert "disabled" not in out


def test_preview_accepts_main_style_date_arg(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    recorder = _PreviewRecorder()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
    )
    monkeypatch.setattr(cli, "get_recorder", lambda: recorder)

    rc = cli.main(["preview", "2026-06-14"])
    out = capsys.readouterr().out

    assert rc == 0
    assert recorder.dates == ["2026-06-14"]
    assert "safe report for 2026-06-14" in out


def test_preview_accepts_direct_date_arg(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    recorder = _PreviewRecorder()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
    )
    monkeypatch.setattr(cli, "get_recorder", lambda: recorder)

    rc = cli.run_preview(["2026-06-13"])
    out = capsys.readouterr().out

    assert rc == 0
    assert recorder.dates == ["2026-06-13"]
    assert "safe report for 2026-06-13" in out


def test_preview_secret_scan_refuses_rendered_body(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: TelemetryConfig(enabled=True, storage_dir=tmp_path / "telemetry"),
    )
    monkeypatch.setattr(cli, "get_recorder", lambda: _SecretPreviewRecorder())

    rc = cli.run_preview(["2026-06-13"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Secret scan matched" in out
    assert "rendered body contains" not in out


def test_main_dispatches_to_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Telemetry status" in out


def test_main_unknown_subcommand(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = cli.main(["bogus"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown subcommand" in captured.out


def test_status_prints_issue_reporting_fields_without_secret(monkeypatch, tmp_path, capsys):
    secret = "ghp_12345678901234567890"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: TelemetryConfig(
            enabled=False,
            storage_dir=tmp_path / "telemetry",
            reporting=ReportingConfig(
                reporting_enabled=True,
                kind="issue",
                platform="gitcode",
                owner="acme",
                repo="widget",
                endpoint="https://gitcode.example/api",
                issue_title="Telemetry Inbox",
                mode="create_daily",
                interval_hours=6,
                token_env="CLAW_TELEMETRY_REPORTING_TOKEN",
                api_key=secret,
            ),
        ),
    )
    reset_recorder_for_tests()

    rc = cli.run_status([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "kind='issue' mode='create_daily'" in out
    assert "platform      : gitcode" in out
    assert "owner/repo    : acme / widget" in out
    assert "issue_title   : Telemetry Inbox" in out
    assert "token_env     : CLAW_TELEMETRY_REPORTING_TOKEN" in out
    assert "api_key_set   : True" in out
    assert secret not in out


def test_enable_snippet_includes_issue_fields_without_api_key(monkeypatch, tmp_path, capsys):
    secret = "ghp_12345678901234567890"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: TelemetryConfig(
            storage_dir=tmp_path / "telemetry",
            reporting=ReportingConfig(
                reporting_enabled=True,
                kind="issue",
                owner="acme",
                repo="widget",
                api_key=secret,
            ),
        ),
    )

    rc = cli.run_enable([])
    out = capsys.readouterr().out
    parsed = json.loads(out[out.find("{") :])
    reporting = parsed["telemetry"]["reporting"]

    assert rc == 0
    assert reporting["kind"] == "issue"
    assert reporting["owner"] == "acme"
    assert reporting["repo"] == "widget"
    assert reporting["token_env"] == "CLAW_TELEMETRY_REPORTING_TOKEN"
    assert "api_key" not in reporting
    assert secret not in out
