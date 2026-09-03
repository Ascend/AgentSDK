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

"""Tests for the telemetry CLI subcommands."""

from __future__ import annotations

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


def test_enable_persists_config(monkeypatch, capsys):
    saved: list[dict] = []
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {"telemetry": {"storage_dir": "/tmp/telemetry"}},
    )
    monkeypatch.setattr("src.config.save_config", saved.append)

    rc = cli.run_enable([])
    out = capsys.readouterr().out

    assert rc == 0
    assert saved[0]["telemetry"]["enabled"] is True
    assert saved[0]["telemetry"]["reporting"]["reporting_enabled"] is True
    assert saved[0]["telemetry"]["storage_dir"] == "/tmp/telemetry"
    assert "Telemetry enabled" in out


def test_disable_persists_config(monkeypatch, capsys):
    saved: list[dict] = []
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {
            "telemetry": {
                "enabled": True,
                "reporting": {"reporting_enabled": True},
            }
        },
    )
    monkeypatch.setattr("src.config.save_config", saved.append)

    rc = cli.run_disable([])
    out = capsys.readouterr().out

    assert rc == 0
    assert saved[0]["telemetry"]["enabled"] is False
    assert saved[0]["telemetry"]["reporting"]["reporting_enabled"] is False
    assert "Telemetry disabled" in out


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


def test_enable_preserves_issue_fields_without_printing_api_key(monkeypatch, capsys):
    secret = "ghp_12345678901234567890"
    saved: list[dict] = []
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {
            "telemetry": {
                "reporting": {
                    "reporting_enabled": True,
                    "kind": "issue",
                    "owner": "acme",
                    "repo": "widget",
                    "api_key": secret,
                }
            }
        },
    )
    monkeypatch.setattr("src.config.save_config", saved.append)

    rc = cli.run_enable([])
    out = capsys.readouterr().out
    reporting = saved[0]["telemetry"]["reporting"]

    assert rc == 0
    assert reporting["kind"] == "issue"
    assert reporting["owner"] == "acme"
    assert reporting["repo"] == "widget"
    assert reporting["api_key"] == secret
    assert secret not in out
