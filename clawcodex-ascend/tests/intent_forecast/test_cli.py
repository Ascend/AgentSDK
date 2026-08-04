#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from __future__ import annotations

from clawcodex_ext.intent_forecast.cli import run_forecast_command
from clawcodex_ext.intent_forecast.learning import read_recent_feedback
from clawcodex_ext.intent_forecast.messages import ForecastResult, ForecastSuggestion
from clawcodex_ext.intent_forecast.persistence import (
    forecast_history_path,
    read_forecast_history,
    save_forecast_result,
)


def test_cli_status_json(capsys) -> None:
    rc = run_forecast_command(["status", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"idle_seconds": 120' in out


def test_cli_process_pending(capsys) -> None:
    rc = run_forecast_command(["summarize", "--pending", "--json"])
    assert rc == 0
    assert '"processed"' in capsys.readouterr().out


def test_cli_accept_saved_result(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    save_forecast_result(
        ForecastResult(
            generated=True,
            suggestions=[ForecastSuggestion(id="s1", title="A", prompt="do saved", confidence=0.8)],
        ),
        trigger="cli",
        cwd=tmp_path,
    )
    monkeypatch.chdir(tmp_path)

    rc = run_forecast_command(["accept", "s1"])

    assert rc == 0
    assert "do saved" in capsys.readouterr().out
    assert forecast_history_path().exists()
    assert read_recent_feedback()[-1]["event"] == "accepted_started"


def test_cli_run_appends_forecast_history(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    assert run_forecast_command(["run"]) == 0
    assert run_forecast_command(["run"]) == 0

    capsys.readouterr()
    rows = read_forecast_history(cwd=tmp_path)
    assert len(rows) == 2
    assert {row["trigger"] for row in rows} == {"cli"}
