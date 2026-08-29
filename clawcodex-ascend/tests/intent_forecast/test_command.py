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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from clawcodex_ext.intent_forecast.command import _forecast_call
from clawcodex_ext.intent_forecast.learning import read_recent_feedback
from clawcodex_ext.intent_forecast.messages import ForecastResult, ForecastSuggestion
from clawcodex_ext.intent_forecast.persistence import read_forecast_history
from src.agent.conversation import Conversation
from src.types.messages import Message


class FakeController:
    def __init__(self) -> None:
        self.last_result = ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[
                ForecastSuggestion(
                    id="s1",
                    title="Do one",
                    prompt="do one",
                    confidence=0.8,
                )
            ],
        )
        self.dismissed = False

    def dismiss(self) -> None:
        self.dismissed = True


def test_forecast_accept_uses_controller_last_result(tmp_path: Path) -> None:
    import clawcodex_ext.intent_forecast.learning as learning

    original = learning.feedback_path
    learning.feedback_path = lambda base_dir=None: tmp_path / "intent_forecast" / "feedback.jsonl"
    ctx = SimpleNamespace(cwd=tmp_path, intent_forecast_controller=FakeController())

    try:
        result = _forecast_call("accept 1", ctx)
    finally:
        learning.feedback_path = original

    assert result.type == "prompt"
    assert result.value == "do one"
    assert read_recent_feedback(base_dir=tmp_path)[-1]["event"] == "accepted_started"


def test_forecast_dismiss_uses_controller(tmp_path: Path) -> None:
    controller = FakeController()
    ctx = SimpleNamespace(cwd=tmp_path, intent_forecast_controller=controller)

    result = _forecast_call("dismiss", ctx)

    assert result.value == "Forecast dismissed."
    assert controller.dismissed is True


def test_forecast_run_does_not_append_to_conversation(tmp_path: Path) -> None:
    conversation = Conversation()
    conversation.messages = [Message(role="user", content="continue the task")]
    ctx = SimpleNamespace(
        cwd=tmp_path,
        workspace_root=tmp_path,
        conversation=conversation,
        provider=None,
    )

    result = _forecast_call("run", ctx)

    assert result.type == "text"
    assert "Forecast" in result.value
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "continue the task"


def test_forecast_run_appends_unified_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conversation = Conversation()
    conversation.messages = [Message(role="user", content="continue the task")]
    ctx = SimpleNamespace(
        cwd=tmp_path,
        workspace_root=tmp_path,
        conversation=conversation,
        provider=None,
    )

    _forecast_call("run", ctx)
    _forecast_call("run", ctx)

    rows = read_forecast_history(cwd=tmp_path)
    assert len(rows) == 2
    assert {row["trigger"] for row in rows} == {"slash"}
