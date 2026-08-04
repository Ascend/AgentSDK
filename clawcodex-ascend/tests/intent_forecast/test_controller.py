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

from src.agent.conversation import Conversation
from src.types.messages import Message

from clawcodex_ext.intent_forecast.config import IntentForecastConfig
from clawcodex_ext.intent_forecast.controller import IntentForecastController
from clawcodex_ext.intent_forecast.learning import read_recent_feedback
from clawcodex_ext.intent_forecast.messages import (
    ForecastResult,
    ForecastSuggestion,
    format_forecast_for_display,
)
from clawcodex_ext.intent_forecast.persistence import read_forecast_history


class FakeTimer:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []
        self.seconds: list[float] = []

    def call_later(self, seconds: float, callback):
        self.seconds.append(seconds)
        timer = FakeTimer(callback)
        self.timers.append(timer)
        return timer


def _conversation() -> Conversation:
    conv = Conversation()
    conv.messages = [Message(role="user", content="implement forecast")]
    return conv


def _empty_conversation() -> Conversation:
    conv = Conversation()
    conv.messages = []
    return conv


class FakeForecastService:
    def __init__(self, **kwargs) -> None:
        pass

    def generate(self, *, trigger: str):
        return ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[ForecastSuggestion(id="s1", title="Initial suggestion", prompt="do initial", confidence=0.8)],
        )


def test_controller_arms_on_mount_and_fires(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("clawcodex_ext.intent_forecast.controller.IntentForecastService", FakeForecastService)
    timers = FakeTimerFactory()
    displayed = []
    conv = _empty_conversation()
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=displayed.append,
        config_loader=lambda: IntentForecastConfig(idle_seconds=7),
        conversation_getter=lambda: conv,
        timer_factory=timers,
    )

    controller.on_mount()
    assert timers.seconds == [7]

    timers.timers[0].fire()
    assert displayed
    assert displayed[0].generated is True
    assert len(conv.messages) == 0
    rows = read_forecast_history(cwd=tmp_path)
    assert len(rows) == 1
    assert rows[0]["trigger"] == "auto"


def test_controller_displays_empty_auto_result(monkeypatch, tmp_path) -> None:
    class EmptyForecastService:
        def __init__(self, **kwargs) -> None:
            pass

        def generate(self, *, trigger: str):
            return ForecastResult(
                generated=False,
                fingerprint="empty",
                suggestions=[],
                reason="Forecast: no suggestions right now.",
            )

    monkeypatch.setattr(
        "clawcodex_ext.intent_forecast.controller.IntentForecastService",
        EmptyForecastService,
    )
    timers = FakeTimerFactory()
    displayed = []
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=displayed.append,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1),
        conversation_getter=_empty_conversation,
        timer_factory=timers,
    )

    controller.on_mount()
    timers.timers[0].fire()

    assert displayed
    assert displayed[0].generated is False
    assert controller.last_result is None


def test_empty_forecast_display_uses_user_friendly_status() -> None:
    result = ForecastResult(
        generated=False,
        suggestions=[],
        reason="No confident next-step suggestions are available.",
    )

    assert format_forecast_for_display(result) == "Forecast: no suggestions right now."


def test_user_interaction_cancels_timer(tmp_path) -> None:
    timers = FakeTimerFactory()
    displayed = []
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=displayed.append,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1),
        conversation_getter=_empty_conversation,
        timer_factory=timers,
    )

    controller.on_mount()
    controller.on_user_interaction("typing")
    timers.timers[0].fire()
    assert displayed == []


def test_existing_user_message_prevents_auto_timer(tmp_path) -> None:
    timers = FakeTimerFactory()
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1),
        conversation_getter=_conversation,
        timer_factory=timers,
    )

    controller.on_mount()

    assert timers.timers == []


def test_non_empty_prompt_draft_prevents_rearm_after_run_finish(tmp_path) -> None:
    timers = FakeTimerFactory()
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1),
        conversation_getter=_conversation,
        timer_factory=timers,
    )

    controller.on_prompt_draft_changed("pending draft")
    controller.on_run_start()
    controller.on_run_finish()

    assert timers.timers == []


def test_valid_run_input_prevents_rearm_after_run_finish(tmp_path) -> None:
    timers = FakeTimerFactory()
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1),
        conversation_getter=_empty_conversation,
        timer_factory=timers,
    )

    controller.on_run_start()
    controller.on_run_finish()

    # After an agent run, auto-forecast is permanently disabled
    # for the rest of the session.
    assert timers.timers == []


def test_stale_generation_is_discarded(tmp_path) -> None:
    class Provider:
        def chat(self, **kwargs):
            controller.on_user_interaction("during-provider")

            class Response:
                content = '{"suggestions":[{"title":"x","prompt":"do x","confidence":0.9}]}'

            return Response()

    timers = FakeTimerFactory()
    displayed = []
    controller = IntentForecastController(
        provider_getter=lambda: Provider(),
        model_getter=lambda: "fake",
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=displayed.append,
        config_loader=lambda: IntentForecastConfig(idle_seconds=1, feedback_enabled=False),
        conversation_getter=_empty_conversation,
        timer_factory=timers,
    )
    controller.on_mount()
    timers.timers[0].fire()
    assert displayed == []


def test_accept_submits_last_result(tmp_path) -> None:
    submitted: list[str] = []
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        submit=submitted.append,
        config_loader=lambda: IntentForecastConfig(feedback_enabled=False),
        conversation_getter=_empty_conversation,
        timer_factory=FakeTimerFactory(),
    )
    controller.remember(
        ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[ForecastSuggestion(id="s1", title="A", prompt="do accepted", confidence=0.8)],
        )
    )

    assert controller.accept(1) is True
    assert submitted


def test_accept_records_started_and_completed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.intent_forecast.learning.feedback_path",
        lambda base_dir=None: tmp_path / "intent_forecast" / "feedback.jsonl",
    )
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        submit=lambda prompt: None,
        config_loader=lambda: IntentForecastConfig(feedback_enabled=True),
        conversation_getter=_empty_conversation,
        timer_factory=FakeTimerFactory(),
    )
    controller.remember(
        ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[ForecastSuggestion(id="s1", title="A", prompt="do accepted", confidence=0.8)],
        )
    )

    assert controller.accept(1) is True
    controller.on_run_finish()

    rows = read_recent_feedback(base_dir=tmp_path)
    assert [row["event"] for row in rows[-2:]] == ["accepted_started", "accepted_completed"]
    assert rows[-1]["features"]["suggestion_kind"]


def test_accept_records_correction(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.intent_forecast.learning.feedback_path",
        lambda base_dir=None: tmp_path / "intent_forecast" / "feedback.jsonl",
    )
    controller = IntentForecastController(
        provider_getter=lambda: None,
        model_getter=lambda: None,
        session_getter=lambda: None,
        workspace_root=tmp_path,
        display=lambda result: None,
        submit=lambda prompt: None,
        config_loader=lambda: IntentForecastConfig(feedback_enabled=True),
        conversation_getter=_empty_conversation,
        timer_factory=FakeTimerFactory(),
    )
    controller.remember(
        ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[ForecastSuggestion(id="s1", title="A", prompt="do accepted", confidence=0.8)],
        )
    )

    assert controller.accept(1) is True
    controller.on_prompt_draft_changed("不是这个方向，改成先写文档")

    rows = read_recent_feedback(base_dir=tmp_path)
    assert [row["event"] for row in rows[-2:]] == ["accepted_started", "accepted_corrected"]
