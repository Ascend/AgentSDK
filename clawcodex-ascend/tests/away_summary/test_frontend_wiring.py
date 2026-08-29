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

# pylint: disable=attribute-defined-outside-init,keyword-arg-before-vararg,no-name-in-module,ungrouped-imports,use-implicit-booleaness-not-comparison,useless-return

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from rich.markdown import Markdown

from src.agent.conversation import Conversation
from src.command_system.registry import CommandRegistry, get_command_registry
from clawcodex_ext.providers.base import ChatResponse
from src.types.messages import Message

from clawcodex_ext.away_summary.config import AwaySummaryConfig
from clawcodex_ext.away_summary.controller import AwaySummaryController
from clawcodex_ext.away_summary.messages import create_away_summary_message
from clawcodex_ext.intent_forecast.messages import ForecastResult, ForecastSuggestion


class _Console:
    def print(self, *args, **kwargs) -> None:
        return None


class _Provider:
    provider_name = "fake"
    model = "fake"

    def chat(self, **kwargs):
        return ChatResponse(content="recap", model="fake", usage={}, finish_reason="stop")


class _Session:
    session_id = "s1"

    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation
        self.saved = 0

    def save(self) -> None:
        self.saved += 1
        return None


class _Runtime:
    provider = _Provider()
    provider_name = "fake"
    tool_registry = None
    tool_context = None


class _Repl:
    def __init__(self) -> None:
        conv = Conversation()
        conv.messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        self.command_registry = CommandRegistry()
        self.session = _Session(conv)
        self.provider = _Provider()
        self.workspace_root = Path(".")
        self.console = _Console()
        self.runtime_context = _Runtime()
        self.tool_context = type("ToolContext", (), {"cwd": None, "workspace_root": Path(".")})()
        self.updated = False
        self.local_command_output: list[tuple[str, str]] = []

    def _update_built_in_commands_with_command_system(self) -> None:
        self.updated = True

    def chat(self, text: str) -> None:
        self.last_chat = text

    def handle_command(self, text: str) -> None:
        self.last_command = text

    def _print_local_command_text(self, text: str, *, command: str = "") -> None:
        self.local_command_output.append((text, command))


class _Timer:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class _Timers:
    def __init__(self) -> None:
        self.timers: list[_Timer] = []

    def call_later(self, seconds: float, callback):
        timer = _Timer(callback)
        self.timers.append(timer)
        return timer


def test_repl_extension_registers_recap(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

    repl = _Repl()
    install_repl_extensions(repl, repl.runtime_context)

    assert repl.command_registry.has("recap")
    assert repl.updated is True
    assert getattr(repl, "_away_summary_controller", None) is not None


def test_repl_extension_registers_forecast_controller() -> None:
    from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

    repl = _Repl()
    install_repl_extensions(repl, repl.runtime_context)

    assert repl.command_registry.has("forecast")
    controller = getattr(repl, "_intent_forecast_controller", None)
    assert controller is not None

    controller.display(
        ForecastResult(
            generated=True,
            fingerprint="fp",
            suggestions=[
                ForecastSuggestion(
                    id="s1",
                    title="Continue implementation",
                    prompt="Continue implementing the feature.",
                    confidence=0.8,
                )
            ],
        )
    )
    assert repl.local_command_output[-1][1] == "forecast"
    assert "Continue implementation" in repl.local_command_output[-1][0]


def test_repl_extension_starts_forecast_idle_timer(monkeypatch) -> None:
    from clawcodex_ext.frontend import repl_extensions

    mounted: list[object] = []

    class FakeForecastController:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def on_mount(self) -> None:
            mounted.append(self)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        repl_extensions,
        "IntentForecastController",
        FakeForecastController,
    )

    repl = _Repl()
    repl_extensions.install_repl_extensions(repl, repl.runtime_context)

    assert mounted
    assert getattr(repl, "_intent_forecast_controller", None) is mounted[0]
    assert repl.command_registry.has("forecast")


def test_repl_extension_does_not_install_removed_goal_controller(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )

    from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

    repl = _Repl()
    install_repl_extensions(repl, repl.runtime_context)

    assert getattr(repl, "_goal_controller", None) is None


def test_repl_auto_recap_display_prints_immediately(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    rendered: list[str] = []

    class RecordingConsole:
        def print(self, text) -> None:
            rendered.append(text)

    repl = _Repl()
    repl.console = RecordingConsole()

    from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

    install_repl_extensions(repl, repl.runtime_context)

    controller = getattr(repl, "_away_summary_controller", None)
    assert controller is not None
    controller.display("Recapitulate\nauto recap")

    assert repl.local_command_output == [("Recapitulate\nauto recap", "recap")]
    assert rendered == []


def test_repl_background_output_queue_initializes_lazily() -> None:
    from src.repl.core import ClawcodexREPL

    rendered: list[str] = []

    class RecordingConsole:
        def print(self, text) -> None:
            rendered.append(text)

    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl.console = RecordingConsole()

    repl._drain_background_outputs()
    repl._enqueue_background_output("background notice")
    repl._drain_background_outputs()

    assert rendered == ["background notice"]


def test_repl_background_recap_output_renders_markdown() -> None:
    from src.repl.core import ClawcodexREPL

    rendered: list[object] = []

    class RecordingConsole:
        def print(self, text=None, *args, **kwargs) -> None:
            rendered.append(text)

    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl.console = RecordingConsole()

    repl._enqueue_background_output("Recapitulate\n- **auto recap**")
    repl._drain_background_outputs()

    assert any(isinstance(item, Markdown) for item in rendered)


def test_tui_suggestions_include_recap_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    registry = get_command_registry()
    try:
        registry.unregister("recap")
        from clawcodex_ext.tui.commands import build_command_suggestions

        names = {item.name for item in build_command_suggestions(tmp_path)}
        assert "recap" in names
    finally:
        if registry.has("recap"):
            registry.unregister("recap")


def test_command_disabled_but_auto_controller_still_runs(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=False),
    )
    registry = CommandRegistry()
    from clawcodex_ext.away_summary.registration import register_away_summary_commands

    register_away_summary_commands(registry)
    assert not registry.has("recap")

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    provider = _Provider()
    timers = _Timers()
    controller = AwaySummaryController(
        conversation=conv,
        provider_getter=lambda: provider,
        model_getter=lambda: "fake",
        session_getter=lambda: _Session(conv),
        config_loader=lambda: AwaySummaryConfig(enabled=True, recap_command_enabled=False),
        timer_factory=timers,
    )

    controller.on_run_start()
    controller.on_run_finish()
    controller.on_assistant_turn_complete()
    timers.timers[0].fire()

    assert conv.messages[-1].subtype == "away_summary"


def _snapshot_text(transcript) -> str:
    parts: list[str] = []
    console = Console(record=True, width=100)
    for item in transcript.snapshot():
        if isinstance(item, tuple):
            for piece in item:
                console.begin_capture()
                console.print(piece)
                parts.append(console.end_capture())
        else:
            console.begin_capture()
            console.print(item)
            parts.append(console.end_capture())
    return "\n".join(parts)


async def _wait_for_repl_screen(
    app: Any,
    pilot: Any,
    attempts: int = 20,
) -> Any:
    for _ in range(attempts):
        await pilot.pause()
        screen = getattr(app, "_repl_screen", None)
        if screen is not None:
            return screen
    raise AssertionError("TUI REPL screen did not mount")


async def _wait_for_snapshot_text(
    transcript: Any,
    pilot: Any,
    needle: str,
    attempts: int = 20,
) -> str:
    rendered = ""
    for _ in range(attempts):
        await pilot.pause()
        rendered = _snapshot_text(transcript)
        if needle in rendered:
            return rendered
    raise AssertionError(f"TUI transcript did not contain {needle!r}; last snapshot: {rendered!r}")


@pytest.mark.asyncio
async def test_tui_replays_recap_history_after_screen_mount(tmp_path) -> None:
    pytest.importorskip("textual")

    from clawcodex_ext.tui.app import ClawCodexTUI
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
        create_away_summary_message(
            "manual recap",
            trigger="manual",
            fingerprint="fp",
            message_count=2,
            model="fake",
        ),
    ]

    app = ClawCodexTUI(
        provider=_Provider(),
        provider_name="fake",
        workspace_root=tmp_path,
        session=_Session(conv),
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        stream=False,
    )

    async with app.run_test() as pilot:
        screen = await _wait_for_repl_screen(app, pilot)
        rendered = await _wait_for_snapshot_text(screen.transcript, pilot, "manual recap")

    assert "hello" in rendered
    assert "hi" in rendered
    assert "Recapitulate" in rendered
    assert "manual recap" in rendered


@pytest.mark.asyncio
async def test_tui_executes_recap_slash_command(monkeypatch, tmp_path) -> None:
    pytest.importorskip("textual")
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command.load_away_summary_config",
        lambda cwd=None: AwaySummaryConfig(enabled=False),
    )

    from clawcodex_ext.tui.app import ClawCodexTUI
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry
    from src.tui.widgets import PromptInput

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    session = _Session(conv)
    app = ClawCodexTUI(
        provider=_Provider(),
        provider_name="fake",
        workspace_root=tmp_path,
        session=session,
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        stream=False,
    )

    async with app.run_test() as pilot:
        screen = await _wait_for_repl_screen(app, pilot)
        prompt = app.screen.query_one(PromptInput)
        prompt._input.value = "/recap"
        await prompt._input.action_submit()
        rendered = ""
        for _ in range(20):
            await pilot.pause()
            rendered = _snapshot_text(screen.transcript)
            if "Recapitulate" in rendered:
                break

    assert "Recapitulate" in rendered
    assert "recap" in rendered
    assert conv.messages[-1].subtype == "away_summary"
    assert session.saved >= 1


def test_tui_away_summary_controller_uses_workspace_config(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.tui.app import ClawCodexTUI
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry

    seen: list[Path | None] = []

    def fake_load_config(*, cwd=None, overrides=None):
        del overrides
        seen.append(Path(cwd) if cwd is not None else None)
        return AwaySummaryConfig(idle_seconds=10)

    monkeypatch.setattr("clawcodex_ext.tui.app.load_away_summary_config", fake_load_config)

    app = ClawCodexTUI(
        provider=_Provider(),
        provider_name="fake",
        workspace_root=tmp_path,
        session=_Session(Conversation()),
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        stream=False,
    )

    assert app._away_summary_controller is not None
    assert app._away_summary_controller.config_loader().idle_seconds == 10
    assert seen == [tmp_path]


def test_tui_away_summary_display_posts_to_ui_thread(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.tui.app import ClawCodexTUI
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry

    app = ClawCodexTUI(
        provider=_Provider(),
        provider_name="fake",
        workspace_root=tmp_path,
        session=_Session(Conversation()),
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        stream=False,
    )
    rendered: list[tuple[str, str, str]] = []
    app._repl_screen = type(
        "Screen",
        (),
        {
            "transcript": type(
                "Transcript",
                (),
                {
                    "is_mounted": True,
                    "append_system": lambda self, text, *, style="muted", render="plain": (
                        rendered.append((text, style, render))
                    ),
                },
            )()
        },
    )()
    posted: list[object] = []

    def fake_call_from_thread(callback):
        posted.append(callback)
        callback()

    monkeypatch.setattr(app, "call_from_thread", fake_call_from_thread)

    assert app._away_summary_controller is not None
    app._away_summary_controller.display("auto recap")

    assert posted
    assert rendered == [("Recapitulate\n\nauto recap", "muted", "markdown")]

    app._away_summary_controller.display("Recap\nsecond recap")
    assert rendered[-1] == ("Recap\nsecond recap", "muted", "markdown")

    app._away_summary_controller.display("Away Summary\nlegacy recap")
    assert rendered[-1] == ("Away Summary\nlegacy recap", "muted", "markdown")
