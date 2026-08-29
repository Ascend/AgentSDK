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

"""End-to-end tests for the Phase 2 dialog screens.

We use Textual's ``App.run_test`` harness to push each dialog onto a
lightweight host screen, drive a few key presses, and assert the
dismissal result. The tests deliberately avoid coupling to the full
``ClawCodexTUI`` boot path (which pulls in providers and the agent
loop) so failures stay scoped to dialog behaviour.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static

from src.tui.screens import (
    CostThresholdScreen,
    EffortPickerScreen,
    ExitFlowScreen,
    ForecastPickerScreen,
    HistoryEntry,
    HistorySearchScreen,
    IdleReturnScreen,
    ModelPickerScreen,
    ThemePickerScreen,
    fuzzy_score,
)
from clawcodex_ext.intent_forecast.messages import ForecastResult, ForecastSuggestion
from clawcodex_ext.tui.app import ClawCodexTUI


class _Host(Screen):
    def compose(self) -> ComposeResult:
        yield Static("host")


class _DialogHost(App):
    """Minimal harness that boots straight into a blank screen so tests
    can push dialogs without dragging in the full TUI.
    """

    def on_mount(self) -> None:
        self.push_screen(_Host())


def _push(app: App, screen) -> asyncio.Future:
    """Push ``screen`` and return a future that resolves with the dismissal.

    Textual's ``wait_for_dismiss=True`` path requires an active
    worker, so we register an explicit callback instead.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def _callback(result):
        if not future.done():
            future.set_result(result)

    app.push_screen(screen, callback=_callback)
    return future


# ------------------------------------------------------------------
# ForecastPicker
# ------------------------------------------------------------------


def _forecast_result() -> ForecastResult:
    return ForecastResult(
        generated=True,
        fingerprint="fp",
        suggestions=[
            ForecastSuggestion(
                id="s1",
                title="Run focused tests",
                prompt="Run the focused tests for intent forecast.",
                reason="The implementation changed testable command paths.",
                confidence=0.82,
            ),
            ForecastSuggestion(
                id="s2",
                title="Review wiring",
                prompt="Review the forecast wiring.",
                confidence=0.61,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_forecast_picker_accepts_selected_suggestion():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = ForecastPickerScreen(_forecast_result())
        result_future = _push(app, picker)
        await pilot.pause()
        assert picker._select is not None
        assert picker._select.current.value == "1"
        await pilot.press("down")
        await pilot.press("enter")
        result = await result_future
        assert result == "2"


@pytest.mark.asyncio
async def test_forecast_picker_cancel_resolves_none():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = ForecastPickerScreen(_forecast_result())
        result_future = _push(app, picker)
        await pilot.pause()
        await pilot.press("escape")
        result = await result_future
        assert result is None


# ------------------------------------------------------------------
# ModelPicker
# ------------------------------------------------------------------


def test_tui_model_picker_uses_refreshed_provider_models(monkeypatch, tmp_path):
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    app = object.__new__(ClawCodexTUI)
    app.provider = SimpleNamespace(
        discover_available_models=lambda: ["provider-live-model"],
        get_available_models=lambda: ["provider-stale-model"],
    )
    app.provider_name = "test-provider"
    app.model = "configured-model"

    app._list_available_models()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        models = app._list_available_models()
        if "provider-live-model" in models:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("refreshed provider models were not published")
    assert models == ["configured-model", "provider-live-model"]


def test_tui_model_picker_uses_fallback_while_catalog_refreshes(monkeypatch, tmp_path):
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    started = threading.Event()
    release = threading.Event()

    def discover():
        started.set()
        release.wait(timeout=2)
        return ["provider-live-model"]

    app = object.__new__(ClawCodexTUI)
    app.provider = SimpleNamespace(
        get_available_models=lambda: ["provider-fallback-model"],
        discover_available_models=discover,
    )
    app.provider_name = "test-provider"
    app.model = "provider-fallback-model"

    before = time.perf_counter()
    models = app._list_available_models()
    elapsed = time.perf_counter() - before

    assert elapsed < 0.25
    assert models == ["provider-fallback-model"]
    assert "refresh" in app._model_discovery_warning.lower()
    assert started.wait(timeout=0.5)
    release.set()


def test_tui_model_picker_schedules_discovery_off_ui_thread():
    app = object.__new__(ClawCodexTUI)
    app.provider = SimpleNamespace()
    app.provider_name = "test-provider"
    app.model = "test-model"
    scheduled = []
    app.run_worker = lambda awaitable, **kwargs: scheduled.append((awaitable, kwargs))
    app._list_available_models = lambda: (_ for _ in ()).throw(AssertionError("discovery must not run synchronously"))

    app._open_model_picker(SimpleNamespace())

    assert len(scheduled) == 1
    awaitable, options = scheduled[0]
    assert options["exclusive"] is True
    assert options["group"] == "model-catalog"
    awaitable.close()


@pytest.mark.asyncio
async def test_tui_model_picker_discards_catalog_after_provider_switch(monkeypatch):
    app = object.__new__(ClawCodexTUI)
    original_provider = SimpleNamespace()
    app.provider = original_provider
    app.provider_name = "old-provider"
    app.model = "old-model"
    app._model_discovery_warning = None
    pushed = []
    app.push_screen = lambda *args, **kwargs: pushed.append((args, kwargs))

    async def switch_during_discovery(func, *args):
        app.provider = SimpleNamespace()
        app.provider_name = "new-provider"
        app.model = "new-model"
        return ["old-provider-model"], None

    monkeypatch.setattr("clawcodex_ext.tui.app.asyncio.to_thread", switch_during_discovery)

    await app._discover_and_open_model_picker(
        SimpleNamespace(),
        provider=original_provider,
        provider_name="old-provider",
        current_model="old-model",
    )

    assert pushed == []


def test_tui_model_picker_records_discovery_fallback(monkeypatch, tmp_path):
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    app = object.__new__(ClawCodexTUI)

    def fail_discovery():
        raise RuntimeError("TLS EOF")

    app.provider = SimpleNamespace(
        discover_available_models=fail_discovery,
        get_available_models=fail_discovery,
    )
    app.provider_name = "openai-codex"
    app.model = "gpt-current"
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: {"default_model": "gpt-fallback"},
    )

    assert app._list_available_models() == ["gpt-current", "gpt-fallback"]
    assert "TLS EOF" in app._model_discovery_warning


def test_tui_command_context_reinstalls_runtime_commands_after_builtins(monkeypatch, tmp_path):
    app = object.__new__(ClawCodexTUI)
    app._command_context = None
    app.workspace_root = tmp_path
    app.session = SimpleNamespace(conversation=SimpleNamespace())
    app.provider = SimpleNamespace()
    app.tool_registry = SimpleNamespace()
    app.tool_context = SimpleNamespace()
    app.runtime_context = SimpleNamespace()
    app._intent_forecast_controller = None
    app.app_state = SimpleNamespace()
    calls = []

    monkeypatch.setattr(
        "src.command_system.builtins.register_builtin_commands",
        lambda registry=None: calls.append("builtins"),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.runtime_commands.register_runtime_commands",
        lambda registry=None: calls.append("runtime"),
    )
    monkeypatch.setattr("src.command_system.load_and_register_skills", lambda registry=None: None)
    monkeypatch.setattr(
        "clawcodex_ext.cli.tool_cmd.register_tool_commands",
        lambda registry=None, tool_registry=None: None,
    )
    monkeypatch.setattr(
        "src.command_system.engine.create_command_context",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr("src.cost_tracker.CostTracker", lambda: SimpleNamespace())
    monkeypatch.setattr("src.history.HistoryLog", lambda: SimpleNamespace())

    assert app._ensure_command_context() is not None
    assert calls == ["builtins", "runtime"]


@pytest.mark.asyncio
async def test_model_picker_resolves_with_selected_model(tmp_path):
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = ModelPickerScreen(
            models=["gpt-4o", "claude-sonnet", "glm-4.5"],
            current_model="claude-sonnet",
        )
        result_future = _push(app, picker)
        await pilot.pause()
        # Cursor should start at the current model.
        assert picker._select is not None
        assert picker._select.current.value == "claude-sonnet"
        await pilot.press("down")
        await pilot.press("enter")
        result = await result_future
        assert result == "glm-4.5"


@pytest.mark.asyncio
async def test_model_picker_cancel_resolves_none():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = ModelPickerScreen(models=["a", "b"], current_model="a")
        result_future = _push(app, picker)
        await pilot.pause()
        await pilot.press("escape")
        result = await result_future
        assert result is None


# ------------------------------------------------------------------
# EffortPicker
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_effort_picker_returns_persisted_true_on_select():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = EffortPickerScreen(current="medium")
        result_future = _push(app, picker)
        await pilot.pause()
        await pilot.press("enter")  # confirm "medium"
        effort, persisted = await result_future
        assert persisted is True
        assert effort == "medium"


@pytest.mark.asyncio
async def test_effort_picker_auto_maps_to_none():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = EffortPickerScreen(current="high")
        result_future = _push(app, picker)
        await pilot.pause()
        await pilot.press("home")  # jump to "Auto"
        await pilot.press("enter")
        effort, persisted = await result_future
        assert persisted is True
        assert effort is None


# ------------------------------------------------------------------
# HistorySearch
# ------------------------------------------------------------------


def test_fuzzy_score_ranks_substring_above_subsequence():
    substring_match, substring_score = fuzzy_score("git status", "status")
    subsequence_match, subsequence_score = fuzzy_score("greatest", "gst")
    assert substring_match and subsequence_match
    assert substring_score > subsequence_score


def test_fuzzy_score_missing_subsequence_returns_false():
    matched, score = fuzzy_score("abcdef", "xyz")
    assert matched is False and score == 0


def test_fuzzy_score_empty_query_matches_everything():
    matched, score = fuzzy_score("whatever", "")
    assert matched is True and score == 0


@pytest.mark.asyncio
async def test_history_search_resolves_with_selected_prompt():
    entries = [
        HistoryEntry(prompt="git status"),
        HistoryEntry(prompt="git log"),
        HistoryEntry(prompt="rm -rf node_modules"),
    ]
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = HistorySearchScreen(entries=entries)
        result_future = _push(app, dlg)
        await pilot.pause()
        # Type "git" to filter — both matching entries should remain.
        for ch in "git":
            await pilot.press(ch)
        await pilot.pause()
        assert dlg._list is not None
        labels = [opt.label for opt in dlg._list.options]
        assert all("git" in label for label in labels)
        await pilot.press("enter")
        result = await result_future
        assert result in ("git status", "git log")


@pytest.mark.asyncio
async def test_history_search_escape_returns_none():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = HistorySearchScreen(entries=[HistoryEntry(prompt="anything")])
        result_future = _push(app, dlg)
        await pilot.pause()
        await pilot.press("escape")
        result = await result_future
        assert result is None


# ------------------------------------------------------------------
# CostThreshold / IdleReturn / ExitFlow / ThemePicker
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_threshold_enter_resolves_true():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = CostThresholdScreen(provider="openai", amount_usd=7.25)
        result_future = _push(app, dlg)
        await pilot.pause()
        await pilot.press("enter")
        assert (await result_future) is True


@pytest.mark.asyncio
async def test_idle_return_selects_clear():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = IdleReturnScreen(idle_minutes=12, total_input_tokens=120_000)
        result_future = _push(app, dlg)
        await pilot.pause()
        await pilot.press("down")  # move to "Start a new conversation"
        await pilot.press("enter")
        assert (await result_future) == "clear"


@pytest.mark.asyncio
async def test_exit_flow_cancel_returns_cancel():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = ExitFlowScreen(has_inflight_work=False)
        result_future = _push(app, dlg)
        await pilot.pause()
        await pilot.press("escape")
        assert (await result_future) == "cancel"


@pytest.mark.asyncio
async def test_theme_picker_selects_light():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        dlg = ThemePickerScreen(themes=["auto", "dark", "light", "claude"], current="dark")
        result_future = _push(app, dlg)
        await pilot.pause()
        await pilot.press("down")  # cursor starts at "dark" (current)
        await pilot.press("enter")
        assert (await result_future) == "light"
