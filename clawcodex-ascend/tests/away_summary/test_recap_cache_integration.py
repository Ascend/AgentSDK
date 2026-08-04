#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=no-name-in-module,ungrouped-imports
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.agent.conversation import Conversation
from src.command_system.engine import create_command_context
from src.command_system.registry import CommandRegistry
from src.history import HistoryLog
from clawcodex_ext.providers.base import ChatResponse
from src.types.messages import Message

from clawcodex_ext.away_summary.command import (
    _try_get_last_cache_safe_params,
)
from clawcodex_ext.away_summary.config import AwaySummaryConfig
from clawcodex_ext.away_summary.registration import register_away_summary_commands


class FakeProvider:
    model = "fake-model"

    def __init__(self, content: str = "manual recap") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return ChatResponse(
            content=self.content,
            model="fake-model",
            usage={},
            finish_reason="stop",
        )


class FakeSession:
    session_id = "s1"

    def __init__(self) -> None:
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _ctx(tmp_path: Path):
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="build recap"),
        Message(role="assistant", content="done"),
    ]
    ctx = create_command_context(
        workspace_root=tmp_path,
        conversation=conv,
        history=HistoryLog(),
        provider=FakeProvider(),
    )
    ctx.session = FakeSession()  # type: ignore[attr-defined]
    return ctx


def test_try_get_last_cache_safe_params_returns_none_when_empty(monkeypatch) -> None:
    """No snapshot has been saved → helper returns None gracefully."""
    monkeypatch.setattr(
        "clawcodex_ext.agent.forked_agent.get_last_cache_safe_params",
        lambda: None,
    )
    assert _try_get_last_cache_safe_params() is None


def test_try_get_last_cache_safe_params_returns_none_when_import_fails(
    monkeypatch,
) -> None:
    """If the fork module can't be imported, we degrade silently."""
    import builtins
    import sys

    real_import = builtins.__import__
    target_module = "clawcodex_ext.agent.forked_agent"
    import_attempted = False

    for module_name in list(sys.modules):
        if module_name == target_module or module_name.startswith(f"{target_module}."):
            monkeypatch.delitem(sys.modules, module_name)

    def fake_import(name, *args, **kwargs):
        nonlocal import_attempted
        if name == target_module:
            import_attempted = True
            raise ImportError("simulated fork module unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _try_get_last_cache_safe_params() is None
    assert import_attempted is True


def test_try_get_last_cache_safe_params_returns_snapshot(monkeypatch) -> None:
    """A saved snapshot is returned verbatim."""
    sentinel = SimpleNamespace(system_prompt="snap")
    monkeypatch.setattr(
        "clawcodex_ext.agent.forked_agent.get_last_cache_safe_params",
        lambda: sentinel,
    )
    assert _try_get_last_cache_safe_params() is sentinel


# ---------------------------------------------------------------------------
# /recap command wiring
# ---------------------------------------------------------------------------


def test_recap_command_picks_up_cache_when_enabled(monkeypatch, tmp_path) -> None:
    """When config.enable_recap_cache=True and a CSP exists, /recap forwards
    it to the service so the recap can hit the fork path.
    """
    fake_csp = SimpleNamespace(system_prompt="cached")

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(
            recap_command_enabled=True,
            enable_recap_cache=True,
        ),
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command.load_away_summary_config",
        lambda cwd=None: AwaySummaryConfig(
            recap_command_enabled=True,
            enable_recap_cache=True,
        ),
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command._try_get_last_cache_safe_params",
        lambda: fake_csp,
    )

    captured = {}

    from clawcodex_ext.away_summary import service as service_mod

    real_generate = service_mod.AwaySummaryService.generate

    def spy_generate(self, *, trigger, force=False, persist=None, cache_safe_params=None):
        captured["cache_safe_params"] = cache_safe_params
        captured["trigger"] = trigger
        return real_generate(
            self,
            trigger=trigger,
            force=force,
            persist=persist,
            cache_safe_params=cache_safe_params,
        )

    monkeypatch.setattr(service_mod.AwaySummaryService, "generate", spy_generate)

    registry = CommandRegistry()
    register_away_summary_commands(registry)
    cmd = registry.get("recap")
    assert cmd is not None

    ctx = _ctx(tmp_path)
    result = cmd._call_impl("", ctx)  # type: ignore[operator]

    assert "Recapitulate" in result.value
    assert captured["cache_safe_params"] is fake_csp
    assert captured["trigger"] == "manual"


def test_recap_command_skips_cache_when_disabled(monkeypatch, tmp_path) -> None:
    """``enable_recap_cache=False`` → CSP is NOT consulted even if available."""
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(
            recap_command_enabled=True,
            enable_recap_cache=False,
        ),
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command.load_away_summary_config",
        lambda cwd=None: AwaySummaryConfig(
            recap_command_enabled=True,
            enable_recap_cache=False,
        ),
    )

    captured = {}

    def fake_try():
        captured["called"] = True
        return SimpleNamespace(system_prompt="would-have-used")

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command._try_get_last_cache_safe_params",
        fake_try,
    )

    from clawcodex_ext.away_summary import service as service_mod

    real_generate = service_mod.AwaySummaryService.generate

    def spy_generate(self, *, trigger, force=False, persist=None, cache_safe_params=None):
        captured["cache_safe_params"] = cache_safe_params
        return real_generate(
            self,
            trigger=trigger,
            force=force,
            persist=persist,
            cache_safe_params=cache_safe_params,
        )

    monkeypatch.setattr(service_mod.AwaySummaryService, "generate", spy_generate)

    registry = CommandRegistry()
    register_away_summary_commands(registry)
    cmd = registry.get("recap")
    assert cmd is not None

    cmd._call_impl("", _ctx(tmp_path))  # type: ignore[operator]

    assert captured["cache_safe_params"] is None
    assert "called" not in captured  # never even tried to read CSP
