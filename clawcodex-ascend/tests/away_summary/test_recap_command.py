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

# pylint: disable=no-name-in-module,ungrouped-imports

from __future__ import annotations

from pathlib import Path

from src.agent.conversation import Conversation
from src.command_system.builtins import execute_command_sync
from src.command_system.engine import create_command_context
from src.command_system.registry import CommandRegistry, get_command_registry
from src.history import HistoryLog
from clawcodex_ext.providers.base import ChatResponse
from src.types.messages import Message

from clawcodex_ext.away_summary.command import _recap_call
from clawcodex_ext.away_summary.config import AwaySummaryConfig
from clawcodex_ext.away_summary.registration import register_away_summary_commands


class FakeProvider:
    model = "fake"

    def chat(self, **kwargs):
        return ChatResponse(content="manual recap", model="fake", usage={}, finish_reason="stop")


class FakeSession:
    session_id = "s1"

    def __init__(self) -> None:
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _context(tmp_path: Path):
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


def test_recap_command_generates_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.command.load_away_summary_config",
        lambda cwd=None: AwaySummaryConfig(enabled=False),
    )
    registry = CommandRegistry()
    register_away_summary_commands(registry)
    cmd = registry.get("recap")
    assert cmd is not None

    ctx = _context(tmp_path)
    result = cmd._call_impl("", ctx)  # type: ignore[operator]
    assert "Recapitulate" in result.value
    assert "manual recap" in result.value
    assert ctx.conversation.messages[-1].subtype == "away_summary"
    assert ctx.session.saved == 1
    assert cmd.run_in_thread is True


def test_recap_command_hides_provider_errors(tmp_path, monkeypatch, caplog) -> None:
    def fail_chat(**_kwargs):
        raise RuntimeError("private provider detail")

    context = _context(tmp_path)
    monkeypatch.setattr(context.provider, "chat", fail_chat)
    result = _recap_call("", context)
    assert result.value == "Recap failed. Check logs for details."
    assert "Manual Away Summary generation failed" in caplog.text


def test_recap_command_skips_sync_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        lambda: AwaySummaryConfig(recap_command_enabled=True),
    )
    registry = get_command_registry()
    register_away_summary_commands(registry)
    try:
        success, text, error = execute_command_sync("recap", "", _context(tmp_path))
    finally:
        registry.unregister("recap")

    assert success is False
    assert text is None
    assert error == "Command requires async execution: recap"


def test_recap_command_can_be_unregistered(monkeypatch) -> None:
    values = {"enabled": True}

    def _cfg():
        return AwaySummaryConfig(recap_command_enabled=values["enabled"])

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.registration.load_away_summary_config",
        _cfg,
    )
    registry = CommandRegistry()
    register_away_summary_commands(registry)
    assert registry.has("recap")

    values["enabled"] = False
    register_away_summary_commands(registry)
    assert not registry.has("recap")


def test_recap_command_has_away_and_catchup_aliases() -> None:
    """The /recap command exposes the canonical aliases."""
    monkeypatch_obj = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch_obj.setattr(
            "clawcodex_ext.away_summary.registration.load_away_summary_config",
            lambda: AwaySummaryConfig(recap_command_enabled=True),
        )
        registry = CommandRegistry()
        register_away_summary_commands(registry)
        cmd = registry.get("recap")
        assert cmd is not None
        assert "away" in cmd.aliases
        assert "catchup" in cmd.aliases
    finally:
        monkeypatch_obj.undo()
