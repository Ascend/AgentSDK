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

from types import SimpleNamespace
from unittest.mock import Mock


from src.entrypoints.tui import TUIOptions


def test_downstream_tui_app_subclasses_upstream_app():
    from clawcodex_ext.tui.app import ClawCodexExtTUI
    from src.tui.app import ClawCodexTUI

    assert issubclass(ClawCodexExtTUI, ClawCodexTUI)


def test_downstream_tui_entrypoint_uses_downstream_app(monkeypatch):
    from clawcodex_ext.tui import entrypoint
    from clawcodex_ext.tui.app import ClawCodexExtTUI

    runner = Mock(return_value=7)
    monkeypatch.setattr(entrypoint, "_run_tui_with_app", runner)

    options = TUIOptions()

    assert entrypoint.run_tui(options) == 7
    runner.assert_called_once()
    call_args = runner.call_args
    assert call_args[0][0] is options
    assert call_args[1].get("app_cls") is ClawCodexExtTUI


def test_tui_entrypoint_marks_matching_session_as_resumed(monkeypatch, tmp_path):
    from clawcodex_ext.tui import entrypoint
    from src.agent.conversation import Conversation
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry

    captured = {}

    class FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(entrypoint, "_textual_available", lambda: True)
    monkeypatch.setattr(entrypoint, "_register_tui_signal_save", lambda _session: None)
    monkeypatch.setattr(entrypoint, "_print_resume_hint", lambda _session: None)
    session = SimpleNamespace(
        session_id="resume-me",
        conversation=Conversation(),
        save=lambda: None,
    )
    provider = SimpleNamespace(model="test-model", provider_name="test-provider")
    options = TUIOptions(workspace_root=tmp_path)

    result = entrypoint._run_tui_with_app(
        options,
        app_cls=FakeApp,
        provider=provider,
        session=session,
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        resume_session_id="resume-me",
    )

    assert result == 0
    assert captured["session_was_resumed"] is True

    created = SimpleNamespace(
        session_id="fresh-session",
        conversation=Conversation(),
        save=lambda: None,
    )
    monkeypatch.setattr(entrypoint.Session, "resume", lambda _session_id: None)
    monkeypatch.setattr(
        entrypoint.Session,
        "create",
        lambda _provider_name, _model: created,
    )
    captured.clear()

    result = entrypoint._run_tui_with_app(
        options,
        app_cls=FakeApp,
        provider=provider,
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        resume_session_id="missing-session",
    )

    assert result == 0
    assert captured["session_was_resumed"] is False


def test_tui_app_installs_runtime_observer_when_context_is_available(monkeypatch, tmp_path):
    from clawcodex_ext.tui.app import ClawCodexTUI
    from src.agent.conversation import Conversation
    from src.tool_system.context import ToolContext
    from src.tool_system.registry import ToolRegistry

    installed = []
    monkeypatch.setattr(
        "clawcodex_ext.frontend.tui_extensions.install_tui_extensions",
        lambda app, runtime: installed.append((app, runtime)),
    )
    provider = SimpleNamespace(model="test-model")
    runtime = SimpleNamespace()
    session = SimpleNamespace(session_id="test-session", conversation=Conversation())

    app = ClawCodexTUI(
        provider=provider,
        provider_name="test-provider",
        workspace_root=tmp_path,
        tool_registry=ToolRegistry(),
        tool_context=ToolContext(workspace_root=tmp_path),
        session=session,
        runtime_context=runtime,
    )

    assert installed == [(app, runtime)]


def test_tui_skill_slash_uses_user_invocation(monkeypatch):
    from clawcodex_ext.tui.app import ClawCodexTUI

    tool_context = object()
    app = SimpleNamespace(tool_context=tool_context, submit_to_agent=Mock())
    transcript = Mock()
    calls = []

    def fake_run_user_invoked_skill(name, args, context):
        calls.append((name, args, context))
        return SimpleNamespace(
            is_error=False,
            output={
                "success": True,
                "commandName": "hello",
                "prompt": "Hello bob",
            },
        )

    monkeypatch.setattr(
        "clawcodex_ext.tool_system.tools.skill.run_user_invoked_skill",
        fake_run_user_invoked_skill,
    )

    handled = ClawCodexTUI._try_run_skill_slash(app, "/hello bob", transcript)

    assert handled is True
    assert calls == [("hello", "bob", tool_context)]
    app.submit_to_agent.assert_called_once_with("Hello bob")
    transcript.append_system.assert_called_once_with("Launching skill: hello", style="info")


def test_tui_forked_skill_slash_renders_result_without_second_query(monkeypatch):
    from clawcodex_ext.tui.app import ClawCodexTUI

    tool_context = object()
    app = SimpleNamespace(tool_context=tool_context, submit_to_agent=Mock())
    transcript = Mock()

    monkeypatch.setattr(
        "clawcodex_ext.tool_system.tools.skill.run_user_invoked_skill",
        lambda name, args, context: SimpleNamespace(
            is_error=False,
            output={
                "success": True,
                "status": "fork",
                "commandName": name,
                "result": "runtime evidence\nVERDICT: PASS",
            },
        ),
    )

    handled = ClawCodexTUI._try_run_skill_slash(app, "/verify target.txt", transcript)

    assert handled is True
    app.submit_to_agent.assert_not_called()
    transcript.append_assistant.assert_called_once_with(
        "runtime evidence\nVERDICT: PASS",
        agent_name="verify",
    )


def test_upstream_tui_entrypoint_uses_upstream_app(monkeypatch):
    """Verify upstream run_tui constructs ClawCodexTUI, not ClawCodexExtTUI."""
    import src.entrypoints.tui as tui_entrypoint
    from src.tui.app import ClawCodexTUI

    # Monkey-patch _textual_available to True so run_tui proceeds.
    monkeypatch.setattr(tui_entrypoint, "_textual_available", lambda: True)

    # Use the explicit provider-factory seam so this entrypoint contract does
    # not depend on local credentials.
    fake_provider = Mock()
    fake_provider.provider_name = "anthropic"

    called = False

    def mock_init(self, **kwargs):
        nonlocal called
        called = True
        assert type(self) is ClawCodexTUI
        monkeypatch.setattr(self, "run", Mock(return_value=None))
        object.__setattr__(self, "_exit_code", 0)
        object.__setattr__(self, "_shutdown_managed_tasks", Mock(return_value=None))

    monkeypatch.setattr(ClawCodexTUI, "__init__", mock_init)

    options = TUIOptions(
        provider_factory=lambda: fake_provider,
        provider_name="anthropic",
    )
    result = tui_entrypoint.run_tui(options)
    assert result == 0
    assert called, "ClawCodexTUI was never instantiated"
