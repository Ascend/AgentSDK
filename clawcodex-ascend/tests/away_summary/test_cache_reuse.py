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

# pylint: disable=no-name-in-module,ungrouped-imports,unnecessary-lambda,use-implicit-booleaness-not-comparison
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from src.agent.conversation import Conversation
from clawcodex_ext.providers.base import ChatResponse
from src.types.messages import Message

from clawcodex_ext.away_summary.config import AwaySummaryConfig
from clawcodex_ext.away_summary.service import (
    _ForkUnavailable,
    _generate_via_chat,
    _generate_via_fork,
    AwaySummaryService,
)


class FakeProvider:
    model = "fake-model"

    def __init__(self, content: str = "chat recap", **kwargs) -> None:
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


def _conv() -> Conversation:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="fix the recap bug"),
        Message(role="assistant", content="Investigating."),
    ]
    return conv


def _fake_csp(*, with_provider: bool = True) -> SimpleNamespace:
    """Build a minimal CacheSafeParams-like object for fork-path tests."""
    ctx = SimpleNamespace(
        _active_provider=FakeProvider() if with_provider else None,
    )
    return SimpleNamespace(
        tool_use_context=ctx,
        system_prompt="you are an assistant",
        fork_context_messages=[],
    )


def _install_fork_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_text: str = "forked recap",
) -> list[Any]:
    """Replace ``run_forked_agent`` so we never hit the real query loop.

    Returns the ``captured`` list so the test can introspect what was
    forwarded to the fork primitive.
    """
    from clawcodex_ext.types.content_blocks import TextBlock
    from clawcodex_ext.types.messages import AssistantMessage

    captured: list[Any] = []

    async def fake_run_forked_agent(params):
        captured.append(params)
        return SimpleNamespace(
            messages=[AssistantMessage(content=[TextBlock(text=response_text)])],
        )

    monkeypatch.setattr(
        "clawcodex_ext.agent.forked_agent.run_forked_agent",
        fake_run_forked_agent,
    )
    return captured


def _extract_first_user_text(messages: list) -> str:
    """Return the text of the first user message in the fork payload."""
    from clawcodex_ext.types.content_blocks import TextBlock

    for msg in messages:
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, list) and content:
            block = content[0]
            if isinstance(block, TextBlock):
                return block.text
            if isinstance(block, dict):
                return str(block.get("text") or "")
        if isinstance(content, str):
            return content
    return ""


# ---------------------------------------------------------------------------
# Service-level: cache_safe_params enables the fork path
# ---------------------------------------------------------------------------


def test_service_uses_fork_when_cache_safe_params_available(monkeypatch) -> None:
    """``/recap`` with a cached CSP takes the fork path, not provider.chat.

    The away-summary formatting instructions must be forwarded in the user
    message because ``run_forked_agent`` reuses the parent's cached system
    prompt.
    """
    captured = _install_fork_stub(monkeypatch, response_text="cache-recap")

    conv = _conv()
    provider = FakeProvider(content="should-not-appear")
    service = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(enable_recap_cache=True),
    )

    result = service.generate(trigger="manual", cache_safe_params=_fake_csp())

    assert result.generated is True
    assert result.summary == "cache-recap"
    # The fork path was exercised.
    assert captured, "run_forked_agent was not called"
    # Provider.chat was NOT called — the fork output replaced it.
    assert provider.calls == []
    # The recap-specific system instructions travelled inside the user message.
    params = captured[0]
    user_text = _extract_first_user_text(params.prompt_messages)
    assert "Use `-` and ONLY `-` as the bullet marker" in user_text
    assert "你刚回来，这是之前的会话摘要" in user_text


def test_service_skips_fork_when_config_disables_cache(monkeypatch) -> None:
    """``enable_recap_cache=False`` keeps the legacy provider.chat path."""
    captured = _install_fork_stub(monkeypatch)

    conv = _conv()
    provider = FakeProvider(content="plain recap")
    service = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(enable_recap_cache=False),
    )

    result = service.generate(trigger="manual", cache_safe_params=_fake_csp())

    assert result.summary == "plain recap"
    assert provider.calls, "provider.chat should have been called"
    assert captured == []


def test_service_skips_fork_for_auto_trigger(monkeypatch) -> None:
    """The auto (idle) path never reuses the parent's cache prefix —
    fork is reserved for ``/recap`` only.
    """
    captured = _install_fork_stub(monkeypatch)

    conv = _conv()
    provider = FakeProvider(content="auto recap")
    service = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(enable_recap_cache=True),
    )

    result = service.generate(trigger="auto", cache_safe_params=_fake_csp())

    assert result.summary == "auto recap"
    assert provider.calls
    assert captured == []


def test_service_skips_fork_when_cache_safe_params_is_none(monkeypatch) -> None:
    """When the main loop hasn't saved any CSP yet, fall back to chat."""
    captured = _install_fork_stub(monkeypatch)

    conv = _conv()
    provider = FakeProvider(content="fallback recap")
    service = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(enable_recap_cache=True),
    )

    result = service.generate(trigger="manual", cache_safe_params=None)

    assert result.summary == "fallback recap"
    assert provider.calls
    assert captured == []


def test_service_falls_back_to_chat_when_fork_raises(monkeypatch) -> None:
    """A failing fork (no assistant text) drops down to provider.chat."""
    from clawcodex_ext.types.messages import AssistantMessage

    async def empty_fork(params):
        return SimpleNamespace(messages=[AssistantMessage(content=[])])

    monkeypatch.setattr(
        "clawcodex_ext.agent.forked_agent.run_forked_agent",
        empty_fork,
    )

    conv = _conv()
    provider = FakeProvider(content="post-fork chat recap")
    service = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(enable_recap_cache=True),
    )

    result = service.generate(trigger="manual", cache_safe_params=_fake_csp())

    assert result.summary == "post-fork chat recap"
    assert provider.calls


# ---------------------------------------------------------------------------
# _generate_via_fork: standalone error / fallback paths
# ---------------------------------------------------------------------------


def test_generate_via_fork_raises_without_active_provider() -> None:
    """A CSP with no ``_active_provider`` cannot be forked — fall back."""
    csp = _fake_csp(with_provider=False)
    with pytest.raises(_ForkUnavailable):
        _generate_via_fork(
            csp,
            [{"role": "user", "content": "recap"}],
            model="fake-model",
            max_output_tokens=200,
        )


def test_generate_via_fork_raises_without_tool_use_context() -> None:
    """A CSP with no ``tool_use_context`` is not forkable."""
    csp = SimpleNamespace(system_prompt="x")  # no tool_use_context
    with pytest.raises(_ForkUnavailable):
        _generate_via_fork(
            csp,
            [{"role": "user", "content": "recap"}],
            model="fake-model",
            max_output_tokens=200,
        )


def test_generate_via_fork_raises_when_no_user_message() -> None:
    """Empty user-content list → nothing to send into the fork."""
    with pytest.raises(_ForkUnavailable):
        _generate_via_fork(
            _fake_csp(),
            [{"role": "system", "content": "no user here"}],
            model="fake-model",
            max_output_tokens=200,
        )


def test_generate_via_fork_raises_when_inside_event_loop(monkeypatch) -> None:
    """Already in an event loop → ``asyncio.run`` would crash → fall back.

    We simulate "we are inside an event loop" by stubbing
    ``asyncio.get_running_loop`` to return a non-None sentinel.
    """

    class FakeLoop:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.service.asyncio.get_running_loop",
        lambda: FakeLoop(),
    )
    with pytest.raises(_ForkUnavailable):
        _generate_via_fork(
            _fake_csp(),
            [{"role": "user", "content": "recap"}],
            model="fake-model",
            max_output_tokens=200,
        )


def test_generate_via_chat_retries_once_on_exception(monkeypatch) -> None:
    """Transient provider errors must be retried before giving up."""
    calls = {"n": 0}

    class FlakyProvider:
        model = "flaky"

        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient SSL EOF")
            return ChatResponse(
                content="second-try recap",
                model="flaky",
                usage={},
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.service.time.sleep",
        lambda _seconds: None,
    )

    response = _generate_via_chat(
        FlakyProvider(),
        [{"role": "user", "content": "recap"}],
        model="flaky",
        max_output_tokens=200,
    )
    assert response.content == "second-try recap"
    assert calls["n"] == 2


def test_generate_via_chat_does_not_retry_permanent_failure(monkeypatch) -> None:
    """A deterministic provider failure must surface without a duplicate call."""
    calls = {"n": 0}

    class AlwaysFails:
        model = "always-fails"

        def chat(self, **kwargs):
            calls["n"] += 1
            raise RuntimeError("permanent failure")

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.service.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="permanent failure"):
        _generate_via_chat(
            AlwaysFails(),
            [{"role": "user", "content": "recap"}],
            model="always-fails",
            max_output_tokens=200,
        )
    assert calls["n"] == 1
