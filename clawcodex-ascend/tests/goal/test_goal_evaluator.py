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
# pylint: disable=attribute-defined-outside-init,no-name-in-module,use-implicit-booleaness-not-comparison,using-constant-test
"""Independent goal-completion evaluator tests."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from clawcodex_ext.goal import evaluator as goal_evaluator
from clawcodex_ext.goal.evaluator import (
    GoalEvaluationError,
    evaluate_goal,
)
from clawcodex_ext.goal.model import ThreadGoal, ThreadGoalStatus
from clawcodex_ext.multimodel import MultiModelRouter, ParallelStrategy, ProviderSlot
from clawcodex_ext.utils.abort_controller import AbortController


def _goal() -> ThreadGoal:
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)
    return ThreadGoal(
        thread_id="thread-1",
        goal_id="goal-1",
        objective="all focused tests pass",
        status=ThreadGoalStatus.ACTIVE,
        token_budget=None,
        tokens_used=0,
        time_used_seconds=0,
        created_at=now,
        updated_at=now,
    )


class RecordingProvider:
    def __init__(self, content: object, usage: object = None) -> None:
        self.content = content
        self.usage = usage
        self.calls: list[tuple[list[dict[str, object]], list[object], dict[str, object]]] = []

    async def chat_async(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((messages, tools, kwargs))
        return SimpleNamespace(content=self.content, usage=self.usage)


@pytest.mark.asyncio
async def test_evaluate_goal_uses_tool_free_provider_call_and_parses_result() -> None:
    provider = RecordingProvider(
        '{"met": false, "reason": "one regression still fails"}',
        {"input_tokens": 12, "output_tokens": 4},
    )

    evaluation = await evaluate_goal(
        provider,
        _goal(),
        [{"role": "assistant", "content": "I ran the focused suite."}],
    )

    assert evaluation.met is False
    assert evaluation.reason == "one regression still fails"
    assert evaluation.usage == {"input_tokens": 12, "output_tokens": 4}
    assert len(provider.calls) == 1
    sent_messages, sent_tools, sent_kwargs = provider.calls[0]
    assert sent_tools == []
    assert [message["role"] for message in sent_messages] == ["system", "user"]
    assert "all focused tests pass" in str(sent_messages[1]["content"])
    assert "system" not in sent_kwargs
    assert "I ran the focused suite." in str(sent_messages)
    assert sent_kwargs["max_tokens"] == 256
    assert sent_kwargs["temperature"] == 0


@pytest.mark.asyncio
async def test_evaluate_goal_uses_inline_prompt_and_anthropic_small_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider('{"met": true, "reason": "done"}')
    provider.provider_name = "anthropic"  # type: ignore[attr-defined]
    monkeypatch.setenv("ANTHROPIC_SMALL_FAST_MODEL", "claude-test-haiku")

    await evaluate_goal(provider, _goal(), [])

    sent_messages, _sent_tools, sent_kwargs = provider.calls[0]
    assert [message.get("role") for message in sent_messages] == ["system", "user"]
    assert "all focused tests pass" in str(sent_messages[1]["content"])
    assert "system" not in sent_kwargs
    assert sent_kwargs["model"] == "claude-test-haiku"


@pytest.mark.asyncio
async def test_evaluate_goal_uses_inline_prompt_for_native_gemini() -> None:
    class NativeGeminiProvider(RecordingProvider):
        pass

    NativeGeminiProvider.__module__ = "clawcodex_ext.providers.native.gemini_adapter"
    provider = NativeGeminiProvider('{"met": true, "reason": "done"}')

    await evaluate_goal(provider, _goal(), [])

    sent_messages, _sent_tools, sent_kwargs = provider.calls[0]
    assert [message["role"] for message in sent_messages] == ["system", "user"]
    assert "all focused tests pass" in str(sent_messages[1]["content"])
    assert "system" not in sent_kwargs


@pytest.mark.asyncio
async def test_evaluate_goal_supports_provider_agnostic_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider('{"met": true, "reason": "done"}')
    monkeypatch.setenv("CLAWCODEX_GOAL_EVALUATOR_MODEL", "fast-evaluator-model")

    await evaluate_goal(provider, _goal(), [])

    _sent_messages, _sent_tools, sent_kwargs = provider.calls[0]
    assert sent_kwargs["model"] == "fast-evaluator-model"


@pytest.mark.asyncio
async def test_evaluate_goal_omits_unsupported_tuning_kwargs_for_openai_codex() -> None:
    class OpenAICodexProvider(RecordingProvider):
        pass

    OpenAICodexProvider.__module__ = "clawcodex_ext.providers.openai_codex_provider"
    provider = OpenAICodexProvider('{"met": true, "reason": "done"}')

    await evaluate_goal(provider, _goal(), [])

    _sent_messages, _sent_tools, sent_kwargs = provider.calls[0]
    assert "max_output_tokens" not in sent_kwargs
    assert "max_tokens" not in sent_kwargs
    assert "temperature" not in sent_kwargs


@pytest.mark.asyncio
async def test_evaluate_goal_projects_bounded_text_without_media_base64() -> None:
    provider = RecordingProvider('{"met": true, "reason": "done"}')
    image_base64 = "IMAGE_SECRET_" + "A" * 8_000
    document_base64 = "DOCUMENT_SECRET_" + "B" * 8_000
    nested_base64 = "NESTED_SECRET_" + "C" * 2_000
    oversized_proof = "proof line " * 2_000
    messages = [
        *({"role": "assistant", "content": "additional evidence " + "word " * 2_000} for _ in range(20)),
        {
            "role": "user",
            "type": "attachment",
            "content": [
                {"type": "text", "text": oversized_proof},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": document_base64,
                    },
                },
                {
                    "type": "tool_use",
                    "name": "inspect",
                    "input": {
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": nested_base64,
                        }
                    },
                },
            ],
            "attachments": [{"base64": image_base64}],
        },
    ]

    await evaluate_goal(provider, _goal(), messages)

    sent_messages, _sent_tools, _sent_kwargs = provider.calls[0]
    prompt = str(sent_messages[1]["content"])
    assert image_base64 not in prompt
    assert document_base64 not in prompt
    assert nested_base64 not in prompt
    assert "IMAGE_SECRET_" not in prompt
    assert "DOCUMENT_SECRET_" not in prompt
    assert "NESTED_SECRET_" not in prompt
    assert "[image omitted media_type=image/png]" in prompt
    assert "[document omitted media_type=application/pdf]" in prompt
    assert "[content truncated]" in prompt
    assert "earlier messages omitted from evaluator evidence" in prompt
    assert len(prompt) < 26_000


@pytest.mark.asyncio
async def test_evaluate_goal_safely_omits_cyclic_and_deep_content() -> None:
    provider = RecordingProvider('{"met": true, "reason": "done"}')
    cyclic: dict[str, object] = {"type": "tool_result"}
    cyclic["content"] = cyclic

    deep: dict[str, object] = {"type": "tool_result", "content": "leaf evidence"}
    for _ in range(50):
        deep = {"type": "tool_result", "content": deep}

    await evaluate_goal(
        provider,
        _goal(),
        [{"role": "assistant", "content": [cyclic, deep]}],
    )

    sent_messages, _sent_tools, _sent_kwargs = provider.calls[0]
    prompt = str(sent_messages[1]["content"])
    assert "[cyclic content omitted]" in prompt
    assert "[nested content omitted]" in prompt
    assert len(prompt) < 26_000


@pytest.mark.asyncio
async def test_evaluate_goal_enforces_shared_projection_node_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingProvider('{"met": true, "reason": "done"}')
    monkeypatch.setattr(goal_evaluator, "_MAX_PROJECTION_NODES", 5)

    await evaluate_goal(
        provider,
        _goal(),
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "inspect",
                        "input": {"one": "1", "two": "2", "three": "3", "four": "4"},
                    }
                ],
            }
        ],
    )

    sent_messages, _sent_tools, _sent_kwargs = provider.calls[0]
    assert "[evidence node limit reached]" in str(sent_messages[1]["content"])


@pytest.mark.asyncio
async def test_evaluate_goal_wraps_projection_failures_before_provider_call() -> None:
    class ExplodingContent(dict):
        def get(self, key, default=None):  # type: ignore[no-untyped-def]
            del key, default
            raise RuntimeError("malformed transcript mapping")

    provider = RecordingProvider('{"met": true, "reason": "done"}')

    with pytest.raises(
        GoalEvaluationError,
        match="evidence projection failed: malformed transcript mapping",
    ):
        await evaluate_goal(
            provider,
            _goal(),
            [{"role": "assistant", "content": ExplodingContent()}],
        )

    assert provider.calls == []


@pytest.mark.asyncio
async def test_evaluate_goal_selects_one_codex_router_slot_without_unsupported_tuning_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CodexSlotProvider(RecordingProvider):
        pass

    CodexSlotProvider.__module__ = "clawcodex_ext.providers.openai_codex_provider"

    class ExplodingAggregator:
        async def aggregate(self, results, context):  # type: ignore[no-untyped-def]
            del results, context
            raise AssertionError("goal evaluator must bypass router aggregation")

    codex = CodexSlotProvider('{"met": true, "reason": "verified"}')
    unused = RecordingProvider("aggregated prose that is not strict JSON")
    router = MultiModelRouter(
        [
            ProviderSlot("codex", codex, model="codex-slot-model"),
            ProviderSlot("other", unused),
        ],
        ParallelStrategy(),
        ExplodingAggregator(),
    )
    monkeypatch.delenv("CLAWCODEX_GOAL_EVALUATOR_MODEL", raising=False)

    evaluation = await evaluate_goal(router, _goal(), [])

    assert evaluation.met is True
    assert evaluation.reason == "verified"
    assert len(codex.calls) == 1
    assert unused.calls == []
    _sent_messages, sent_tools, sent_kwargs = codex.calls[0]
    assert sent_tools == []
    assert sent_kwargs["model"] == "codex-slot-model"
    assert "max_output_tokens" not in sent_kwargs
    assert "max_tokens" not in sent_kwargs
    assert "temperature" not in sent_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '```json\n{"met": true, "reason": "done"}\n```',
        '{"met": 1, "reason": "done"}',
        '{"met": true}',
        '{"met": true, "reason": 42}',
        '{"met": true, "reason": "   "}',
        '{"met": true, "reason": "done", "extra": true}',
    ],
)
async def test_evaluate_goal_rejects_non_strict_json(content: str) -> None:
    provider = RecordingProvider(content)

    with pytest.raises(GoalEvaluationError):
        await evaluate_goal(provider, _goal(), [])


@pytest.mark.asyncio
async def test_evaluate_goal_preserves_usage_when_response_is_invalid() -> None:
    provider = RecordingProvider(
        "not json",
        {"input_tokens": 9, "output_tokens": 2},
    )

    with pytest.raises(GoalEvaluationError) as raised:
        await evaluate_goal(provider, _goal(), [])

    assert raised.value.usage == {"input_tokens": 9, "output_tokens": 2}


@pytest.mark.asyncio
async def test_evaluate_goal_wraps_provider_failures() -> None:
    class FailingProvider:
        async def chat_async(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            del messages, tools, kwargs
            raise RuntimeError("provider unavailable")

    with pytest.raises(GoalEvaluationError, match="provider unavailable"):
        await evaluate_goal(FailingProvider(), _goal(), [])


@pytest.mark.asyncio
async def test_evaluate_goal_cancellation_returns_without_recording_result() -> None:
    started = asyncio.Event()

    class WaitingProvider:
        async def chat_async(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            del messages, tools, kwargs
            started.set()
            await asyncio.Future()

    controller = AbortController()
    task = asyncio.create_task(
        evaluate_goal(
            WaitingProvider(),
            _goal(),
            [],
            abort_signal=controller.signal,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pytest.fail("provider did not start before cancellation deadline")
    controller.abort("user_interrupt")

    try:
        await asyncio.wait_for(task, timeout=1)
        pytest.fail("expected cancellation but evaluator completed normally")
    except asyncio.TimeoutError:
        pytest.fail("evaluator timed out instead of propagating cancellation")
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_evaluate_goal_cancels_inherited_sync_provider_without_executor_wait() -> None:
    from clawcodex_ext.providers.base import BaseProvider, ChatResponse

    started = threading.Event()
    release = threading.Event()

    class BlockingSyncProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__(api_key="test", model="test-model")

        def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            del messages, tools, kwargs
            started.set()
            release.wait(timeout=5)
            return ChatResponse(
                content='{"met": true, "reason": "done"}',
                model="test-model",
                usage={},
                finish_reason="end_turn",
            )

        def chat_stream(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            del messages, tools, kwargs
            if False:
                yield ""

        def get_available_models(self) -> list[str]:
            return ["test-model"]

    controller = AbortController()
    task = asyncio.create_task(
        evaluate_goal(
            BlockingSyncProvider(),
            _goal(),
            [],
            abort_signal=controller.signal,
        )
    )
    try:
        if not await asyncio.to_thread(started.wait, 1):
            pytest.fail("sync provider did not start before cancellation deadline")
        controller.abort("user_interrupt")
        try:
            await asyncio.wait_for(task, timeout=0.5)
            pytest.fail("expected cancellation but evaluator completed normally")
        except asyncio.TimeoutError:
            pytest.fail("evaluator timed out instead of propagating cancellation")
        except asyncio.CancelledError:
            pass
    finally:
        release.set()
