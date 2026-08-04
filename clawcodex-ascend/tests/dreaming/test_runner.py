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

"""Tests for ``clawcodex_ext.dreaming.runner`` — F-100.

The Phase A runner is intentionally a stub. Tests cover the stub
contract (callback invocation, no-op result) and the factory swap
mechanism.
"""

from __future__ import annotations

import pytest

from clawcodex_ext.dreaming.runner import (
    DreamRunResult,
    DreamRunnerUnavailable,
    run_dream_consolidation,
    set_dream_runner_factory,
)


@pytest.fixture(autouse=True)
def _clear_runner_factory() -> None:
    """Reset the runner factory between tests so each starts from the stub."""
    set_dream_runner_factory(None)
    yield  # type: ignore[misc]
    set_dream_runner_factory(None)


def test_stub_returns_empty_result_when_no_callback() -> None:
    result = run_dream_consolidation("any prompt")
    assert isinstance(result, DreamRunResult)
    assert result.files_touched == []
    assert result.usage == {}
    assert result.summary == "(stub run)"


def test_stub_invokes_callback_once() -> None:
    calls: list[dict] = []

    def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
        calls.append({"text": text, "tool_use_count": tool_use_count, "touched_paths": touched_paths})

    run_dream_consolidation("any prompt", on_message=on_msg)
    assert len(calls) == 1
    assert calls[0] == {"text": "", "tool_use_count": 0, "touched_paths": []}


def test_stub_swallows_callback_exceptions() -> None:
    def on_msg(**_kwargs) -> None:
        raise RuntimeError("callback exploded")

    # Must not propagate — the stub treats the callback as best-effort.
    result = run_dream_consolidation("any prompt", on_message=on_msg)
    assert result.summary == "(stub run)"


def test_factory_swap_replaces_stub() -> None:
    captured: dict = {}

    def factory():
        def runner(prompt, on_message):
            captured["prompt"] = prompt
            captured["on_message"] = on_message
            return DreamRunResult(
                files_touched=["X.md"],
                usage={"output_tokens": 42},
                summary="custom",
            )

        return runner

    set_dream_runner_factory(factory)
    result = run_dream_consolidation("hello")
    assert result.files_touched == ["X.md"]
    assert result.usage == {"output_tokens": 42}
    assert result.summary == "custom"
    assert captured["prompt"] == "hello"


def test_factory_construct_failure_raises_dream_runner_unavailable() -> None:
    def bad_factory():
        raise RuntimeError("construction failed")

    set_dream_runner_factory(bad_factory)
    with pytest.raises(DreamRunnerUnavailable, match="construction failed"):
        run_dream_consolidation("hello")


def test_factory_runner_exception_raises_dream_runner_unavailable() -> None:
    def bad_factory():
        def runner(_prompt, _on_message):
            raise RuntimeError("LLM exploded")

        return runner

    set_dream_runner_factory(bad_factory)
    with pytest.raises(DreamRunnerUnavailable, match="LLM exploded"):
        run_dream_consolidation("hello")


def test_clear_factory_restores_stub() -> None:
    def factory():
        def runner(_prompt, _on_message):
            return DreamRunResult(summary="custom")

        return runner

    set_dream_runner_factory(factory)
    assert run_dream_consolidation("x").summary == "custom"
    set_dream_runner_factory(None)
    assert run_dream_consolidation("x").summary == "(stub run)"
