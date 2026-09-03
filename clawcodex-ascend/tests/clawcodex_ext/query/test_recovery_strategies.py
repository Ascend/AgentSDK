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

"""Tests for clawcodex_ext/query/recovery_strategies.py (P102-B)."""

from __future__ import annotations

import pytest

from clawcodex_ext.query.recovery_strategies import (
    MAX_OUTPUT_TOKENS_ESCALATE,
    MAX_OUTPUT_TOKENS_RECOVERY,
    PROMPT_TOO_LONG_FALLBACK,
    MEDIA_SIZE_FALLBACK,
    RecoveryContext,
    clear_recovery_strategies,
    find_recovery_strategies,
    register_recovery_strategy,
)
from clawcodex_ext.query.transitions import QueryState
from clawcodex_ext.types.messages import AssistantMessage


@pytest.fixture(autouse=True)
def _clear_strategies():
    """Isolate custom strategies while restoring production defaults."""
    clear_recovery_strategies()
    yield
    # The module stays cached for the rest of the pytest session, so merely
    # clearing here would permanently remove production recovery behavior.
    clear_recovery_strategies()
    from clawcodex_ext.query.recovery_strategies import _register_builtin_strategies

    _register_builtin_strategies()


class TestRegisterRecoveryStrategy:
    def test_builtin_strategies_registered(self):
        """Verify that built-in strategies are registered on import."""
        # Re-register builtins since fixture cleared them
        from clawcodex_ext.query.recovery_strategies import _register_builtin_strategies

        _register_builtin_strategies()
        strategies = find_recovery_strategies("max_output_tokens", QueryState(messages=[], tool_use_context=None))
        names = {s.name for s in strategies}
        assert MAX_OUTPUT_TOKENS_ESCALATE in names
        assert MAX_OUTPUT_TOKENS_RECOVERY in names
        assert PROMPT_TOO_LONG_FALLBACK in names
        assert MEDIA_SIZE_FALLBACK in names

    def test_custom_strategy_registration(self):
        def my_strategy(ctx):
            return None

        register_recovery_strategy("my_custom", my_strategy, priority=5)
        strategies = find_recovery_strategies("any", QueryState(messages=[], tool_use_context=None))
        assert any(s.name == "my_custom" for s in strategies)
        assert strategies[0].priority == 5

    def test_priority_ordering(self):
        def low_priority(ctx):
            return None

        def high_priority(ctx):
            return None

        register_recovery_strategy("low", low_priority, priority=100)
        register_recovery_strategy("high", high_priority, priority=1)
        strategies = find_recovery_strategies("any", QueryState(messages=[], tool_use_context=None))
        assert strategies[0].name == "high"
        assert strategies[1].name == "low"


class TestMaxOutputTokensEscalate:
    def test_escalate_first_hit(self):
        from clawcodex_ext.query.recovery_strategies import _max_output_tokens_escalate
        from clawcodex_ext.query.query import ESCALATED_MAX_TOKENS

        state = QueryState(
            messages=[],
            tool_use_context=None,
            max_output_tokens_override=None,
            max_output_tokens_recovery_count=0,
        )
        ctx = RecoveryContext(
            state=state,
            last_message=None,
            config=None,
            params=None,
            messages=[],
            assistant_messages=[],
            error_type="max_output_tokens",
        )
        result = _max_output_tokens_escalate(ctx)
        assert result is not None
        new_state, yield_msgs = result
        assert new_state is not None
        assert new_state.max_output_tokens_override == ESCALATED_MAX_TOKENS
        assert yield_msgs == []

    def test_not_applicable_when_already_recovering(self):
        from clawcodex_ext.query.recovery_strategies import _max_output_tokens_escalate

        state = QueryState(
            messages=[],
            tool_use_context=None,
            max_output_tokens_override=64000,
            max_output_tokens_recovery_count=0,
        )
        ctx = RecoveryContext(
            state=state,
            last_message=None,
            config=None,
            params=None,
            messages=[],
            assistant_messages=[],
            error_type="max_output_tokens",
        )
        result = _max_output_tokens_escalate(ctx)
        assert result is None


class TestPromptTooLongFallback:
    def test_fallback_when_already_attempted(self):
        from clawcodex_ext.query.recovery_strategies import _prompt_too_long_fallback

        msg = AssistantMessage(content="too long")
        msg._api_error = "prompt_too_long"
        state = QueryState(
            messages=[],
            tool_use_context=None,
            has_attempted_reactive_compact=True,
        )
        ctx = RecoveryContext(
            state=state,
            last_message=msg,
            config=None,
            params=None,
            messages=[],
            assistant_messages=[],
            error_type="prompt_too_long",
        )
        result = _prompt_too_long_fallback(ctx)
        assert result is not None
        new_state, yield_msgs = result
        assert new_state is None
        assert len(yield_msgs) == 1

    def test_not_applicable_when_not_attempted(self):
        from clawcodex_ext.query.recovery_strategies import _prompt_too_long_fallback

        state = QueryState(
            messages=[],
            tool_use_context=None,
            has_attempted_reactive_compact=False,
        )
        ctx = RecoveryContext(
            state=state,
            last_message=None,
            config=None,
            params=None,
            messages=[],
            assistant_messages=[],
            error_type="prompt_too_long",
        )
        result = _prompt_too_long_fallback(ctx)
        assert result is None


class TestClearRecoveryStrategies:
    def test_clear(self):
        def my_strategy(ctx):
            return None

        register_recovery_strategy("my", my_strategy)
        clear_recovery_strategies()
        strategies = find_recovery_strategies("any", QueryState(messages=[], tool_use_context=None))
        assert strategies == []
