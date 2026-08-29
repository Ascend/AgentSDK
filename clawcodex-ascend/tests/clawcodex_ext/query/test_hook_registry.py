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

"""Tests for clawcodex_ext/query/hook_registry.py (P102-D)."""

from __future__ import annotations

import pytest

from clawcodex_ext.query.hook_registry import (
    call_hooks,
    clear_hooks,
    list_hooks,
    register_loop_hook,
    unregister_loop_hook,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clean registry before each test."""
    clear_hooks()
    yield
    clear_hooks()


class TestRegisterLoopHook:
    def test_register_pre_llm(self):
        def my_hook(messages, system_prompt, state, params):
            return (messages, system_prompt)

        register_loop_hook("my_hook", my_hook, "pre_llm", priority=10)
        hooks = list_hooks("pre_llm")
        assert len(hooks) == 1
        assert hooks[0].name == "my_hook"
        assert hooks[0].phase == "pre_llm"
        assert hooks[0].priority == 10

    def test_register_same_name_replaces(self):
        def hook_a(messages, system_prompt):
            return (messages, system_prompt)

        def hook_b(messages, system_prompt):
            return (messages, system_prompt)

        register_loop_hook("same", hook_a, "pre_llm", priority=1)
        register_loop_hook("same", hook_b, "pre_llm", priority=2)
        hooks = list_hooks("pre_llm")
        assert len(hooks) == 1
        assert hooks[0].fn is hook_b

    def test_unregister_removes(self):
        def my_hook(messages, system_prompt):
            return (messages, system_prompt)

        register_loop_hook("my_hook", my_hook, "pre_llm")
        unregister_loop_hook("my_hook", "pre_llm")
        assert len(list_hooks("pre_llm")) == 0

    def test_priority_sorting(self):
        called = []

        def hook_high(m, s):
            called.append("high")
            return (m, s)

        def hook_low(m, s):
            called.append("low")
            return (m, s)

        register_loop_hook("low", hook_low, "pre_llm", priority=20)
        register_loop_hook("high", hook_high, "pre_llm", priority=5)

        call_hooks("pre_llm", [], "sys")
        assert called == ["high", "low"]


class TestCallHooks:
    def test_pre_llm_modifies_messages(self):
        def prepend_system(messages, system_prompt, **kwargs):
            return ([{"role": "system", "content": "prepended"}] + list(messages), system_prompt)

        register_loop_hook("prepend", prepend_system, "pre_llm")
        result = call_hooks("pre_llm", [{"role": "user"}], "sys")
        assert result[0][0]["role"] == "system"
        assert result[0][1]["role"] == "user"
        assert result[1] == "sys"

    def test_hook_raising_is_ignored(self):
        def bad_hook(*args, **kwargs):
            raise RuntimeError("boom")

        def good_hook(messages, system_prompt, **kwargs):
            return (messages, system_prompt + "\nmodified")

        register_loop_hook("bad", bad_hook, "pre_llm")
        register_loop_hook("good", good_hook, "pre_llm")
        result = call_hooks("pre_llm", [], "sys")
        assert result[1] == "sys\nmodified"

    def test_return_none_does_not_modify(self):
        def noop_hook(*args, **kwargs):
            return None

        register_loop_hook("noop", noop_hook, "pre_llm")
        result = call_hooks("pre_llm", ["msg"], "sys")
        assert result == (["msg"], "sys")

    def test_all_phases(self):
        for phase in (
            "pre_llm",
            "post_llm",
            "pre_tool",
            "post_tool",
            "on_turn_start",
            "on_turn_end",
        ):
            clear_hooks(phase)

            def make_hook(p):
                def hook(*args, **kwargs):
                    return args

                return hook

            register_loop_hook(f"hook_{phase}", make_hook(phase), phase)
            assert len(list_hooks(phase)) == 1


class TestClearHooks:
    def test_clear_single_phase(self):
        def hook(*args, **kwargs):
            return args

        register_loop_hook("h", hook, "pre_llm")
        clear_hooks("pre_llm")
        assert list_hooks("pre_llm") == []

    def test_clear_all(self):
        def hook(*args, **kwargs):
            return args

        register_loop_hook("h1", hook, "pre_llm")
        register_loop_hook("h2", hook, "post_llm")
        clear_hooks()
        assert list_hooks("pre_llm") == []
        assert list_hooks("post_llm") == []
