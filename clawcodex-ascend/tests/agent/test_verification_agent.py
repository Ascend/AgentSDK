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

"""Contract coverage for the built-in verification agent."""

from __future__ import annotations


# pylint: disable=E0611
from clawcodex_ext.agent.agent_definitions import (
    VERIFICATION_AGENT,
    find_agent_by_type,
    get_built_in_agents,
)
from clawcodex_ext.agent.agent_tool_utils import resolve_agent_tools
from clawcodex_ext.agent.constants import VERIFICATION_AGENT_TYPE
from clawcodex_ext.agent.prompt import get_agent_system_prompt
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.tools.agent import _resolve_agent_background


def test_verification_agent_is_registered_as_builtin_one_shot() -> None:
    agent = find_agent_by_type(get_built_in_agents(), VERIFICATION_AGENT_TYPE)

    assert agent is VERIFICATION_AGENT
    assert agent.source == "built-in"
    assert agent.model == "inherit"
    assert agent.background is True
    assert agent.color == "red"


def test_verification_agent_prompt_requires_runtime_evidence_and_verdict() -> None:
    prompt = get_agent_system_prompt(VERIFICATION_AGENT)

    assert "verification specialist" in prompt
    assert "at least one relevant adversarial probe" in prompt
    assert "Reading code is not runtime verification" in prompt
    assert "CRITICAL: This is a VERIFICATION-ONLY task" in prompt
    assert "VERDICT: PASS" in prompt
    assert "VERDICT: FAIL" in prompt
    assert "VERDICT: PARTIAL" in prompt


def test_verification_agent_filters_project_mutation_and_nested_agents() -> None:
    registry = build_default_registry()
    resolved = resolve_agent_tools(
        VERIFICATION_AGENT,
        registry.list_tools(),
        is_async=False,
    )
    names = {tool.name for tool in resolved.resolved_tools}

    assert {"Read", "Bash"}.issubset(names)
    assert {
        "Agent",
        "Edit",
        "Write",
        "NotebookEdit",
        "ExitPlanMode",
        "SkillSearch",
    }.isdisjoint(names)


def test_verification_agent_background_default_can_be_forced_foreground() -> None:
    assert _resolve_agent_background({}, VERIFICATION_AGENT) is True
    assert (
        _resolve_agent_background(
            {"run_in_background": False},
            VERIFICATION_AGENT,
        )
        is True
    )
    assert (
        _resolve_agent_background(
            {"_force_foreground": True, "run_in_background": True},
            VERIFICATION_AGENT,
        )
        is False
    )
