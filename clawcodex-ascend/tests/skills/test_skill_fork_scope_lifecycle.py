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

"""Regression tests for forked-skill active-stack lifetime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.agent.subagent_context import create_subagent_context
from clawcodex_ext.query.query import (
    _begin_skill_runtime_scope,
    _restore_skill_runtime_scope,
)
from clawcodex_ext.skills.invocation import (
    SkillInvocationErrorCode,
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationService,
)
from clawcodex_ext.skills.model import Skill
from clawcodex_ext.tool_system.context import ToolContext


@pytest.mark.parametrize(
    ("scope_pending", "scope_active"),
    [(True, False), (False, True), (False, False)],
)
def test_subagent_marks_inherited_active_skill_scope_pending(
    tmp_path: Path,
    scope_pending: bool,
    scope_active: bool,
) -> None:
    parent = ToolContext(workspace_root=tmp_path)
    parent.active_skill_names = ("pure-fork",)
    parent.skill_scope_pending = scope_pending
    parent.skill_scope_active = scope_active

    child = create_subagent_context(parent)

    assert child.active_skill_names == ("pure-fork",)
    assert child.skill_scope_pending is True
    assert child.skill_scope_active is False


def test_child_query_preserves_stack_and_rejects_same_skill_before_fork(
    tmp_path: Path,
) -> None:
    parent = ToolContext(workspace_root=tmp_path)
    parent.active_skill_names = ("pure-fork",)
    parent.skill_scope_pending = True
    child = create_subagent_context(parent)

    _begin_skill_runtime_scope(child)
    assert child.active_skill_names == ("pure-fork",)
    assert child.skill_scope_active is True

    launched = False

    def should_not_launch(*_args: Any) -> Any:
        nonlocal launched
        launched = True
        raise AssertionError("recursive skill reached the Agent runner")

    skill = Skill(
        name="pure-fork",
        description="pure context fork",
        content="run the fork",
        context="fork",
    )
    service = SkillInvocationService(
        resolver=lambda name, _context: skill if name == "pure-fork" else None,
        recorder=lambda *_args: None,
        fork_executor=should_not_launch,
    )

    result = service.invoke(
        SkillInvocationRequest(
            "pure-fork",
            origin=SkillInvocationOrigin.MODEL,
        ),
        child,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is SkillInvocationErrorCode.RECURSIVE_INVOCATION
    assert launched is False

    _restore_skill_runtime_scope(child)
    assert child.active_skill_names == ()
