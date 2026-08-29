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

"""Regression coverage for standalone versus compound ``cd`` execution."""

from __future__ import annotations


# pylint: disable=E0611
import shlex
from pathlib import Path

from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.registry import ToolCall
from src.tool_system.context import ToolContext


def _dispatch(command: str, context: ToolContext):
    assert context.tool_registry is not None
    return context.tool_registry.dispatch(
        ToolCall(name="Bash", input={"command": command}),
        context,
    )


def test_compound_cd_executes_the_remaining_shell_command(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.mkdir()
    registry = build_default_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        tool_registry=registry,
    )

    result = _dispatch(
        f"cd {shlex.quote(str(target))} && printf 'compound-ran\\n' && pwd",
        context,
    )

    assert result.is_error is False
    assert result.output["exit_code"] == 0
    assert result.output["stdout"] == f"compound-ran\n{target}\n"
    assert context.cwd == target


def test_standalone_cd_still_updates_persistent_context_cwd(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.mkdir()
    registry = build_default_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        tool_registry=registry,
    )

    cd_result = _dispatch(f"cd {shlex.quote(str(target))}", context)
    pwd_result = _dispatch("pwd", context)

    assert cd_result.is_error is False
    assert cd_result.output["stdout"] == ""
    assert context.cwd == target
    assert pwd_result.output["stdout"] == f"{target}\n"
