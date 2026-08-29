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

import asyncio
import json
from pathlib import Path

from clawcodex_ext.command_system.types import CommandContext
from clawcodex_ext.command_system.builtins import get_builtin_commands
from clawcodex_ext.command_system.ultraplan_command import ULTRAPLAN_COMMAND
from clawcodex_ext.providers.base import BaseProvider, ChatResponse


def test_ultraplan_is_registered_builtin() -> None:
    names = {cmd.name for cmd in get_builtin_commands()}
    assert "ultraplan" in names


class FakeProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="fake")

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return ChatResponse(
            content=json.dumps(
                {
                    "id": "cmd-plan",
                    "title": "Command plan",
                    "goal": "Make a plan",
                    "sub_plans": [
                        {
                            "id": "sp1",
                            "title": "Work",
                            "description": "Do work",
                            "steps": [
                                {
                                    "id": "s1",
                                    "title": "Step",
                                    "description": "Do the step",
                                    "kind": "implement",
                                    "criteria": [],
                                }
                            ],
                        }
                    ],
                }
            ),
            model="fake",
            usage={},
            finish_reason="stop",
        )

    def chat_stream(self, messages, tools=None, **kwargs):  # noqa: ANN001
        yield ""

    def get_available_models(self) -> list[str]:
        return ["fake"]


def test_ultraplan_create_command_persists_plan(tmp_path) -> None:
    context = CommandContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        config={"ultraplan_dir": str(tmp_path / "ultraplan")},
        provider=FakeProvider(),
    )
    result = asyncio.run(ULTRAPLAN_COMMAND.call("create Make a plan", context))
    assert "Created ultraplan cmd-plan" in result.value
    assert Path(tmp_path / "ultraplan" / "plans" / "cmd-plan.json").exists()


def test_ultraplan_create_respects_llm_feature_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ULTRAPLAN_LLM_PLANNER", "off")
    context = CommandContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        config={"ultraplan_dir": str(tmp_path / "ultraplan")},
        provider=FakeProvider(),
    )
    result = asyncio.run(ULTRAPLAN_COMMAND.call("create Make a plan", context))
    assert "ULTRAPLAN_LLM_PLANNER is disabled" in result.value


def test_ultraplan_remote_respects_feature_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ULTRAPLAN_REMOTE", "off")
    context = CommandContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        config={"ultraplan_dir": str(tmp_path / "ultraplan")},
    )
    result = asyncio.run(ULTRAPLAN_COMMAND.call("run --remote http://localhost:9999 p1", context))
    assert "ULTRAPLAN_REMOTE is disabled" in result.value
