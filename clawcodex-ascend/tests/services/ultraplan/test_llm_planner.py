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

# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import asyncio

import pytest

from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.services.ultraplan import CheckKind, PlannerFailedError
from clawcodex_ext.services.ultraplan.llm_planner import LLMPlanner, PlannerContext


class FakeProvider(BaseProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="test", model="fake-model")
        self.responses = responses
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, D401
        value = self.responses[self.calls]
        self.calls += 1
        return ChatResponse(content=value, model="fake-model", usage={}, finish_reason="stop")

    def chat_stream(self, messages, tools=None, **kwargs):  # noqa: ANN001
        yield ""

    def get_available_models(self) -> list[str]:
        return ["fake-model"]


def _plan_json(**overrides) -> str:
    data = {
        "title": "Refactor executor",
        "goal": "Refactor executor",
        "sub_plans": [
            {
                "id": "sp1",
                "title": "Inspect",
                "description": "Inspect current behavior",
                "steps": [
                    {
                        "id": "s1",
                        "title": "Read code",
                        "description": "Read the executor module",
                        "kind": "research",
                        "criteria": [
                            {
                                "id": "c1",
                                "description": "file exists",
                                "kind": CheckKind.FILE_EXISTS.value,
                                "target": "clawcodex_ext/services/ultraplan/executor.py",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    data.update(overrides)
    return json.dumps(data)


def test_generate_plan_retries_invalid_json() -> None:
    provider = FakeProvider(["not json", _plan_json()])
    planner = LLMPlanner(provider)
    result = asyncio.run(
        planner.generate_plan(PlannerContext(user_prompt="refactor executor", cwd="C:/WorkSpace/clawcodex"))
    )
    assert result.plan.title == "Refactor executor"
    assert result.retry_count == 1
    assert provider.calls == 2


def test_generate_plan_rejects_dangerous_acceptance() -> None:
    provider = FakeProvider(
        [
            _plan_json(
                sub_plans=[
                    {
                        "id": "sp1",
                        "title": "Bad",
                        "description": "Bad",
                        "steps": [
                            {
                                "id": "s1",
                                "title": "Bad",
                                "description": "Bad",
                                "kind": "verify",
                                "criteria": [
                                    {
                                        "id": "c1",
                                        "description": "bad",
                                        "kind": "shell_command",
                                        "target": "rm -rf /",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            ),
            _plan_json(),
        ]
    )
    planner = LLMPlanner(provider)
    result = asyncio.run(planner.generate_plan(PlannerContext(user_prompt="x", cwd=".")))
    assert result.retry_count == 1


def test_generate_plan_fails_after_retry_budget() -> None:
    provider = FakeProvider(["not json", "still not json"])
    planner = LLMPlanner(provider)
    with pytest.raises(PlannerFailedError):
        asyncio.run(planner.generate_plan(PlannerContext(user_prompt="x", cwd=".")))
