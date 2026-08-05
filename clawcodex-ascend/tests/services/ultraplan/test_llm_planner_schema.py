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

# pylint: disable=no-name-in-module
from __future__ import annotations

import pytest

from clawcodex_ext.services.ultraplan import PlannerFailedError
from clawcodex_ext.services.ultraplan.llm_planner import PlannerContext, _validate_plan_shape


def _valid_plan() -> dict:
    return {
        "title": "Plan",
        "goal": "Goal",
        "sub_plans": [
            {
                "id": "sp1",
                "title": "Sub",
                "description": "Sub",
                "steps": [
                    {
                        "id": "s1",
                        "title": "Step",
                        "description": "Step",
                        "kind": "implement",
                        "depends_on": [],
                        "criteria": [
                            {
                                "id": "c1",
                                "description": "Criterion",
                                "kind": "custom",
                                "target": "manual",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_schema_accepts_valid_plan() -> None:
    _validate_plan_shape(_valid_plan(), PlannerContext(user_prompt="x", cwd="."))


def test_schema_rejects_unknown_fields() -> None:
    data = _valid_plan()
    data["surprise"] = True
    with pytest.raises(PlannerFailedError, match="unknown fields"):
        _validate_plan_shape(data, PlannerContext(user_prompt="x", cwd="."))


def test_schema_rejects_invalid_step_kind() -> None:
    data = _valid_plan()
    data["sub_plans"][0]["steps"][0]["kind"] = "explode"
    with pytest.raises(PlannerFailedError, match="kind is invalid"):
        _validate_plan_shape(data, PlannerContext(user_prompt="x", cwd="."))


def test_schema_rejects_cross_or_unknown_dependency() -> None:
    data = _valid_plan()
    data["sub_plans"][0]["steps"][0]["depends_on"] = ["missing"]
    with pytest.raises(PlannerFailedError, match="depends_on references unknown"):
        _validate_plan_shape(data, PlannerContext(user_prompt="x", cwd="."))
