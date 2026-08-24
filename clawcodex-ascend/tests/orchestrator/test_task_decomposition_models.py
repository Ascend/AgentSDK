# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Focused unit tests for the F-118 task graph contracts."""

import pytest

from extensions.orchestrator.task_decomposition.models import Subtask, TaskPlan


def _valid_plan() -> TaskPlan:
    return TaskPlan(
        goal="migrate feature",
        subtasks=(
            Subtask("discover", "Discover", "Inspect the change"),
            Subtask(
                "implement",
                "Implement",
                "Apply the change",
                depends_on=("discover",),
                affected_files=("src/app.py",),
            ),
        ),
        waves=(("discover",), ("implement",)),
        max_parallel=1,
    )


def test_valid_plan_serializes_runtime_fields() -> None:
    plan = _valid_plan()
    plan.validate(max_subtasks=4, max_waves=3)

    payload = plan.to_dict()
    assert payload["waves"] == [["discover"], ["implement"]]
    assert payload["subtasks"][0]["status"] == "pending"
    assert payload["subtasks"][1]["affected_files"] == ["src/app.py"]


def test_plan_rejects_unknown_dependency() -> None:
    plan = TaskPlan(
        goal="invalid",
        subtasks=(Subtask("one", "One", "", depends_on=("missing",)),),
        waves=(("one",),),
        max_parallel=1,
    )

    with pytest.raises(ValueError, match="unknown ids"):
        plan.validate(max_subtasks=2, max_waves=2)


def test_plan_rejects_parallel_file_conflict() -> None:
    plan = TaskPlan(
        goal="conflict",
        subtasks=(
            Subtask("one", "One", "", affected_files=("src/app.py",)),
            Subtask("two", "Two", "", affected_files=("src/app.py",)),
        ),
        waves=(("one", "two"),),
        max_parallel=2,
    )

    with pytest.raises(ValueError, match="file conflict"):
        plan.validate(max_subtasks=3, max_waves=2)
