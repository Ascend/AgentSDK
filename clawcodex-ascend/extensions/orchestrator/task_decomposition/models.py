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
"""Validated task graph used by F-118 swarm execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Subtask:
    id: str
    title: str
    description: str
    depends_on: tuple[str, ...] = ()
    verification: str = ""
    affected_files: tuple[str, ...] = ()
    token_cost: float = 0.0
    budget: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["depends_on"] = list(self.depends_on)
        row["affected_files"] = list(self.affected_files)
        row.update(
            {
                "status": "pending",
                "evidence": "",
                "started_at": None,
                "completed_at": None,
            }
        )
        return row


@dataclass(frozen=True)
class TaskPlan:
    goal: str
    subtasks: tuple[Subtask, ...]
    waves: tuple[tuple[str, ...], ...]
    max_parallel: int
    version: int = 1

    def validate(self, *, max_subtasks: int, max_waves: int) -> None:
        if not self.subtasks:
            raise ValueError("task plan must contain at least one subtask")
        if len(self.subtasks) > max_subtasks:
            raise ValueError(f"task plan has {len(self.subtasks)} subtasks; limit is {max_subtasks}")
        ids = [task.id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task plan contains duplicate subtask ids")
        known = set(ids)
        for task in self.subtasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(f"subtask {task.id} depends on unknown ids: {sorted(unknown)}")
            if task.id in task.depends_on:
                raise ValueError(f"subtask {task.id} cannot depend on itself")
        flattened = [task_id for wave in self.waves for task_id in wave]
        if set(flattened) != known or len(flattened) != len(known):
            raise ValueError("execution waves must contain every subtask exactly once")
        if len(self.waves) > max_waves:
            raise ValueError(f"task plan has {len(self.waves)} waves; limit is {max_waves}")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least one")
        if any(len(wave) > self.max_parallel for wave in self.waves):
            raise ValueError("execution wave exceeds max_parallel")
        self._check_file_conflicts()
        wave_index = {task_id: index for index, wave in enumerate(self.waves) for task_id in wave}
        for task in self.subtasks:
            for dependency in task.depends_on:
                if wave_index[dependency] >= wave_index[task.id]:
                    raise ValueError(f"subtask {task.id} must run after dependency {dependency}")

    def _check_file_conflicts(self) -> None:
        """Check that no two parallel subtasks (same wave) write to the same file.

        Only subtasks with non-empty ``affected_files`` are considered.
        """
        file_owners: dict[str, str] = {}
        for wave in self.waves:
            for task_id in wave:
                task = next(t for t in self.subtasks if t.id == task_id)
                if not task.affected_files:
                    continue
                for f in task.affected_files:
                    if f in file_owners:
                        prev_id = file_owners[f]
                        raise ValueError(
                            f"file conflict: {f} is claimed by both subtask {prev_id} and subtask {task_id} (same wave)"
                        )
                    file_owners[f] = task_id
            file_owners.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "goal": self.goal,
            "max_parallel": self.max_parallel,
            "subtasks": [task.to_dict() for task in self.subtasks],
            "waves": [list(wave) for wave in self.waves],
        }
