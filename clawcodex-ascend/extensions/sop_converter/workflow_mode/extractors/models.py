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

"""Workflow graph IR models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedStage:
    id: int
    name: str
    label: str
    source_class: str | None = None
    source_value: int | None = None
    file_path: str | None = None
    entry_function: str | None = None
    description: str = ""
    capability_profile: Any | None = None
    inferred: bool = False


@dataclass
class Transition:
    from_stage: int
    to_stage: int
    condition: str | None = None
    is_default: bool = True


@dataclass
class GateSpec:
    stage_id: int
    approval_mode: str = "manual"
    description: str = ""
    source_name: str | None = None


@dataclass
class OutcomeSpec:
    next_stage: int | None = None  # None = unknown, needs manual fill
    rollback_to: int | None = None
    max_times: int | None = None
    on_exhaust: str = "rollback"


@dataclass
class DecisionSpec:
    stage_id: int
    outcomes: dict[str, OutcomeSpec] = field(default_factory=dict)
    source_func: str | None = None
    inferred: bool = False


@dataclass
class StageContract:
    stage_id: int
    input_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    dod: str = ""
    source_class: str | None = None


@dataclass
class WorkflowGraph:
    stages: list[ExtractedStage]
    transitions: list[Transition]
    gates: dict[int, GateSpec]
    decisions: dict[int, DecisionSpec]
    contracts: dict[int, StageContract]
    source_dir: str
    extraction_quality: str = "full"

    def is_empty(self) -> bool:
        return len(self.stages) == 0
