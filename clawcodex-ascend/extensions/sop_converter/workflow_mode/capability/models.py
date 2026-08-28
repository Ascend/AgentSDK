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

"""Capability profile data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CapabilityKind(str, Enum):
    LLM_CALL = "llm"
    ACADEMIC_API = "academic_api"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_exec"
    FILE_IO = "file_io"
    EXTERNAL_CLI = "external_cli"
    DOMAIN_SPECIFIC = "domain"
    DATA_PROCESSING = "data_proc"
    HTTP_API = "http_api"


class ExecutionMode(str, Enum):
    AGENT_NATIVE = "agent_native"
    WRAPPER = "wrapper"
    HYBRID = "hybrid"


@dataclass
class Capability:
    kind: CapabilityKind
    evidence: str = ""
    confidence: float = 1.0


@dataclass
class StageCapabilityProfile:
    stage_id: int
    capabilities: list[Capability] = field(default_factory=list)
    complexity: float = 0.0
    fragility: float = 0.0
    execution_mode: ExecutionMode = ExecutionMode.AGENT_NATIVE
    recommended_tools: list[str] = field(default_factory=list)
    mapped_skill: str | None = None
    mapped_agent: str | None = None
    mapping_confidence: float = 0.0
    entry_function: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class StageAgentMap:
    """Global stage → agent mapping shared by D/E/overview."""

    by_stage_id: dict[int, StageCapabilityProfile]
    skill_to_agent: dict[str, str]

    def agent_for_stage(self, stage_id: int) -> str | None:
        profile = self.by_stage_id.get(stage_id)
        return profile.mapped_agent if profile else None

    def profile_for_stage(self, stage_id: int) -> StageCapabilityProfile | None:
        return self.by_stage_id.get(stage_id)

    @property
    def has_mapped_stages(self) -> bool:
        """True if at least one stage matched a real skill."""
        return any(p.mapping_confidence > 0 for p in self.by_stage_id.values())
