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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Workflow extractor base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..scan_context import SourceScanContext
from .models import (
    DecisionSpec,
    ExtractedStage,
    GateSpec,
    StageContract,
    Transition,
    WorkflowGraph,
)

logger = logging.getLogger(__name__)


class WorkflowExtractorBase(ABC):
    def __init__(
        self,
        scan: SourceScanContext | None = None,
        *,
        mode: str = "fwa",
        allow_coarse: bool = False,
    ) -> None:
        self._scan = scan
        self._mode = mode
        self._allow_coarse = allow_coarse

    @abstractmethod
    def extract_stages(self, source_dir: Path) -> list[ExtractedStage]: ...

    @abstractmethod
    def extract_transitions(self, source_dir: Path) -> list[Transition]: ...

    @abstractmethod
    def extract_gates(self, source_dir: Path) -> dict[int, GateSpec]: ...

    @abstractmethod
    def extract_decisions(self, source_dir: Path) -> dict[int, DecisionSpec]: ...

    @abstractmethod
    def extract_contracts(self, source_dir: Path) -> dict[int, StageContract]: ...

    def extract(self, source_dir: Path) -> WorkflowGraph:
        stages = self.extract_stages(source_dir)
        transitions = self.extract_transitions(source_dir) if stages else []
        gates = self.extract_gates(source_dir) if stages else {}
        if self._mode == "hybrid":
            logger.debug("Skipping decision extraction in hybrid mode")
            decisions: dict[int, DecisionSpec] = {}
        else:
            decisions = self.extract_decisions(source_dir) if stages else {}
        contracts = self.extract_contracts(source_dir) if stages else {}

        quality = "full"
        if stages and any(s.inferred for s in stages):
            quality = "coarse" if all(s.inferred for s in stages) else "partial"
        elif stages and not transitions:
            quality = "partial"

        return WorkflowGraph(
            stages=stages,
            transitions=transitions,
            gates=gates,
            decisions=decisions,
            contracts=contracts,
            source_dir=str(source_dir),
            extraction_quality=quality,
        )
