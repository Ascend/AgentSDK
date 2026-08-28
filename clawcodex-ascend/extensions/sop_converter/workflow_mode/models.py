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


"""Data models — discrimination results and thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scan_context import SourceScanContext

THRESHOLD_SDK = 0.3
THRESHOLD_FWA = 0.7

_STAGE_KEYWORDS = ("STAGE", "PHASE", "STEP", "PIPELINE")


@dataclass
class HeuristicMatch:
    name: str
    weight: float
    matched: bool
    evidence: str = ""
    score: float = 0.0


@dataclass
class DiscriminationResult:
    source_dir: str
    total_score: float
    matches: list[HeuristicMatch]
    mode: str  # sdk | hybrid | fwa
    forced: bool = False
    confidence: float = 0.0
    recommended_extractor: str = "generic"
    scan: SourceScanContext | None = None
    fwa_qualified: bool = False  # combo gate for fwa mode

    def to_dict(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "total_score": self.total_score,
            "mode": self.mode,
            "forced": self.forced,
            "confidence": self.confidence,
            "recommended_extractor": self.recommended_extractor,
            "fwa_qualified": self.fwa_qualified,
            "matches": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "matched": m.matched,
                    "evidence": m.evidence,
                    "score": m.score,
                }
                for m in self.matches
            ],
        }
