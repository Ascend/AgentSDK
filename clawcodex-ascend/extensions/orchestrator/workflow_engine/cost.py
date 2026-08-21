#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=relative-beyond-top-level

"""Cost tracking and budget control."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import CostExceededError


@dataclass
class CostBudget:
    """Cost budget configuration."""

    max_total_usd: float = 50.0
    max_per_stage_usd: float = 10.0
    warn_threshold_pct: float = 0.8  # Warn at 80%

    def __post_init__(self) -> None:
        if not 0 < self.warn_threshold_pct < 1:
            raise ValueError(f"warn_threshold_pct must be in (0, 1), got {self.warn_threshold_pct}")

    @property
    def warn_threshold_usd(self) -> float:
        return self.max_total_usd * self.warn_threshold_pct


@dataclass
class CostTracker:
    """Cost tracker -- stage-level + global budget + warning threshold."""

    budget: CostBudget = field(default_factory=CostBudget)
    _total_usd: float = 0.0
    _stage_usd: float = 0.0
    _warned_total: bool = False
    _warned_stage: bool = False

    def reset_stage(self) -> None:
        """Reset the stage-level counter."""
        self._stage_usd = 0.0
        self._warned_stage = False

    def add(self, usd: float) -> None:
        """Add cost. Raises ValueError for negative values."""
        if usd < 0:
            raise ValueError(f"Cost must be non-negative, got {usd}")
        self._total_usd += usd
        self._stage_usd += usd

    def check_budget(self) -> list[str]:
        """Check the budget and return a list of warnings.

        Raises:
            CostExceededError: total budget exceeded.
        """
        warnings: list[str] = []

        if self._total_usd > self.budget.max_total_usd:
            raise CostExceededError(
                f"Total cost ${self._total_usd:.2f} exceeds budget ${self.budget.max_total_usd:.2f}"
            )

        if not self._warned_total and self._total_usd >= self.budget.warn_threshold_usd:
            self._warned_total = True
            warnings.append(
                f"Cost warning: ${self._total_usd:.2f} / ${self.budget.max_total_usd:.2f} "
                f"({self.budget.warn_threshold_pct:.0%} threshold reached)"
            )

        if not self._warned_stage and self._stage_usd >= self.budget.max_per_stage_usd * self.budget.warn_threshold_pct:
            self._warned_stage = True
            warnings.append(f"Stage cost warning: ${self._stage_usd:.2f} / ${self.budget.max_per_stage_usd:.2f}")

        if self._stage_usd > self.budget.max_per_stage_usd:
            raise CostExceededError(
                f"Stage cost ${self._stage_usd:.2f} exceeds per-stage budget ${self.budget.max_per_stage_usd:.2f}"
            )

        return warnings

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def stage_usd(self) -> float:
        return self._stage_usd

    def load_state(
        self,
        total_usd: float,
        stage_usd: float = 0.0,
        warned_total: bool = False,
        warned_stage: bool = False,
    ) -> None:
        """Load accumulated cost from external state (for checkpoint recovery)."""
        self._total_usd = float(total_usd)
        self._stage_usd = float(stage_usd)
        self._warned_total = bool(warned_total)
        self._warned_stage = bool(warned_stage)
