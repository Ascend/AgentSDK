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

"""DECISION handler.

Handles workflow decision points -- multi-outcome branching, loops, and convergence detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .workflow_state import StageNode, StageResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A single decision record."""

    stage_id: int
    outcome: str
    timestamp: str = ""
    next_stage: int | None = None


@dataclass
class DecisionHistory:
    """Decision history tracker."""

    records: list[DecisionRecord] = field(default_factory=list)

    def record(self, stage_id: int, outcome: str, next_stage: int | None = None) -> None:
        self.records.append(
            DecisionRecord(
                stage_id=stage_id,
                outcome=outcome,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                next_stage=next_stage,
            )
        )

    def count(self, outcome: str, stage_id: int) -> int:
        """Count occurrences of a specific stage+outcome."""
        return sum(1 for r in self.records if r.stage_id == stage_id and r.outcome == outcome)

    def is_degenerate(self, outcome: str, stage_id: int, window: int = 5) -> bool:
        """Detect a degenerate loop: the same outcome appearing N consecutive times."""
        recent = [r for r in self.records[-window:] if r.stage_id == stage_id]
        if len(recent) < window:
            return False
        return all(r.outcome == outcome for r in recent)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": r.stage_id,
                "outcome": r.outcome,
                "timestamp": r.timestamp,
                "next_stage": r.next_stage,
            }
            for r in self.records
        ]

    @classmethod
    def from_dict_list(cls, records: list[dict[str, Any]]) -> "DecisionHistory":
        """Rebuild decision history from a checkpoint record list."""
        history = cls()
        for r in records:
            history.records.append(
                DecisionRecord(
                    stage_id=int(r.get("stage", 0)),
                    outcome=r.get("outcome", ""),
                    timestamp=r.get("timestamp", ""),
                    next_stage=r.get("next_stage"),
                )
            )
        return history

    def count_by_stage(self, stage_id: int) -> int:
        """Count all decision records for a given stage."""
        return sum(1 for r in self.records if r.stage_id == stage_id)

    def counts(self) -> dict[int, int]:
        """Return a mapping of stage_id to decision counts."""
        counts: dict[int, int] = {}
        for r in self.records:
            counts[r.stage_id] = counts.get(r.stage_id, 0) + 1
        return counts


@dataclass
class DecisionResult:
    """Decision processing result."""

    outcome: str
    next_stage: int | None = None
    rollback_to: int | None = None
    exhausted: bool = False
    converged: bool = False
    reason: str = ""


class DecisionHandler:
    """DECISION handler.

    Core logic:
    1. Parse the decision outcome from LLM output (proceed / pivot / refine / ...)
    2. Loop-count check (max_times)
    3. Convergence detection (degenerate loop detection)
    """

    def __init__(self) -> None:
        self._history = DecisionHistory()

    @property
    def history(self) -> DecisionHistory:
        return self._history

    def resolve(
        self,
        node: StageNode,
        result: StageResult,
    ) -> DecisionResult:
        """Parse a decision result.

        Args:
            node: decision stage node
            result: stage execution result (contains the LLM-provided decision_outcome)

        Returns:
            DecisionResult: decision result.
        """
        outcome = result.decision_outcome
        if outcome is None:
            outcome = "proceed"
        decision_spec = node.decision_outcomes.get(outcome, {})

        # Loop-count check
        max_times = decision_spec.get("max_times")
        if max_times is not None:
            times = self._history.count(outcome, node.id)
            if times >= max_times:
                rollback_to = decision_spec.get("rollback_to", node.depends_on[0] if node.depends_on else None)
                self._history.record(node.id, outcome, None)
                return DecisionResult(
                    outcome=outcome,
                    exhausted=True,
                    rollback_to=rollback_to,
                    reason=f"Max retries ({max_times}) exceeded for outcome '{outcome}'",
                )

        # Convergence check
        convergence_check = decision_spec.get("convergence_check", False)
        if convergence_check:
            if self._history.is_degenerate(outcome, node.id):
                self._history.record(node.id, outcome, None)
                return DecisionResult(
                    outcome=outcome,
                    converged=True,
                    reason=f"Convergence detected: degenerate loop for outcome '{outcome}'",
                )

        next_stage = decision_spec.get("next")
        rollback_to = decision_spec.get("rollback_to")

        self._history.record(node.id, outcome, next_stage)

        return DecisionResult(
            outcome=outcome,
            next_stage=next_stage,
            rollback_to=rollback_to,
            reason=f"Decision resolved: {outcome} -> next_stage={next_stage}",
        )

    def reset(self) -> None:
        """Reset decision history."""
        self._history = DecisionHistory()
