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

"""GATE handler.

Handles GATE stages -- human approval, auto threshold, and rollback.
Three approval modes: manual, auto, threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .validators import ContractValidator
from .workflow_state import StageNode, StageResult, StageStatus, WorkflowState

logger = logging.getLogger(__name__)


class GateMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"
    THRESHOLD = "threshold"


@dataclass
class GateResult:
    """GATE processing result."""

    approved: bool
    mode: GateMode
    reason: str = ""
    score: float | None = None
    stage_result: StageResult | None = None


class GateHandler:
    """GATE handler.

    Reuses:
    - ClarificationQueue (manual mode)
    - Human Review Gate (approval flow)
    """

    def __init__(
        self,
        clarification_queue: Any = None,
        journal: Any = None,
        workspace_dir: str = "",
        llm_client: Any = None,
    ) -> None:
        self._clarification_queue = clarification_queue
        self._journal = journal
        self._validator = ContractValidator(
            workspace_dir=workspace_dir,
            llm_client=llm_client,
        )

    async def process(
        self,
        stage_node: StageNode,
        state: WorkflowState,
        stage_result: StageResult,
    ) -> GateResult:
        """Process a GATE stage.

        Pick the approval strategy based on gate_mode.
        """
        mode = GateMode(stage_node.gate_mode) if stage_node.gate_mode else GateMode.MANUAL

        if mode == GateMode.AUTO:
            return await self._process_auto(stage_node, state, stage_result)
        elif mode == GateMode.THRESHOLD:
            return await self._process_threshold(stage_node, state, stage_result)
        else:
            return await self._process_manual(stage_node, state, stage_result)

    async def _process_manual(
        self,
        stage_node: StageNode,
        state: WorkflowState,
        stage_result: StageResult,
    ) -> GateResult:
        """Manual approval mode.

        Pauses the workflow via ClarificationQueue to await human approval.
        """
        if self._clarification_queue is not None:
            try:
                self._clarification_queue.enqueue(
                    issue_id=f"workflow-{state.workflow_name}",
                    issue_identifier=f"stage-{stage_node.id}",
                    question=f"Approve stage {stage_node.id}: {stage_node.name}?",
                    options=["approve", "reject"],
                    context_summary=f"Workflow: {state.workflow_name}, Stage: {stage_node.name}",
                )
            except Exception as exc:
                logger.warning("Failed to enqueue clarification: %s", exc)

        return GateResult(
            approved=False,
            mode=GateMode.MANUAL,
            reason=f"GATE stage {stage_node.id} awaiting manual approval",
            stage_result=StageResult(
                stage_id=stage_node.id,
                status=StageStatus.GATE_PENDING,
            ),
        )

    async def _process_auto(
        self,
        stage_node: StageNode,
        state: WorkflowState,
        stage_result: StageResult,
    ) -> GateResult:
        """Auto approval mode.

        Auto-judges from ValidatorSpec: approve if all validators pass.
        """
        if not stage_node.validators:
            logger.info("Auto-gate stage %s: no validators, auto-approved", stage_node.id)
            return GateResult(
                approved=True,
                mode=GateMode.AUTO,
                reason="No validators configured, auto-approved",
            )

        validator = self._validator
        results = await validator.validate_all(stage_node.validators)
        all_passed = all(r.passed for r in results)
        failures = [r.message for r in results if not r.passed]

        return GateResult(
            approved=all_passed,
            mode=GateMode.AUTO,
            reason="All validators passed" if all_passed else f"Failed: {failures}",
            stage_result=StageResult(
                stage_id=stage_node.id,
                status=StageStatus.GATE_APPROVED if all_passed else StageStatus.GATE_REJECTED,
                error=None if all_passed else "; ".join(failures),
            ),
        )

    async def _process_threshold(
        self,
        stage_node: StageNode,
        state: WorkflowState,
        stage_result: StageResult,
    ) -> GateResult:
        """Threshold approval mode.

        LLM-as-judge scoring; auto-approve when the threshold is met.
        Score extraction: pull the score from stage_result output.
        """
        threshold = stage_node.gate_threshold
        if threshold is None:
            logger.warning(
                "Threshold GATE stage %s: no gate_threshold configured, rejecting",
                stage_node.id,
            )
            return GateResult(
                approved=False,
                mode=GateMode.THRESHOLD,
                score=0.0,
                reason="No gate_threshold configured",
                stage_result=StageResult(
                    stage_id=stage_node.id,
                    status=StageStatus.GATE_REJECTED,
                ),
            )
        score = self._extract_score_from_result(stage_result)

        approved = score >= threshold
        return GateResult(
            approved=approved,
            mode=GateMode.THRESHOLD,
            score=score,
            reason=f"Score {score:.2f} {'>=' if approved else '<'} threshold {threshold}",
            stage_result=StageResult(
                stage_id=stage_node.id,
                status=StageStatus.GATE_APPROVED if approved else StageStatus.GATE_REJECTED,
            ),
        )

    def _extract_score_from_result(self, result: StageResult) -> float:
        """Extract a score from a stage result (outputs first, error as fallback)."""
        import re

        # Prefer the actual stage outputs (LLM judge emits the score here)
        for output in result.outputs or []:
            match = re.search(r"(?:score)[:\s]*([0-9]*\.?[0-9]+)", str(output), re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        # Fallback: try extracting score from the error message
        if result.error:
            match = re.search(r"(?:score)[:\s]*([0-9]*\.?[0-9]+)", result.error, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return 0.0

    async def approve_manual(self, stage_id: int) -> StageResult:
        """Manually approve a GATE stage."""
        return StageResult(
            stage_id=stage_id,
            status=StageStatus.GATE_APPROVED,
        )

    async def reject_manual(self, stage_id: int, reason: str = "") -> StageResult:
        """Manually reject a GATE stage."""
        return StageResult(
            stage_id=stage_id,
            status=StageStatus.GATE_REJECTED,
            error=reason or "Manually rejected",
        )
