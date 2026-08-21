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

"""GATE rollback handler.

Handles rollback after a GATE rejection:
- Determine the rollback target stage
- Restore the workspace snapshot
- Update workflow state
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .errors import RollbackError
from .rollback import RollbackManager, RollbackTarget
from .workflow_state import StageNode, StageStatus, WorkflowState

logger = logging.getLogger(__name__)


# -- GATE rollback result --───────────────────────────────────────────────────


@dataclass
class GateRollbackResult:
    """GATE rollback result."""

    success: bool
    target_stage_id: int
    target_stage_name: str
    reason: str
    snapshot_restored: bool = False


# -- GATE rollback handler --─────────────────────────────────────────────────


class GateRollbackHandler:
    """GATE rollback handler.

    Performs rollback when a GATE stage is rejected.
    Reuses the Human Review Gate workspace-preservation strategy.
    """

    def __init__(
        self,
        rollback_manager: RollbackManager,
    ) -> None:
        self._rollback = rollback_manager

    def resolve_gate_rollback(
        self,
        stage: StageNode,
        state: WorkflowState,
        rejection_reason: str = "",
    ) -> RollbackTarget:
        """Resolve the Rollback target after a GATE rejection.

        Priority:
        1. Explicitly specified by stage.gate_rollback_to
        2. First dependency in stage.depends_on
        3. The workflow's starting stage

        Args:
            stage: GATE stage node
            state: workflow state
            rejection_reason: rejection reason

        Returns:
            RollbackTarget: rollback target
        """
        # Explicit target
        if stage.gate_rollback_to is not None:
            target_id = int(stage.gate_rollback_to)
            target_snapshot = self._rollback.get_snapshot(target_id)
            return RollbackTarget(
                stage_id=target_id,
                stage_name=f"stage-{target_id}",
                reason=f"GATE rejected: {rejection_reason} (explicit rollback_to={target_id})",
                snapshot=target_snapshot,
            )

        # Dependency stage
        if stage.depends_on:
            target_id = stage.depends_on[0]
            target_snapshot = self._rollback.get_snapshot(target_id)
            return RollbackTarget(
                stage_id=target_id,
                stage_name=f"stage-{target_id}",
                reason=f"GATE rejected: {rejection_reason} (rollback to dependency)",
                snapshot=target_snapshot,
            )

        raise RollbackError(
            f"No rollback target for GATE stage {stage.id}",
            stage_id=stage.id,
        )

    def execute_rollback(
        self,
        target: RollbackTarget,
        state: WorkflowState,
        failed_stage_id: int,
    ) -> GateRollbackResult:
        """Execute a GATE rollback.

        1. Restore the target stage snapshot
        2. Update WorkflowState
        3. Return the rollback result

        Args:
            target: rollback target
            state: workflow state
            failed_stage_id: ID of the failed GATE stage

        Returns:
            GateRollbackResult: rollback result
        """
        snapshot_restored = False

        try:
            if target.snapshot is not None:
                snapshot_restored = self._rollback.restore_snapshot(target.stage_id)
        except Exception as exc:
            logger.warning("Snapshot restore failed, continuing with state rollback only: %s", exc)

        # Update state
        self._rollback.update_state_on_rollback(state, target.stage_id, failed_stage_id)

        # Mark the GATE stage as rejected
        if failed_stage_id in state.stage_statuses:
            state.stage_statuses[failed_stage_id] = StageStatus.GATE_REJECTED

        return GateRollbackResult(
            success=True,
            target_stage_id=target.stage_id,
            target_stage_name=target.stage_name,
            reason=target.reason,
            snapshot_restored=snapshot_restored,
        )

    def determine_dag_index(
        self,
        target: RollbackTarget,
        dag_order: list[int],
    ) -> int:
        """Determine the rollback target's index in the DAG.

        Args:
            target: rollback target
            dag_order: DAG ordering list

        Returns:
            The target stage's index in the DAG
        """
        try:
            return dag_order.index(target.stage_id)
        except ValueError:
            # If the target is not in the DAG, start from the first stage
            logger.warning(
                "Rollback target %s not in DAG order, starting from beginning",
                target.stage_id,
            )
            return 0

    async def handle_gate_rejection(
        self,
        stage: StageNode,
        state: WorkflowState,
        dag_order: list[int],
        rejection_reason: str = "",
    ) -> tuple[int, GateRollbackResult]:
        """Full flow for handling a GATE rejection.

        Resolve target -> execute rollback -> return the new DAG index.

        Args:
            stage: GATE stage node
            state: workflow state
            dag_order: DAG ordering list
            rejection_reason: rejection reason

        Returns:
            (new DAG index, rollback result)
        """
        target = self.resolve_gate_rollback(stage, state, rejection_reason)
        result = self.execute_rollback(target, state, stage.id)
        new_index = self.determine_dag_index(target, dag_order)

        logger.info(
            "GATE rollback: stage %s rejected, rolling back to stage %s (index %s)",
            stage.id,
            target.stage_id,
            new_index,
        )

        return new_index, result
