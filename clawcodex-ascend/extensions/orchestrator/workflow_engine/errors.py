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

"""Workflow engine exception type definitions."""

from __future__ import annotations

from typing import Any


class WorkflowEngineError(Exception):
    """Base exception for the workflow engine."""

    def __init__(self, message: str, stage_id: int | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.stage_id = stage_id
        self.details = details or {}


class StageTimeoutError(WorkflowEngineError):
    """Stage execution timed out."""


class StageFailureError(WorkflowEngineError):
    """Stage execution failed."""


class CostExceededError(WorkflowEngineError):
    """Cost budget exceeded."""


class ValidationError(WorkflowEngineError):
    """Stage output validation failed."""


class GateRejectedError(WorkflowEngineError):
    """GATE approval rejected."""


class DecisionExhaustedError(WorkflowEngineError):
    """DECISION loop retries exhausted."""


class ConvergenceError(WorkflowEngineError):
    """DECISION convergence detected -- degenerate loop."""


class CheckpointError(WorkflowEngineError):
    """Checkpoint read/write failed."""


class WorkflowSchemaError(WorkflowEngineError):
    """Invalid workflow.yaml format."""


class ResumeError(WorkflowEngineError):
    """Failed to resume execution from checkpoint."""


class RollbackError(WorkflowEngineError):
    """Stage rollback failed."""
