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

"""Ultraplan service primitives and user-facing planning helpers.

This package ships a hierarchical plan model (Plan → SubPlan → Step)
with strict dataclass validation, an atomic JSON store, a step state
machine executor, an adjuster for mid-execution changes, and a
sandboxed acceptance-criteria verifier. adds LLM plan generation,
templates, keyword detection, CCR client plumbing, audit logging, and
the controller used by the ``/ultraplan`` command.
"""

from __future__ import annotations

from .adjuster import PlanAdjuster
from .exceptions import (
    CCRTimeoutError,
    CCRUnavailableError,
    DuplicateStepIdError,
    DuplicateSubPlanIdError,
    IllegalStepTransitionError,
    PlanCorruptError,
    PlannerFailedError,
    PlanNotFoundError,
    ProviderUnavailableError,
    StepHasDependentsError,
    StepNotFoundError,
    SubPlanNotFoundError,
    TemplateNotFoundError,
    UltraplanError,
    UnknownCheckKindError,
    UnsafeCheckExpressionError,
    VerificationCheckFailedError,
)
from .executor import PlanExecutor, Progress, StepTransition
from .feature_gates import (
    ULTRAPLAN_LLM_PLANNER,
    ULTRAPLAN_RAINBOW,
    ULTRAPLAN_REMOTE,
    is_ccr_endpoint_allowed,
    is_ultraplan_llm_enabled,
    is_ultraplan_rainbow_enabled,
    is_ultraplan_remote_enabled,
)
from .llm_planner import LLMPlanner, PlannerContext, PlannerResult
from .models import (
    AcceptanceCriteria,
    CheckKind,
    Plan,
    PlanStatus,
    Step,
    StepKind,
    StepStatus,
    SubPlan,
)
from .store import PlanStore
from .templates import BUILTIN_TEMPLATES, PlanTemplate, TemplateLibrary
from .verifier import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    AcceptanceVerifier,
    CheckResult,
)

__all__ = [
    "BUILTIN_TEMPLATES",
    "DEFAULT_SHELL_TIMEOUT_SECONDS",
    "ULTRAPLAN_LLM_PLANNER",
    "ULTRAPLAN_RAINBOW",
    "ULTRAPLAN_REMOTE",
    "AcceptanceCriteria",
    "AcceptanceVerifier",
    "CCRTimeoutError",
    "CCRUnavailableError",
    "CheckKind",
    "CheckResult",
    "DuplicateStepIdError",
    "DuplicateSubPlanIdError",
    "IllegalStepTransitionError",
    "LLMPlanner",
    "Plan",
    "PlanAdjuster",
    "PlanCorruptError",
    "PlanExecutor",
    "PlanNotFoundError",
    "PlanStatus",
    "PlanStore",
    "PlanTemplate",
    "PlannerContext",
    "PlannerFailedError",
    "PlannerResult",
    "Progress",
    "ProviderUnavailableError",
    "Step",
    "StepHasDependentsError",
    "StepKind",
    "StepNotFoundError",
    "StepStatus",
    "StepTransition",
    "SubPlan",
    "SubPlanNotFoundError",
    "TemplateLibrary",
    "TemplateNotFoundError",
    "UltraplanError",
    "UnknownCheckKindError",
    "UnsafeCheckExpressionError",
    "VerificationCheckFailedError",
    "is_ccr_endpoint_allowed",
    "is_ultraplan_llm_enabled",
    "is_ultraplan_rainbow_enabled",
    "is_ultraplan_remote_enabled",
]
