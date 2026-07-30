"""Ultraplan service primitives and F-87 user-facing planning helpers.

This package ships a hierarchical plan model (Plan → SubPlan → Step)
with strict dataclass validation, an atomic JSON store, a step state
machine executor, an adjuster for mid-execution changes, and a
sandboxed acceptance-criteria verifier. F-87 adds LLM plan generation,
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
