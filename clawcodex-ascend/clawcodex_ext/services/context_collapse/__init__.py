"""Context collapse service layer (F-84) — ClawCodex subsystem.

This package complements the existing ``src/services/compact/context_collapse.py``
module, which ships a :class:`ContextCollapseStore` data model and a
``project_view`` placeholder-injection step. The new service layer
adds the missing primitives from the F-84 spec:

* :mod:`tokens` — pluggable token counter with a tiktoken-default +
  heuristic-fallback chain (P84-A).
* :mod:`summary` — pluggable summary generator with a no-LLM
  default and an LLM-callback variant (P84-B).
* :mod:`trigger` — token-threshold and 413-emergency triggers
  (P84-A + P84-E), with a composite for "any of" decisions.
* :mod:`boundary` — explicit ``ContextCollapseBoundary`` marker
  (P84-C), with a detector for downstream consumers.
* :mod:`persistence` — atomic file-based persistence for the
  collapse state, with corruption handling and a merge helper
  for session-restore (P84-D).
* :mod:`engine` — high-level orchestrator that ties the trigger,
  the summary generator, and the store together, plus a recovery
  path for 413-class errors (P84-E + P84-F).

The existing ``ContextCollapseStore`` and ``project_view`` continue
to be the canonical data model and projection; the new layer is a
service-level wrapper that the compression pipeline and CLI can
drive without knowing the internal details.
"""

from __future__ import annotations

from clawcodex_ext.services.context_collapse import (
    boundary,
    engine,
    exceptions,
    persistence,
    summary,
    tokens,
    trigger,
)
from clawcodex_ext.services.context_collapse.boundary import (
    BOUNDARY_PREFIX,
    BoundaryDetector,
    BoundaryHit,
    make_boundary_text,
)
from clawcodex_ext.services.context_collapse.engine import (
    CollapseEngine,
    CollapseEngineConfig,
    CollapseRecoveryResult,
)
from clawcodex_ext.services.context_collapse.exceptions import (
    CollapseStateCorruptError,
    CollapseStateNotFoundError,
    ContextCollapseError,
    ContextLengthExceededError,
    SummaryGeneratorError,
    TokenCountUnavailableError,
)
from clawcodex_ext.services.context_collapse.persistence import (
    CollapseStateFile,
    load_store,
    merge_stores,
    save_store,
)
from clawcodex_ext.services.context_collapse.summary import (
    HeadlineSummaryGenerator,
    LLMSummaryGenerator,
    SummaryGenerator,
    count_words,
    extract_text,
)
from clawcodex_ext.services.context_collapse.tokens import (
    CharTokenCounter,
    FallbackTokenCounter,
    TiktokenCounter,
    TokenCounter,
    TokenEstimate,
    heuristic_only,
    tiktoken_first_then_heuristic,
)
from clawcodex_ext.services.context_collapse.trigger import (
    CollapseDecision,
    CollapseKind,
    CompositeTrigger,
    Emergency413Trigger,
    TokenThresholdTrigger,
    Trigger,
    TriggerContext,
    default_composite_trigger,
    default_error_predicate,
)

__all__ = [
    "boundary",
    "engine",
    "exceptions",
    "persistence",
    "summary",
    "tokens",
    "trigger",
    "BOUNDARY_PREFIX",
    "BoundaryDetector",
    "BoundaryHit",
    "CharTokenCounter",
    "CollapseDecision",
    "CollapseEngine",
    "CollapseEngineConfig",
    "CollapseKind",
    "CollapseRecoveryResult",
    "CollapseStateCorruptError",
    "CollapseStateFile",
    "CollapseStateNotFoundError",
    "CompositeTrigger",
    "ContextCollapseError",
    "ContextLengthExceededError",
    "Emergency413Trigger",
    "FallbackTokenCounter",
    "HeadlineSummaryGenerator",
    "LLMSummaryGenerator",
    "SummaryGenerator",
    "SummaryGeneratorError",
    "TiktokenCounter",
    "TokenCountUnavailableError",
    "TokenCounter",
    "TokenEstimate",
    "TokenThresholdTrigger",
    "Trigger",
    "TriggerContext",
    "count_words",
    "default_composite_trigger",
    "default_error_predicate",
    "extract_text",
    "heuristic_only",
    "load_store",
    "make_boundary_text",
    "merge_stores",
    "save_store",
    "tiktoken_first_then_heuristic",
]
