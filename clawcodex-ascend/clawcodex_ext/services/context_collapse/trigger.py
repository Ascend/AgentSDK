"""Collapse trigger: when should we fold?

The trigger decides whether the current message set is over budget and,
if so, what kind of collapse to perform. Three triggers are bundled:

* :class:`TokenThresholdTrigger` — folds when the input token count
  exceeds a fraction of the configured context window. This is the
  primary P84-A use case.
* :class:`Emergency413Trigger` — folds when an upstream API returns
  HTTP 413 (or any error matching a registered predicate). This is
  the P84-E use case.
* :class:`CompositeTrigger` — combines several triggers with an
  "any" / "all" policy. The default policy is "any".

Triggers are pure functions: they take a snapshot of the current
state and return a :class:`CollapseDecision`. They do not mutate the
collapse store; that is the responsibility of the caller (typically
the compression pipeline).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .exceptions import ContextLengthExceededError
from .tokens import TokenCounter, TokenEstimate, heuristic_only


class CollapseKind(str, Enum):
    """The kind of collapse the trigger is asking for."""

    NOOP = "noop"
    PARTIAL = "partial"  # Collapse only the oldest N messages.
    FULL = "full"  # Collapse everything except the most recent K.


@dataclass(frozen=True)
class CollapseDecision:
    """The trigger's verdict on whether and how to collapse."""

    kind: CollapseKind
    reason: str = ""
    token_estimate: TokenEstimate | None = None
    # For ``PARTIAL`` decisions, how many messages to archive. For
    # ``FULL`` decisions, how many to keep.
    count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Trigger(Protocol):
    """Decide whether to collapse given the current state."""

    def decide(self, messages: list[Any], context: TriggerContext) -> CollapseDecision: ...


@dataclass
class TriggerContext:
    """Snapshot of state passed to a trigger's ``decide`` method."""

    context_window: int
    threshold_fraction: float = 0.80
    # Last measured error, if any. Used by the 413 trigger.
    last_error: BaseException | None = None
    # Optional: extra caller-supplied hints (model, request id, etc.).
    hints: dict[str, Any] = field(default_factory=dict)


# A predicate that, given a captured exception, returns True if it
# should be treated as a 413-style "context too long" signal.
ErrorPredicate = Callable[[BaseException], bool]


def default_error_predicate(exc: BaseException) -> bool:
    """Treat the canonical exceptions as 413 signals.

    Returns True for the project's ``ContextLengthExceededError`` and
    for any HTTP error with status 413 (or a "context length exceeded"
    string in the message). OpenAI, Anthropic, and the project's
    custom exceptions all surface the same canonical message.
    """
    if isinstance(exc, ContextLengthExceededError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "413" in msg or "request_too_large" in name or "requestentitytoolarge" in name:
        return True
    if "context_length" in msg or "context length" in msg or "too long" in msg:
        return True
    # Last resort: the exception exposes a ``status_code`` attribute.
    status = getattr(exc, "status_code", None)
    return status == 413


@dataclass
class TokenThresholdTrigger:
    """Fold when tokens exceed a fraction of the context window."""

    counter: TokenCounter
    threshold_fraction: float = 0.80
    keep_recent: int = 4
    partial_archive_count: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold_fraction <= 1.0):
            raise ValueError("threshold_fraction must be in (0, 1]")
        if self.keep_recent < 0:
            raise ValueError("keep_recent must be non-negative")

    def decide(self, messages: list[Any], context: TriggerContext) -> CollapseDecision:
        if not messages:
            return CollapseDecision(kind=CollapseKind.NOOP, reason="no messages")
        estimate = TokenEstimate(
            tokens=self.counter.count_messages(messages),
            counter_name=getattr(self.counter, "name", "unknown"),
        )
        budget = int(context.context_window * context.threshold_fraction)
        if estimate.tokens <= budget:
            return CollapseDecision(
                kind=CollapseKind.NOOP,
                reason=f"under budget ({estimate.tokens} <= {budget})",
                token_estimate=estimate,
            )
        # Decide between PARTIAL and FULL.
        if self.partial_archive_count is not None:
            return CollapseDecision(
                kind=CollapseKind.PARTIAL,
                reason=f"over budget ({estimate.tokens} > {budget})",
                token_estimate=estimate,
                count=self.partial_archive_count,
            )
        return CollapseDecision(
            kind=CollapseKind.FULL,
            reason=f"over budget ({estimate.tokens} > {budget})",
            token_estimate=estimate,
            count=self.keep_recent,
        )


@dataclass
class Emergency413Trigger:
    """Fold when the upstream API returns 413 / context-too-long."""

    keep_recent: int = 2
    archive_count: int = 64
    predicate: ErrorPredicate = default_error_predicate

    def __post_init__(self) -> None:
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")
        if self.keep_recent < 0:
            raise ValueError("keep_recent must be non-negative")
        if self.archive_count < 0:
            raise ValueError("archive_count must be non-negative")

    def decide(self, messages: list[Any], context: TriggerContext) -> CollapseDecision:
        if context.last_error is None:
            return CollapseDecision(kind=CollapseKind.NOOP, reason="no error")
        if not self.predicate(context.last_error):
            return CollapseDecision(kind=CollapseKind.NOOP, reason="error is not 413-class")
        # Always go full when we're in emergency mode; keep_recent is
        # just a safety floor.
        return CollapseDecision(
            kind=CollapseKind.FULL,
            reason=f"413-class error: {context.last_error!r}",
            count=self.keep_recent,
            extra={"archive_count": self.archive_count},
        )


class CompositeTrigger:
    """Combine several triggers with "any" or "all" semantics.

    The default is "any": if any sub-trigger asks for collapse, the
    composite reports the most aggressive decision. With "all", the
    composite collapses only when every sub-trigger asks for it.
    """

    def __init__(
        self,
        triggers: list[Trigger],
        *,
        policy: str = "any",
    ) -> None:
        if not triggers:
            raise ValueError("CompositeTrigger requires at least one sub-trigger")
        if policy not in {"any", "all"}:
            raise ValueError("policy must be 'any' or 'all'")
        self._triggers = list(triggers)
        self._policy = policy
        self._lock = threading.RLock()

    def decide(self, messages: list[Any], context: TriggerContext) -> CollapseDecision:
        with self._lock:
            decisions = [t.decide(messages, context) for t in self._triggers]
        non_noop = [d for d in decisions if d.kind != CollapseKind.NOOP]
        if self._policy == "any":
            if not non_noop:
                return CollapseDecision(
                    kind=CollapseKind.NOOP,
                    reason="all sub-triggers reported NOOP",
                )
            # Return the most aggressive (FULL > PARTIAL).
            for kind in (CollapseKind.FULL, CollapseKind.PARTIAL):
                for d in non_noop:
                    if d.kind is kind:
                        return d
        else:  # "all"
            if len(non_noop) != len(decisions):
                return CollapseDecision(
                    kind=CollapseKind.NOOP,
                    reason="not every sub-trigger asked to collapse",
                )
            for kind in (CollapseKind.FULL, CollapseKind.PARTIAL):
                if all(d.kind is kind for d in non_noop):
                    return CollapseDecision(
                        kind=kind,
                        reason="all sub-triggers agreed",
                    )
        return CollapseDecision(
            kind=CollapseKind.NOOP,
            reason="no sub-trigger asked for collapse",
        )


# Pre-built default for production use: token threshold OR 413, with
# the 413 trigger always going FULL.
def default_composite_trigger(
    *,
    counter: TokenCounter | None = None,
    context_window: int = 200_000,
    threshold_fraction: float = 0.80,
    keep_recent: int = 4,
) -> CompositeTrigger:
    return CompositeTrigger(
        [
            TokenThresholdTrigger(
                counter=counter or heuristic_only(),
                threshold_fraction=threshold_fraction,
                keep_recent=keep_recent,
            ),
            Emergency413Trigger(keep_recent=2),
        ]
    )
