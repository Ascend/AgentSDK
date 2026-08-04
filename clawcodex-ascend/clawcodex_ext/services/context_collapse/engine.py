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

"""High-level orchestrator: trigger + summary + store = engine.

The :class:`CollapseEngine` ties the lower-level pieces together so the
compression pipeline (or a CLI command) can do useful work with a
single call. It owns:

* a :class:`Trigger` (default: :func:`default_composite_trigger`),
* a :class:`SummaryGenerator` (default: :class:`HeadlineSummaryGenerator`),
* a :class:`ContextCollapseStore` (default: an in-memory store),
* a :class:`BoundaryDetector` (default: detects both new and legacy
  boundary formats).

The public methods are:

* :meth:`CollapseEngine.evaluate` — ask the trigger what to do
  without mutating the store. Returns a :class:`CollapseDecision`.
* :meth:`CollapseEngine.apply` — given a decision, archive the
  indicated messages, generate a summary, and record a commit.
  Returns a :class:`CollapseRecoveryResult` so the caller can see
  what was archived and what summary was injected.
* :meth:`CollapseEngine.decide_and_apply` — convenience that runs
  both steps in one call.
* :meth:`CollapseEngine.recover_from_413` — emergency path: if the
  last API call raised a 413-class error, fold aggressively and
  return the recovery summary.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .boundary import BoundaryDetector, make_boundary_text
from .exceptions import ContextLengthExceededError
from .summary import HeadlineSummaryGenerator, SummaryGenerator
from .trigger import (
    CollapseDecision,
    CollapseKind,
    Trigger,
    TriggerContext,
    default_composite_trigger,
)

if TYPE_CHECKING:
    from ...services.compact.context_collapse import ContextCollapseStore


def _load_store_cls() -> type[ContextCollapseStore]:
    from ...services.compact.context_collapse import ContextCollapseStore

    return ContextCollapseStore


def _archive_id_for(messages: Iterable[Any]) -> list[str]:
    """Return the archived message ids (UUIDs) for ``messages``."""
    out: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            uuid = m.get("uuid")
        else:
            uuid = getattr(m, "uuid", None)
        if isinstance(uuid, str) and uuid:
            out.append(uuid)
    return out


@dataclass
class CollapseEngineConfig:
    """Configuration for a :class:`CollapseEngine`."""

    context_window: int = 200_000
    threshold_fraction: float = 0.80
    keep_recent: int = 4
    partial_archive_count: int = 32
    # The engine does not call LLM APIs by default; the caller can
    # pass a custom generator when wiring it into a live application.
    use_legacy_boundary: bool = True


@dataclass
class CollapseRecoveryResult:
    """The outcome of an engine apply / recovery operation."""

    applied: bool
    kind: CollapseKind
    archived_count: int
    summary: str
    boundary_text: str | None = None
    decision: CollapseDecision | None = None
    notes: list[str] = field(default_factory=list)


class CollapseEngine:
    """High-level orchestrator: trigger + summary + store."""

    def __init__(
        self,
        store: ContextCollapseStore | None = None,
        *,
        trigger: Trigger | None = None,
        summary_generator: SummaryGenerator | None = None,
        detector: BoundaryDetector | None = None,
        config: CollapseEngineConfig | None = None,
    ) -> None:
        cls = _load_store_cls()
        if store is None:
            self._store: ContextCollapseStore = cls()
        else:
            if not isinstance(store, cls):
                raise TypeError("store must be a ContextCollapseStore")
            self._store = store
        self._config = config or CollapseEngineConfig()
        self._trigger: Trigger = trigger or default_composite_trigger(
            context_window=self._config.context_window,
            threshold_fraction=self._config.threshold_fraction,
            keep_recent=self._config.keep_recent,
        )
        self._summary: SummaryGenerator = summary_generator or HeadlineSummaryGenerator()
        self._detector = detector or BoundaryDetector(treat_legacy_as_boundary=self._config.use_legacy_boundary)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> ContextCollapseStore:
        return self._store

    @property
    def trigger(self) -> Trigger:
        return self._trigger

    @property
    def summary_generator(self) -> SummaryGenerator:
        return self._summary

    @property
    def detector(self) -> BoundaryDetector:
        return self._detector

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        messages: list[Any],
        *,
        last_error: BaseException | None = None,
        hints: dict[str, Any] | None = None,
    ) -> CollapseDecision:
        ctx = TriggerContext(
            context_window=self._config.context_window,
            threshold_fraction=self._config.threshold_fraction,
            last_error=last_error,
            hints=dict(hints or {}),
        )
        with self._lock:
            return self._trigger.decide(messages, ctx)

    # ------------------------------------------------------------------
    # Apply a decision
    # ------------------------------------------------------------------

    def apply(
        self,
        messages: list[Any],
        decision: CollapseDecision,
    ) -> CollapseRecoveryResult:
        with self._lock:
            if decision.kind is CollapseKind.NOOP:
                return CollapseRecoveryResult(
                    applied=False,
                    kind=CollapseKind.NOOP,
                    archived_count=0,
                    summary="",
                    decision=decision,
                )
            archived, kept = self._split(messages, decision)
            if not archived:
                return CollapseRecoveryResult(
                    applied=False,
                    kind=decision.kind,
                    archived_count=0,
                    summary="",
                    decision=decision,
                    notes=["nothing to archive"],
                )
            summary = self._summary.summarize(archived)
            archive_id = self._detector.mint_archive_id()
            boundary_text = make_boundary_text(archive_id)
            archived_uuids = _archive_id_for(archived)
            # Record the commit on the in-memory store. Use the
            # existing add_commit API for compatibility.
            self._store.add_commit(archived_uuids, summary)
            return CollapseRecoveryResult(
                applied=True,
                kind=decision.kind,
                archived_count=len(archived),
                summary=summary,
                boundary_text=boundary_text,
                decision=decision,
                notes=[f"kept {len(kept)} recent message(s)"],
            )

    # ------------------------------------------------------------------
    # Convenience: decide + apply
    # ------------------------------------------------------------------

    def decide_and_apply(
        self,
        messages: list[Any],
        *,
        last_error: BaseException | None = None,
        hints: dict[str, Any] | None = None,
    ) -> CollapseRecoveryResult:
        decision = self.evaluate(messages, last_error=last_error, hints=hints)
        return self.apply(messages, decision)

    # ------------------------------------------------------------------
    # 413 emergency path
    # ------------------------------------------------------------------

    def recover_from_413(
        self,
        messages: list[Any],
        error: BaseException,
        *,
        max_attempts: int = 1,
    ) -> CollapseRecoveryResult:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        last_exc: BaseException | None = None
        for _ in range(max_attempts):
            decision = self.evaluate(messages, last_error=error)
            result = self.apply(messages, decision)
            if not result.applied:
                # No further folding possible; surface the error.
                last_exc = error
                break
            # Re-evaluate; if the new projected view is still over
            # budget and the error is 413-class, raise. Otherwise the
            # caller can retry.
            notes = list(result.notes)
            notes.append("recovered via emergency collapse")
            result.notes = notes
            return result
        if last_exc is not None:
            raise ContextLengthExceededError(
                f"context still over budget after emergency collapse: {last_exc!r}"
            ) from last_exc
        return CollapseRecoveryResult(
            applied=False,
            kind=CollapseKind.NOOP,
            archived_count=0,
            summary="",
            notes=["no recovery applied"],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split(
        self,
        messages: list[Any],
        decision: CollapseDecision,
    ) -> tuple[list[Any], list[Any]]:
        if not messages:
            return [], []
        if decision.kind is CollapseKind.PARTIAL:
            count = decision.count
            if count is None or count <= 0:
                count = 32
            return list(messages[:count]), list(messages[count:])
        # FULL
        keep = decision.count
        if keep is None or keep < 0:
            keep = self._config.keep_recent
        if keep >= len(messages):
            return [], list(messages)
        return list(messages[:-keep]) if keep else list(messages), list(messages[-keep:])


__all__ = [
    "CollapseEngine",
    "CollapseEngineConfig",
    "CollapseRecoveryResult",
]
