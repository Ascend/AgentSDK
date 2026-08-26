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

"""Issue → collaboration mode router.

Phase 1 shipped only the static decision path (label → mode, else
default). Phase 2 plugs in an optional :class:`Router` backend that
``ModeSelector`` consults whenever an issue carries no explicit
``mode:*`` label, or carries ``mode:auto``.

Decision flow (top-down, first match wins):

1. ``mode:<name>`` label among ``issue.labels`` that names a known
   non-``auto`` mode → use it. Source: ``"label"``.
2. ``mode:auto`` label → consult the router (if any). Source: ``"router"``
   on success; ``"fallback"`` if the router is missing / returns
   unknown / raises.
3. No mode label and a router is configured → consult the router.
4. Otherwise → ``DEFAULT_MODE``. Source: ``"fallback"``.

The router itself is duck-typed against :class:`mode_router.Router` so
tests and ops can inject fakes / heuristics / LLM-backed routers
without touching ``ModeSelector``.

The selector never raises out of ``choose`` — every error path returns
a valid ``ModeDecision`` (possibly with ``source="fallback"`` and a
``reason`` explaining what went wrong). Callers can treat the result as
always-valid.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .modes.base import DEFAULT_MODE, ModeDecision

if TYPE_CHECKING:
    from .mode_router import Router
    from .tracker import Issue

logger = logging.getLogger(__name__)

MODE_LABEL_PREFIX: str = "mode:"
"""Issue labels starting with this prefix request a specific mode."""

KNOWN_MODES: frozenset[str] = frozenset({"single", "pipeline", "coordinator", "debate", "swarm", "auto"})
"""All valid mode names the selector / CLI / dashboard accept.

``auto`` is a meta-mode meaning "let the router decide" and is treated
identically to "no label". The other four map to ``ModeRunner`` keys.
"""

# A router pick below this confidence falls back to the default mode
# instead of being honored. Conservative default — keeps Phase 2
# routers (which are heuristic-based and not great at edge cases) from
# silently misrouting issues. Override via ModeSelector(min_confidence=...).
_DEFAULT_MIN_CONFIDENCE: float = 0.5


class ModeSelector:
    """Decide which collaboration mode an issue should run under."""

    def __init__(
        self,
        *,
        default_mode: str = DEFAULT_MODE,
        router: "Router | None" = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if default_mode not in KNOWN_MODES:
            raise ValueError(f"default_mode={default_mode!r} not in KNOWN_MODES={sorted(KNOWN_MODES)}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence={min_confidence!r} must be in [0.0, 1.0]")
        self._default_mode = default_mode
        self._router = router
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def choose(self, issue: "Issue") -> ModeDecision:
        """Return the chosen mode for ``issue``.

        Never raises — failures fall back to ``DEFAULT_MODE`` and log a
        warning. Callers can treat the result as always-valid.
        """
        labelled = self._mode_from_labels(getattr(issue, "labels", None))
        if labelled is not None:
            mode, label = labelled
            if mode == "auto":
                return self._choose_via_router(
                    issue,
                    fallback_reason=f"label {label!r} requests router",
                )
            return ModeDecision(
                mode=mode,
                reason=f"explicit label {label!r}",
                source="label",
            )

        if self._router is not None:
            return self._choose_via_router(issue, fallback_reason="no mode label; consulting router")

        return self._default_decision("no mode label and no router configured")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mode_from_labels(self, labels: list[str] | tuple[str, ...] | None) -> tuple[str, str] | None:
        """Find the first ``mode:<name>`` label among ``labels``.

        Returns ``(mode_name, raw_label)`` or ``None``. Labels are
        normalized to lowercase before matching so ``mode:Pipeline``
        still wins.
        """
        if not labels:
            return None
        for raw in labels:
            if not isinstance(raw, str):
                continue
            label = raw.strip().lower()
            if not label.startswith(MODE_LABEL_PREFIX):
                continue
            mode = label[len(MODE_LABEL_PREFIX) :]
            if mode in KNOWN_MODES:
                return mode, raw
            logger.warning(
                "ModeSelector: ignoring unknown mode label %r (known=%s)",
                raw,
                sorted(KNOWN_MODES),
            )
        return None

    def _choose_via_router(self, issue: "Issue", *, fallback_reason: str) -> ModeDecision:
        """Ask the router which mode to use, with full error containment.

        Failure modes (all → fallback decision with a useful ``reason``):
        - No router configured.
        - Router raises.
        - Router returns an unknown / ``auto`` mode.
        - Router's confidence below ``min_confidence``.
        """
        if self._router is None:
            return self._default_decision(f"{fallback_reason}; but no router available")

        try:
            result = self._router.choose(issue)
        except Exception:
            logger.exception("ModeSelector: router.choose raised")
            return self._default_decision(f"{fallback_reason}; router raised — see logs")

        mode = result.mode
        if mode not in KNOWN_MODES or mode == "auto":
            logger.warning(
                "ModeSelector: router returned mode=%r (not a runnable mode); falling back to %s",
                mode,
                self._default_mode,
            )
            return self._default_decision(f"router returned mode={mode!r}; falling back")

        if result.confidence < self._min_confidence:
            return self._default_decision(
                f"router picked {mode} but confidence "
                f"{result.confidence:.2f} < {self._min_confidence:.2f}; "
                f"falling back ({result.reason})"
            )

        return ModeDecision(
            mode=mode,
            reason=result.reason,
            source="router",
            confidence=result.confidence,
        )

    def _default_decision(self, reason: str) -> ModeDecision:
        return ModeDecision(
            mode=self._default_mode,
            reason=reason,
            source="fallback",
        )


__all__ = ["KNOWN_MODES", "MODE_LABEL_PREFIX", "ModeSelector"]
