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

"""Lazy compression pipeline gate.

Decides whether :func:`run_compression_pipeline` should run on the
current turn. When the estimated input token count is well below the
context window, the pipeline is a no-op anyway — skipping it earlier
avoids the 20-500ms cost of allocating, traversing, and emitting
"applied" log lines for layers that did nothing.

Design rationale
----------------
The per-turn compression pipeline is a performance hotspot. The TS upstream skips the
pipeline when the message list is small; this module ports that
behaviour without touching ``clawcodex_ext/query/query.py`` — the
gate is invoked inside :class:`CompressionPipeline` itself.

Force-run conditions (any one of these overrides the threshold):

- ``query_source`` in ``{"compact", "session_memory"}`` — these flows
 exist to *reduce* the message list, so skipping them would deadlock
 the loop. Mirrors the existing ``skip_blocking_guards`` short-circuit
 in ``clawcodex_ext/query/query.py``.
- ``transition_reason`` in ``{"reactive_compact_retry",
 "collapse_drain_retry"}`` — recovery transitions must always see a
 fresh pipeline run.
- ``previous_pipeline_errored`` — when the previous run errored, the
 next turn gets another shot at recovering.

Env-var override
----------------
``CLAWCODEX_COMPRESSION_GATE_SKIP_RATIO`` (float, ``0 < x < 1``)
overrides the configured ratio at runtime. Useful for ops triage
without redeploying config. Mirrors the env-var pattern used in
``clawcodex_ext/services/compact/autocompact.py``.

Decoupling note
---------------
This module imports only stdlib; it depends on nothing inside
``src/``. The pipeline that calls it (``clawcodex_ext/services/compact/pipeline.py``)
is in the same Layer 1 namespace, so no cross-layer plumbing is
needed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


DEFAULT_COMPRESSION_GATE_SKIP_RATIO: float = 0.6
ENV_COMPRESSION_GATE_SKIP_RATIO: str = "CLAWCODEX_COMPRESSION_GATE_SKIP_RATIO"

_FORCED_QUERY_SOURCES: frozenset[str] = frozenset({"compact", "session_memory"})
_FORCED_TRANSITION_REASONS: frozenset[str] = frozenset({"reactive_compact_retry", "collapse_drain_retry"})


def _get_env_float(name: str) -> float | None:
    """Read a float env var. Returns ``None`` when unset or unparseable.

    Negative or zero values are rejected to keep the gate from being
    silently disabled by a typo (``CLAWCODEX_COMPRESSION_GATE_SKIP_RATIO=0``
    would mean "always skip", which is the opposite of the documented
    semantic of the env var).
    """
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        parsed = float(val)
    except ValueError:
        return None
    if parsed <= 0 or parsed >= 1.0:
        # Out of range: treat as unset. The configured default still applies.
        logger.debug("ignoring %s=%s (must satisfy 0 < x < 1)", name, parsed)
        return None
    return parsed


def resolve_skip_ratio_from_env(configured: float | None) -> float:
    """Pick the effective skip ratio: env override → configured → default.

    Layered precedence keeps ops in control without forcing a config
    change, while still respecting explicit per-instance overrides
    passed via :class:`PipelineConfig.gate_skip_ratio`.
    """
    env_val = _get_env_float(ENV_COMPRESSION_GATE_SKIP_RATIO)
    if env_val is not None:
        return env_val
    if configured is not None:
        try:
            return float(configured)
        except (TypeError, ValueError):
            return DEFAULT_COMPRESSION_GATE_SKIP_RATIO
    return DEFAULT_COMPRESSION_GATE_SKIP_RATIO


def should_run_compression_pipeline(
    *,
    est_input_tokens: int,
    context_window: int,
    skip_ratio: float = DEFAULT_COMPRESSION_GATE_SKIP_RATIO,
    query_source: str = "",
    transition_reason: str | None = None,
    previous_pipeline_errored: bool = False,
) -> tuple[bool, str]:
    """Return ``(should_run, reason)``.

    ``reason`` is one of:

    - ``"forced_source"`` — ``query_source`` is in
      :data:`_FORCED_QUERY_SOURCES` (must run regardless of size).
    - ``"forced_transition"`` — ``transition_reason`` is in
      :data:`_FORCED_TRANSITION_REASONS` (recovery retry must run).
    - ``"forced_prev_error"`` — last pipeline call errored; retry.
    - ``"below_threshold"`` — gate decision was applied; skip.
    - ``"above_threshold"`` — gate decision was applied; run.

    Defensive behaviour:

    - If ``skip_ratio <= 0`` the gate is disabled and the pipeline
      always runs (reason ``"above_threshold"``).
    - If ``context_window <= 0`` we cannot evaluate the threshold
      meaningfully, so the pipeline always runs
      (reason ``"above_threshold"``).
    - Negative ``est_input_tokens`` is treated as zero.
    """
    if previous_pipeline_errored:
        return True, "forced_prev_error"

    if query_source in _FORCED_QUERY_SOURCES:
        return True, "forced_source"

    if transition_reason in _FORCED_TRANSITION_REASONS:
        return True, "forced_transition"

    if skip_ratio <= 0 or context_window <= 0:
        return True, "above_threshold"

    threshold = int(context_window * skip_ratio)
    tokens = max(0, int(est_input_tokens))
    if tokens < threshold:
        return False, "below_threshold"

    return True, "above_threshold"


__all__ = [
    "DEFAULT_COMPRESSION_GATE_SKIP_RATIO",
    "ENV_COMPRESSION_GATE_SKIP_RATIO",
    "resolve_skip_ratio_from_env",
    "should_run_compression_pipeline",
]
