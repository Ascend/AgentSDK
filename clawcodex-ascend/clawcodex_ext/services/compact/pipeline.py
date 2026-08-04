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
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""
Compression pipeline — orchestrates the 5 compression layers in order.

Runs cheap → expensive.  If earlier layers free enough tokens, later
layers are no-ops.

Layers:
  1. apply_tool_result_budget  — Persist large results to disk
  2. snip_compact              — Trim old tool results
  3. microcompact              — Compress intermediate tool calls
  4. context_collapse          — Read-time projection
  5. autocompact               — Full LLM summarization (last resort)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clawcodex_ext.context_system.microcompact import (  # pylint: disable=no-name-in-module
    TimeBasedMCConfig,
    microcompact_typed_messages,
)
from clawcodex_ext.providers.base import BaseProvider  # pylint: disable=no-name-in-module
from clawcodex_ext.types.messages import Message  # pylint: disable=no-name-in-module

from .autocompact import (
    AutoCompactTracking,
    auto_compact_if_needed,
)
from .context_collapse import ContextCollapseStore, get_context_collapse_state
from .gating import (
    resolve_skip_ratio_from_env,
    should_run_compression_pipeline,
)
from .snip_compact import snip_compact
from .tool_result_budget import apply_tool_result_budget

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of running the compression pipeline."""

    messages: list[Message]
    tokens_saved: int = 0
    layers_applied: list[str] = field(default_factory=list)
    autocompact_result: Any | None = None  # CompactionResult if layer 5 ran


@dataclass
class PipelineConfig:
    """Configuration for the compression pipeline."""

    # Layer 1: tool result budget
    budget_dir: Path | str | None = None
    max_result_tokens: int = 8_000

    # Layer 2: snip compact
    snip_keep_recent: int = 10

    # Layer 3: microcompact
    # TS time-based MC is disabled by default (GrowthBook enabled: false),
    # cached MC uses API cache_edits (no local mutation), and legacy MC was
    # removed.  So microcompact is effectively a no-op on the main thread.
    mc_enabled: bool = False
    mc_keep_recent: int = 5
    mc_time_config: TimeBasedMCConfig | None = None

    # Layer 4: context collapse
    collapse_store: ContextCollapseStore | None = None

    # Layer 5: autocompact
    context_window: int = 200_000
    max_output_tokens: int | None = None
    autocompact_threshold: float = 0.80
    autocompact_tracking: AutoCompactTracking | None = None

    # F-106: lazy compression gate. When ``est_input_tokens`` is below
    # ``context_window * gate_skip_ratio``, the pipeline short-circuits
    # and returns an empty result instead of running all 5 layers. Set
    # to 0 to disable the gate (always run). Overridable at runtime via
    # the ``CLAWCODEX_COMPRESSION_GATE_SKIP_RATIO`` env var.
    gate_skip_ratio: float = 0.6
    # Force-run overrides for the gate. Forwarded by the caller when
    # the current turn is a compact/session_memory flow, a recovery
    # retry, or follows a pipeline error. The defaults are permissive
    # — they must be set explicitly for the gate to actually skip.
    gate_query_source: str = ""
    gate_transition_reason: str | None = None
    gate_previous_pipeline_errored: bool = False

    # Layer 5: post-compact attachment context
    # Forwarded into auto_compact_if_needed → CompactContext so post-compact
    # file/plan restoration fires on auto-compact, not just /compact.
    read_file_state: dict[str, Any] | None = None
    plan_file_path: str | None = None
    memory_paths: set[str] | None = None

    # Global
    provider: BaseProvider | None = None
    model: str = ""
    custom_instructions: str | None = None

    # Token budget: if pipeline frees this many tokens, skip remaining layers
    early_exit_tokens: int = 20_000

    # Goal-aware compression (F-9 / F-38).
    # When ``goal_active`` is True, the autocompact threshold is raised
    # (compact less aggressively) and messages containing
    # ``<goal-steering`` are preserved from compaction so the model
    # never loses sight of the current objective.
    goal_active: bool = False


def build_production_pipeline_config(
    provider: Any,
    tool_context: Any,
    autocompact_tracking: AutoCompactTracking | None,
) -> PipelineConfig:
    """The minimal correct PipelineConfig for the live surfaces.

    ch05 round-4 GAP A — mirrors the test-only ``QueryEngine``'s
    construction (``query/engine.py:233-250``) exactly: provider + model +
    the read-file fingerprints (so post-compact file restoration fires) +
    the SESSION-scoped ``autocompact_tracking``. The tracking instance MUST
    outlive single turns — the 3-consecutive-failures circuit breaker
    counts across prompts; a per-turn instance would reset it every turn
    (the exact reason the engine holds one at ``engine.py:74-79``).
    """
    fingerprints = getattr(tool_context, "read_file_fingerprints", None) or {}
    read_file_state = {
        str(path): {"timestamp": fp[0]} for path, fp in fingerprints.items() if isinstance(fp, (tuple, list)) and fp
    }
    model = getattr(provider, "model", "") or ""
    context_window = 200_000
    max_output_tokens = None
    if model:
        try:
            from src.models.context import (
                get_context_window_for_model,
                get_model_max_output_tokens,
            )

            context_window = get_context_window_for_model(model, base_url=getattr(provider, "base_url", None))
            max_output_tokens = get_model_max_output_tokens(model, base_url=getattr(provider, "base_url", None))
        except Exception:
            logger.debug("model context-window resolution failed", exc_info=True)
    return PipelineConfig(
        provider=provider,
        model=model,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        read_file_state=read_file_state or None,
        autocompact_tracking=autocompact_tracking,
    )


class CompressionPipeline:
    """
    Orchestrates the 5-layer compression pipeline.

    Usage::

        pipeline = CompressionPipeline(config)
        result = await pipeline.run(messages, input_token_count)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    async def run(
        self,
        messages: list[Message],
        input_token_count: int = 0,
    ) -> CompressionResult:
        """
        Run all compression layers in order, short-circuiting if enough
        tokens are freed by early layers.

        Args:
            messages: Current conversation messages.
            input_token_count: Estimated input token count (for autocompact decision).

        Returns:
            ``CompressionResult`` with the (potentially modified) messages,
            total tokens saved, and which layers were applied.
        """
        cfg = self._config
        total_saved = 0
        layers_applied: list[str] = []
        current_messages = messages

        # F-106: lazy compression gate. When est_input_tokens is well
        # below context_window * skip_ratio, all five layers would be
        # no-ops; short-circuit before doing any work.  A zero (or
        # negative) input_token_count means "not measured" — skip the
        # gate so the pipeline runs normally for backward compat.
        if input_token_count > 0:
            effective_ratio = resolve_skip_ratio_from_env(cfg.gate_skip_ratio)
            gate_should_run, gate_reason = should_run_compression_pipeline(
                est_input_tokens=input_token_count,
                context_window=cfg.context_window,
                skip_ratio=effective_ratio,
                query_source=cfg.gate_query_source,
                transition_reason=cfg.gate_transition_reason,
                previous_pipeline_errored=cfg.gate_previous_pipeline_errored,
            )
            if not gate_should_run:
                logger.debug(
                    "F-106 compression pipeline skipped reason=%s est=%d threshold=%d",
                    gate_reason,
                    input_token_count,
                    int(cfg.context_window * effective_ratio),
                )
                return CompressionResult(
                    messages=messages,
                    tokens_saved=0,
                    layers_applied=[f"skipped:{gate_reason}"],
                )

        # --- Layer 1: Tool Result Budget ---
        try:
            current_messages, saved = apply_tool_result_budget(
                current_messages,
                budget_dir=cfg.budget_dir,
                max_result_tokens=cfg.max_result_tokens,
            )
            if saved > 0:
                total_saved += saved
                layers_applied.append("tool_result_budget")
                logger.debug("Layer 1 (tool_result_budget): saved %d tokens", saved)
                if total_saved >= cfg.early_exit_tokens:
                    return CompressionResult(
                        messages=current_messages,
                        tokens_saved=total_saved,
                        layers_applied=layers_applied,
                    )
        except Exception:
            logger.warning("Layer 1 (tool_result_budget) failed", exc_info=True)

        # --- Layer 2: Snip Compact ---
        try:
            current_messages, saved = snip_compact(
                current_messages,
                keep_recent=cfg.snip_keep_recent,
            )
            if saved > 0:
                total_saved += saved
                layers_applied.append("snip_compact")
                logger.debug("Layer 2 (snip_compact): saved %d tokens", saved)
                if total_saved >= cfg.early_exit_tokens:
                    return CompressionResult(
                        messages=current_messages,
                        tokens_saved=total_saved,
                        layers_applied=layers_applied,
                    )
        except Exception:
            logger.warning("Layer 2 (snip_compact) failed", exc_info=True)

        # --- Layer 3: Microcompact ---
        # Gated by mc_enabled (default False) to match TS where microcompact
        # is a no-op on the main thread (time-based disabled, cached MC uses
        # API cache_edits, legacy removed).  Clearing tool results locally
        # breaks file_unchanged (model told to "refer to earlier content"
        # that microcompact already erased → falls back to Bash cat).
        if cfg.mc_enabled:
            try:
                current_messages, saved = microcompact_typed_messages(
                    current_messages,
                    keep_recent=cfg.mc_keep_recent,
                    time_config=cfg.mc_time_config,
                    force=True,
                )
                if saved > 0:
                    total_saved += saved
                    layers_applied.append("microcompact")
                    logger.debug("Layer 3 (microcompact): saved %d tokens", saved)
                    if total_saved >= cfg.early_exit_tokens:
                        return CompressionResult(
                            messages=current_messages,
                            tokens_saved=total_saved,
                            layers_applied=layers_applied,
                        )
            except Exception:
                logger.warning("Layer 3 (microcompact) failed", exc_info=True)

        # --- Layer 4: Context Collapse ---
        try:
            store = cfg.collapse_store or get_context_collapse_state()
            if store is not None and store.enabled and store.commits:
                current_messages = store.project_view(current_messages)
                layers_applied.append("context_collapse")
                logger.debug("Layer 4 (context_collapse): projected %d commits", len(store.commits))
        except Exception:
            logger.warning("Layer 4 (context_collapse) failed", exc_info=True)

        # --- Layer 5: Autocompact ---
        autocompact_result = None
        if cfg.provider is not None and cfg.model:
            # F-9: when a goal is active, raise the autocompact threshold
            # (compact less aggressively) and pre-filter goal-steering
            # messages so they survive compaction.
            effective_threshold = cfg.autocompact_threshold
            filtered_messages = current_messages
            if cfg.goal_active:
                effective_threshold = min(effective_threshold + 0.10, 0.95)
                # Remove goal-steering messages before the autocompact
                # check so the token estimate doesn't count them, but
                # keep them in the final message stream.
                filtered_messages = [m for m in current_messages if not _is_goal_steering_message(m)]
            try:
                result = await auto_compact_if_needed(
                    filtered_messages if cfg.goal_active else current_messages,
                    input_token_count - total_saved,
                    cfg.context_window,
                    cfg.provider,
                    cfg.model,
                    max_output_tokens=cfg.max_output_tokens,
                    threshold_fraction=cfg.autocompact_threshold,
                    tracking=cfg.autocompact_tracking,
                    custom_instructions=cfg.custom_instructions,
                    read_file_state=cfg.read_file_state,
                    plan_file_path=cfg.plan_file_path,
                    memory_paths=cfg.memory_paths,
                )
                if result is not None:
                    total_saved += result.tokens_saved
                    layers_applied.append("autocompact")
                    autocompact_result = result
                    # R6 — APPLY the compaction to the working messages. This
                    # was the bug: the pipeline computed the summary (an LLM
                    # call, cost incurred) but returned the UNCOMPACTED
                    # current_messages, so query() kept the full conversation
                    # and auto-compact re-fired every turn without ever
                    # shrinking the context — the exact opposite of what a
                    # long task needs. Assemble the compacted conversation the
                    # way the manual /compact path does (compact_service/
                    # service.py:101-104): the summary (a USER message — safe
                    # as the first message) + kept recent messages + the
                    # post-compact attachments (file-state a coding agent needs
                    # after compaction). The system boundary_marker is
                    # bookkeeping for the persisted conversation and is omitted
                    # from the working set (as the reactive path does), so the
                    # working messages stay API-clean.
                    n_before = len(current_messages)
                    current_messages = (
                        list(result.summary_messages) + list(result.messages_to_keep) + list(result.attachments)
                    )
                    logger.debug(
                        "Layer 5 (autocompact): saved %d tokens, %d -> %d msgs",
                        result.tokens_saved,
                        n_before,
                        len(current_messages),
                    )
            except Exception:
                logger.warning("Layer 5 (autocompact) failed", exc_info=True)

        return CompressionResult(
            messages=current_messages,
            tokens_saved=total_saved,
            layers_applied=layers_applied,
            autocompact_result=autocompact_result,
        )


def _is_goal_steering_message(message: Message) -> bool:
    """Check if a message is a goal-steering injection (F-9).

    Goal-steering prompts are wrapped in ``<goal-steering type=...>``
    XML tags.  These must survive autocompact so the model never loses
    sight of the active objective.
    """
    if not hasattr(message, "content"):
        return False
    content = message.content
    if isinstance(content, str):
        return "<goal-steering" in content
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None) or getattr(block, "content", None)
            if isinstance(text, str) and "<goal-steering" in text:
                return True
    return False


async def run_compression_pipeline(
    messages: list[Message],
    input_token_count: int = 0,
    config: PipelineConfig | None = None,
) -> CompressionResult:
    """
    Convenience function: run the full compression pipeline.

    Args:
        messages: Current conversation messages.
        input_token_count: Estimated input token count.
        config: Pipeline configuration.

    Returns:
        ``CompressionResult``
    """
    pipeline = CompressionPipeline(config)
    return await pipeline.run(messages, input_token_count)
