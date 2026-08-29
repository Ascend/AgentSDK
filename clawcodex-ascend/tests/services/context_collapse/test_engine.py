#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""End-to-end CollapseEngine tests: evaluate / apply / recover_from_413."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import threading

import pytest

from clawcodex_ext.services.compact.context_collapse import (
    CollapseCommit,
    ContextCollapseStore,
)
from src.services.context_collapse.engine import (
    CollapseEngine,
    CollapseEngineConfig,
)
from src.services.context_collapse.exceptions import (
    ContextLengthExceededError,
)
from src.services.context_collapse.summary import HeadlineSummaryGenerator
from src.services.context_collapse.tokens import HeuristicTokenCounter
from src.services.context_collapse.trigger import (
    CollapseDecision,
    CollapseKind,
    CompositeTrigger,
    TokenThresholdTrigger,
)


def _msg(text: str, uuid: str = "u") -> dict:
    return {"role": "user", "content": text, "uuid": uuid}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_engine_default_construction() -> None:
    engine = CollapseEngine()
    assert isinstance(engine.store, ContextCollapseStore)
    assert engine.store.commits == []


def test_engine_uses_provided_store() -> None:
    pre = ContextCollapseStore()
    engine = CollapseEngine(store=pre)
    assert engine.store is pre


def test_engine_rejects_non_store() -> None:
    with pytest.raises(TypeError):
        CollapseEngine(store="not a store")  # type: ignore[arg-type]


def test_engine_uses_provided_trigger_and_summary() -> None:
    counter = HeuristicTokenCounter()
    custom_trigger = TokenThresholdTrigger(counter=counter, threshold_fraction=0.10)
    custom_summary = HeadlineSummaryGenerator(max_headlines=4)
    engine = CollapseEngine(
        trigger=custom_trigger,
        summary_generator=custom_summary,
        config=CollapseEngineConfig(context_window=500),
    )
    assert engine.trigger is custom_trigger
    assert engine.summary_generator is custom_summary


def test_engine_uses_default_composite_trigger() -> None:
    engine = CollapseEngine()
    assert isinstance(engine.trigger, CompositeTrigger)


def test_engine_default_uses_headline_summary() -> None:
    engine = CollapseEngine()
    assert isinstance(engine.summary_generator, HeadlineSummaryGenerator)


# ---------------------------------------------------------------------------
# evaluate (no mutation)
# ---------------------------------------------------------------------------


def test_evaluate_noop_does_not_mutate_store() -> None:
    engine = CollapseEngine()
    msgs = [_msg("hello world", uuid="a")]
    decision = engine.evaluate(msgs)
    assert decision.kind is CollapseKind.NOOP
    assert engine.store.commits == []


def test_evaluate_under_budget_is_noop() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000))
    decision = engine.evaluate([_msg("tiny", uuid="a")])
    assert decision.kind is CollapseKind.NOOP


def test_evaluate_over_budget_returns_full() -> None:
    engine = CollapseEngine(
        config=CollapseEngineConfig(
            context_window=200,
            threshold_fraction=0.50,
            keep_recent=2,
        )
    )
    msgs = [_msg("x " * 200, uuid=str(i)) for i in range(20)]
    decision = engine.evaluate(msgs)
    assert decision.kind in (CollapseKind.FULL, CollapseKind.PARTIAL)


def test_evaluate_threads_last_error() -> None:
    engine = CollapseEngine()
    err = ContextLengthExceededError("boom")
    decision = engine.evaluate([_msg("x", uuid="a")], last_error=err)
    # The default composite trigger sees the error via the 413 sub-trigger.
    assert decision.kind is CollapseKind.FULL


def test_evaluate_threads_hints() -> None:
    """Hints are forwarded into the TriggerContext but do not change behavior."""
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000))
    decision = engine.evaluate([_msg("x", uuid="a")], hints={"model": "test"})
    assert decision.kind is CollapseKind.NOOP


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def test_apply_noop_returns_empty_result() -> None:
    engine = CollapseEngine()
    msgs = [_msg("x", uuid="a")]
    decision = engine.evaluate(msgs)
    result = engine.apply(msgs, decision)
    assert result.applied is False
    assert result.kind is CollapseKind.NOOP
    assert result.archived_count == 0
    assert result.summary == ""
    assert engine.store.commits == []


def test_apply_full_archives_and_records_commit() -> None:
    engine = CollapseEngine(
        config=CollapseEngineConfig(
            context_window=200,
            threshold_fraction=0.10,
            keep_recent=2,
        )
    )
    msgs = [_msg(f"text {i}", uuid=f"u{i}") for i in range(10)]
    decision = engine.evaluate(msgs)
    assert decision.kind is CollapseKind.FULL

    result = engine.apply(msgs, decision)
    assert result.applied is True
    assert result.kind is CollapseKind.FULL
    assert result.archived_count == 8  # 10 - 2 kept
    assert result.summary
    assert result.boundary_text is not None
    assert result.boundary_text.startswith("[CTX-COLLAPSE:")
    assert len(engine.store.commits) == 1
    commit = engine.store.commits[0]
    assert isinstance(commit, CollapseCommit)
    assert len(commit.archived) == 8


def test_apply_partial_archives_first_n() -> None:
    engine = CollapseEngine()
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(10)]
    decision = CollapseDecision(
        kind=CollapseKind.PARTIAL,
        reason="forced partial",
        count=4,
    )
    result = engine.apply(msgs, decision)
    assert result.applied is True
    assert result.archived_count == 4
    assert len(engine.store.commits) == 1
    assert engine.store.commits[0].archived == ["u0", "u1", "u2", "u3"]


def test_apply_partial_with_zero_count_falls_back_to_32() -> None:
    engine = CollapseEngine()
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(50)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=0)
    result = engine.apply(msgs, decision)
    assert result.applied is True
    assert result.archived_count == 32


def test_apply_partial_with_missing_count_falls_back_to_32() -> None:
    engine = CollapseEngine()
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(50)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced")
    result = engine.apply(msgs, decision)
    assert result.archived_count == 32


def test_apply_full_with_empty_messages_returns_not_applied() -> None:
    engine = CollapseEngine()
    decision = CollapseDecision(kind=CollapseKind.FULL, reason="forced", count=0)
    result = engine.apply([], decision)
    assert result.applied is False
    assert "nothing to archive" in result.notes


def test_apply_preserves_existing_commits() -> None:
    engine = CollapseEngine()
    engine.store.add_commit(["u1"], "old summary")
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(5)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=2)
    engine.apply(msgs, decision)
    assert len(engine.store.commits) == 2
    assert engine.store.commits[0].summary == "old summary"


def test_apply_summary_uses_registered_generator() -> None:
    seen: list[int] = []

    class CountingSummary:
        name = "counting"

        def summarize(self, messages):
            seen.append(len(messages))
            return "CUSTOM-SUMMARY"

    engine = CollapseEngine(summary_generator=CountingSummary())
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(5)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=3)
    result = engine.apply(msgs, decision)
    assert result.summary == "CUSTOM-SUMMARY"
    assert seen == [3]


def test_apply_extracts_uuid_from_object_messages() -> None:
    class M:
        def __init__(self, text: str, uuid: str) -> None:
            self.content = text
            self.uuid = uuid

    engine = CollapseEngine()
    msgs = [M(f"x{i}", uuid=f"u{i}") for i in range(5)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=2)
    engine.apply(msgs, decision)
    assert engine.store.commits[0].archived == ["u0", "u1"]


# ---------------------------------------------------------------------------
# decide_and_apply (one-shot)
# ---------------------------------------------------------------------------


def test_decide_and_apply_noop() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000))
    result = engine.decide_and_apply([_msg("tiny", uuid="a")])
    assert result.applied is False


def test_decide_and_apply_collapses_over_budget() -> None:
    engine = CollapseEngine(
        config=CollapseEngineConfig(
            context_window=200,
            threshold_fraction=0.05,
            keep_recent=3,
        )
    )
    msgs = [_msg(f"text {i}", uuid=f"u{i}") for i in range(20)]
    result = engine.decide_and_apply(msgs)
    assert result.applied is True
    assert result.archived_count == 17
    assert engine.store.commits


def test_decide_and_apply_propagates_error_to_trigger() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000))
    err = ContextLengthExceededError("over")
    # Many messages so the FULL keep_recent=2 actually archives something.
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(10)]
    result = engine.decide_and_apply(msgs, last_error=err)
    assert result.applied is True
    assert result.kind is CollapseKind.FULL
    assert result.archived_count == 8  # 10 - 2 kept


def test_decide_and_apply_noop_when_keep_recent_exceeds_messages() -> None:
    """When keep_recent >= len(messages), the FULL collapse archives nothing."""
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000, keep_recent=10))
    err = ContextLengthExceededError("over")
    result = engine.decide_and_apply([_msg("x", uuid="a")], last_error=err)
    assert result.applied is False
    assert "nothing to archive" in result.notes


# ---------------------------------------------------------------------------
# recover_from_413
# ---------------------------------------------------------------------------


def test_recover_from_413_collapses_and_records_note() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=100_000, keep_recent=2))
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(20)]
    err = ContextLengthExceededError("over")
    result = engine.recover_from_413(msgs, err)
    assert result.applied is True
    assert result.kind is CollapseKind.FULL
    assert "recovered via emergency collapse" in result.notes
    assert engine.store.commits


def test_recover_from_413_raises_when_max_attempts_invalid() -> None:
    engine = CollapseEngine()
    msgs = [_msg("x", uuid="a")]
    with pytest.raises(ValueError):
        engine.recover_from_413(msgs, ContextLengthExceededError("x"), max_attempts=0)


def test_recover_from_413_raises_when_no_messages_to_archive() -> None:
    """With only 1 message and keep_recent=4, FULL archives 0; result is no-op."""
    engine = CollapseEngine(config=CollapseEngineConfig(keep_recent=10))
    msgs = [_msg("x", uuid="a")]
    err = ContextLengthExceededError("over")
    with pytest.raises(ContextLengthExceededError):
        engine.recover_from_413(msgs, err)


def test_recover_from_413_raises_when_trigger_says_noop() -> None:
    """When the trigger returns NOOP, recover_from_413 surfaces the original error.

    This documents the actual behaviour: the engine cannot recover from a 413
    if its trigger refuses to fold, so the error is re-raised wrapped in a
    ContextLengthExceededError that chains the original.
    """

    class NoopTrigger:
        name = "noop-stub"

        def decide(self, messages, context):
            return CollapseDecision(kind=CollapseKind.NOOP, reason="stub")

    engine = CollapseEngine(trigger=NoopTrigger())
    msgs = [_msg("x", uuid="a")]
    err = ContextLengthExceededError("over")
    with pytest.raises(ContextLengthExceededError) as exc_info:
        engine.recover_from_413(msgs, err)
    assert exc_info.value.__cause__ is err


# ---------------------------------------------------------------------------
# _split internal
# ---------------------------------------------------------------------------


def test_split_full_with_keep_larger_than_messages() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(keep_recent=100))
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(5)]
    decision = CollapseDecision(kind=CollapseKind.FULL, reason="forced", count=100)
    result = engine.apply(msgs, decision)
    # keep >= len => no archive
    assert result.applied is False


def test_split_full_with_zero_keep_archives_all() -> None:
    engine = CollapseEngine()
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(5)]
    decision = CollapseDecision(kind=CollapseKind.FULL, reason="forced", count=0)
    result = engine.apply(msgs, decision)
    assert result.applied is True
    assert result.archived_count == 5


def test_split_partial_with_negative_count_falls_back() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(partial_archive_count=8))
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(20)]
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=-1)
    result = engine.apply(msgs, decision)
    # count<=0 falls back to 32
    assert result.archived_count == 20  # capped at len(msgs)
    assert result.applied is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_engine_thread_safe_under_concurrent_apply() -> None:
    """Many threads applying decisions concurrently should not corrupt the store."""
    engine = CollapseEngine()
    decision = CollapseDecision(kind=CollapseKind.PARTIAL, reason="forced", count=2)

    def worker(i: int) -> None:
        msgs = [_msg(f"x{i}-{j}", uuid=f"u{i}-{j}") for j in range(4)]
        engine.apply(msgs, decision)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each apply wrote 1 commit with 2 archived UUIDs; 20 workers -> 20 commits.
    assert len(engine.store.commits) == 20
    # Every commit must have exactly 2 archived UUIDs.
    for commit in engine.store.commits:
        assert len(commit.archived) == 2
    # All UUIDs should be unique across commits (no overlap between workers).
    seen: set[str] = set()
    for commit in engine.store.commits:
        for uuid in commit.archived:
            assert uuid not in seen
            seen.add(uuid)
    assert len(seen) == 40


def test_engine_thread_safe_under_concurrent_evaluate() -> None:
    engine = CollapseEngine(config=CollapseEngineConfig(context_window=200, threshold_fraction=0.05))
    msgs = [_msg(f"x{i}", uuid=f"u{i}") for i in range(20)]
    results: list[CollapseDecision] = []
    lock = threading.Lock()

    def worker() -> None:
        decision = engine.evaluate(msgs)
        with lock:
            results.append(decision)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20
    # All decisions should be the same kind (deterministic for the same inputs).
    kinds = {r.kind for r in results}
    assert len(kinds) == 1


# ---------------------------------------------------------------------------
# CollapseEngineConfig defaults
# ---------------------------------------------------------------------------


def test_engine_config_defaults() -> None:
    cfg = CollapseEngineConfig()
    assert cfg.context_window == 200_000
    assert cfg.threshold_fraction == 0.80
    assert cfg.keep_recent == 4
    assert cfg.partial_archive_count == 32
    assert cfg.use_legacy_boundary is True
