#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Trigger tests: TokenThresholdTrigger, Emergency413Trigger, CompositeTrigger."""

from __future__ import annotations

import pytest

from src.services.context_collapse.exceptions import ContextLengthExceededError
from src.services.context_collapse.tokens import HeuristicTokenCounter
from src.services.context_collapse.trigger import (
    CollapseKind,
    CompositeTrigger,
    Emergency413Trigger,
    TokenThresholdTrigger,
    TriggerContext,
    default_composite_trigger,
    default_error_predicate,
)


# ---------------------------------------------------------------------------
# TokenThresholdTrigger
# ---------------------------------------------------------------------------


def _msgs(n_words: int) -> list[dict]:
    return [{"role": "user", "content": " ".join(["w"] * n_words)}]


def _ctx(window: int = 1000, fraction: float = 0.80) -> TriggerContext:
    """Build a TriggerContext with the trigger's threshold_fraction."""
    return TriggerContext(context_window=window, threshold_fraction=fraction)


def test_threshold_trigger_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        TokenThresholdTrigger(counter=HeuristicTokenCounter(), threshold_fraction=0.0)
    with pytest.raises(ValueError):
        TokenThresholdTrigger(counter=HeuristicTokenCounter(), threshold_fraction=1.5)


def test_threshold_trigger_rejects_negative_keep_recent() -> None:
    with pytest.raises(ValueError):
        TokenThresholdTrigger(counter=HeuristicTokenCounter(), keep_recent=-1)


def test_threshold_trigger_returns_noop_for_no_messages() -> None:
    trig = TokenThresholdTrigger(counter=HeuristicTokenCounter())
    decision = trig.decide([], _ctx())
    assert decision.kind is CollapseKind.NOOP
    assert decision.reason == "no messages"


def test_threshold_trigger_returns_noop_under_budget() -> None:
    counter = HeuristicTokenCounter()
    trig = TokenThresholdTrigger(counter=counter, threshold_fraction=0.80)
    # A few short messages should fit easily under 1000*0.80 == 800 tokens.
    decision = trig.decide(_msgs(5), _ctx(fraction=0.80))
    assert decision.kind is CollapseKind.NOOP
    assert decision.token_estimate is not None
    assert decision.token_estimate.tokens <= 800


def test_threshold_trigger_returns_full_over_budget() -> None:
    counter = HeuristicTokenCounter()
    trig = TokenThresholdTrigger(counter=counter, threshold_fraction=0.05, keep_recent=3)
    # 200 words -> ~260 tokens; budget = 1000 * 0.05 = 50; should trigger.
    decision = trig.decide(_msgs(200), _ctx(fraction=0.05))
    assert decision.kind is CollapseKind.FULL
    assert decision.count == 3


def test_threshold_trigger_returns_partial_when_count_set() -> None:
    counter = HeuristicTokenCounter()
    trig = TokenThresholdTrigger(counter=counter, threshold_fraction=0.05, partial_archive_count=10)
    decision = trig.decide(_msgs(200), _ctx(fraction=0.05))
    assert decision.kind is CollapseKind.PARTIAL
    assert decision.count == 10


def test_threshold_trigger_decision_includes_token_estimate() -> None:
    counter = HeuristicTokenCounter()
    trig = TokenThresholdTrigger(counter=counter, threshold_fraction=0.01)
    decision = trig.decide(_msgs(200), _ctx(fraction=0.01))
    assert decision.token_estimate is not None
    assert decision.token_estimate.counter_name == "heuristic"


# ---------------------------------------------------------------------------
# Emergency413Trigger
# ---------------------------------------------------------------------------


def test_emergency_trigger_returns_noop_when_no_error() -> None:
    trig = Emergency413Trigger()
    decision = trig.decide(_msgs(5), _ctx())
    assert decision.kind is CollapseKind.NOOP


def test_emergency_trigger_returns_noop_for_unrelated_error() -> None:
    trig = Emergency413Trigger()
    decision = trig.decide(_msgs(5), _ctx().__class__(context_window=1000, last_error=ValueError("oops")))
    assert decision.kind is CollapseKind.NOOP
    assert "not 413-class" in decision.reason


def test_emergency_trigger_handles_canonical_exception() -> None:
    trig = Emergency413Trigger(keep_recent=2)
    decision = trig.decide(
        _msgs(5),
        TriggerContext(
            context_window=1000,
            last_error=ContextLengthExceededError("over"),
        ),
    )
    assert decision.kind is CollapseKind.FULL
    assert decision.count == 2


def test_emergency_trigger_handles_status_code_413() -> None:
    class FakeHTTPError(Exception):
        status_code = 413

    trig = Emergency413Trigger()
    decision = trig.decide(
        _msgs(5),
        TriggerContext(context_window=1000, last_error=FakeHTTPError("boom")),
    )
    assert decision.kind is CollapseKind.FULL


def test_emergency_trigger_handles_string_match() -> None:
    trig = Emergency413Trigger()
    decision = trig.decide(
        _msgs(5),
        TriggerContext(
            context_window=1000,
            last_error=RuntimeError("context_length_exceeded: too long"),
        ),
    )
    assert decision.kind is CollapseKind.FULL


def test_emergency_trigger_validates_inputs() -> None:
    with pytest.raises(TypeError):
        Emergency413Trigger(predicate="not callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Emergency413Trigger(keep_recent=-1)
    with pytest.raises(ValueError):
        Emergency413Trigger(archive_count=-1)


# ---------------------------------------------------------------------------
# default_error_predicate
# ---------------------------------------------------------------------------


def test_default_error_predicate_matches_various_signals() -> None:
    assert default_error_predicate(ContextLengthExceededError("x"))
    assert default_error_predicate(RuntimeError("413 too large"))
    assert default_error_predicate(RuntimeError("context_length too long"))
    assert default_error_predicate(RuntimeError("context length too long"))
    assert default_error_predicate(RuntimeError("too long to send"))

    class HasStatus:
        status_code = 413

    assert default_error_predicate(HasStatus())

    assert not default_error_predicate(ValueError("something else"))
    assert not default_error_predicate(KeyError("x"))


def test_default_error_predicate_matches_request_too_large_class_name() -> None:
    """The source checks ``request_too_large`` against the (lowercased) class name."""

    class Request_Too_Large_Error(Exception):
        pass

    assert default_error_predicate(Request_Too_Large_Error("boom"))


def test_default_error_predicate_ignores_request_too_large_only_in_message() -> None:
    """A plain RuntimeError with that string in its message is NOT matched.

    The predicate checks ``request_too_large`` against the class name; a
    RuntimeError never carries that name. This is a quirk worth documenting
    so future maintainers don't trip over it.
    """
    assert not default_error_predicate(RuntimeError("request_too_large"))


# ---------------------------------------------------------------------------
# CompositeTrigger
# ---------------------------------------------------------------------------


def test_composite_trigger_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        CompositeTrigger([])


def test_composite_trigger_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError):
        CompositeTrigger([TokenThresholdTrigger(HeuristicTokenCounter())], policy="maybe")


def test_composite_any_policy_returns_most_aggressive() -> None:
    threshold = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.05,
        keep_recent=4,
    )
    emergency = Emergency413Trigger(keep_recent=2)
    composite = CompositeTrigger([threshold, emergency], policy="any")

    decision = composite.decide(_msgs(200), _ctx(fraction=0.05))
    # Threshold alone returns FULL; composite agrees.
    assert decision.kind is CollapseKind.FULL


def test_composite_any_policy_noop_when_all_subs_noop() -> None:
    threshold = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.80,
        keep_recent=4,
    )
    emergency = Emergency413Trigger()
    composite = CompositeTrigger([threshold, emergency], policy="any")

    decision = composite.decide(_msgs(5), _ctx(window=100_000))
    assert decision.kind is CollapseKind.NOOP


def test_composite_all_policy_requires_every_sub_to_ask() -> None:
    threshold = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.80,
    )
    emergency = Emergency413Trigger()
    composite = CompositeTrigger([threshold, emergency], policy="all")

    decision = composite.decide(
        _msgs(5),
        TriggerContext(
            context_window=100_000,
            threshold_fraction=0.80,
            last_error=ContextLengthExceededError("x"),
        ),
    )
    # Threshold says NOOP (under budget), emergency says FULL.
    # "all" requires everyone to agree, so composite is NOOP.
    assert decision.kind is CollapseKind.NOOP


def test_composite_all_policy_collapses_when_all_agree() -> None:
    threshold = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.01,
        keep_recent=3,
    )
    emergency = Emergency413Trigger(keep_recent=2)
    composite = CompositeTrigger([threshold, emergency], policy="all")

    decision = composite.decide(
        _msgs(200),
        TriggerContext(
            context_window=1000,
            threshold_fraction=0.01,
            last_error=ContextLengthExceededError("x"),
        ),
    )
    assert decision.kind is CollapseKind.FULL


def test_composite_prefers_full_over_partial() -> None:
    partial = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.05,
        partial_archive_count=10,
    )
    full = TokenThresholdTrigger(
        counter=HeuristicTokenCounter(),
        threshold_fraction=0.05,
        keep_recent=2,
    )
    composite = CompositeTrigger([partial, full], policy="any")
    decision = composite.decide(_msgs(200), _ctx(fraction=0.05))
    assert decision.kind is CollapseKind.FULL


# ---------------------------------------------------------------------------
# default_composite_trigger
# ---------------------------------------------------------------------------


def test_default_composite_trigger_noop_under_budget() -> None:
    composite = default_composite_trigger(context_window=100_000)
    decision = composite.decide(_msgs(5), TriggerContext(context_window=100_000))
    assert decision.kind is CollapseKind.NOOP


def test_default_composite_trigger_collapses_over_budget() -> None:
    composite = default_composite_trigger(context_window=1000, threshold_fraction=0.05, keep_recent=4)
    decision = composite.decide(
        _msgs(200),
        TriggerContext(context_window=1000, threshold_fraction=0.05),
    )
    assert decision.kind is CollapseKind.FULL
    # Default emergency trigger uses keep_recent=2, threshold uses 4.
    # Composite picks the first sub-trigger that asked (most-aggressive pass),
    # so the count is 4 (threshold's keep_recent) or 2 (emergency's).
    assert decision.count in (2, 4)


def test_default_composite_trigger_collapses_on_413_error() -> None:
    composite = default_composite_trigger(context_window=100_000)
    decision = composite.decide(
        _msgs(5),
        TriggerContext(
            context_window=100_000,
            last_error=ContextLengthExceededError("over"),
        ),
    )
    assert decision.kind is CollapseKind.FULL
    assert decision.count == 2  # default emergency keep_recent
