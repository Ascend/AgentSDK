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

"""Tests for :class:`BriefSummaryBuilder`.

The builder is a deterministic Markdown formatter that consumes a
:class:`BriefSummarySnapshot` and returns a small status string. No
LLM calls. Output should be stable for a given snapshot.
"""

from __future__ import annotations


import pytest

from src.services.kairos import (
    BriefGenerationError,
    BriefSummaryBuilder,
    BriefSummarySnapshot,
)


def _snapshot(**overrides) -> BriefSummarySnapshot:
    base = dict(
        agent_id="agent1",
        session_id="sess-abc",
        tick_number=7,
        pending_tasks=(),
        last_action=None,
        metadata={},
        captured_at=1749696000.0,
    )
    base.update(overrides)
    return BriefSummarySnapshot(**base)


class TestBriefSummaryBuilderConstruction:
    def test_default_constructor(self) -> None:
        b = BriefSummaryBuilder()
        # Defaults are positive int, no metadata, real clock.
        snap = _snapshot()
        out = b.build(snap)
        assert "[brief] agent=agent1" in out

    def test_max_pending_must_be_positive_int(self) -> None:
        with pytest.raises(ValueError, match="max_pending_tasks"):
            BriefSummaryBuilder(max_pending_tasks=0)
        with pytest.raises(ValueError, match="max_pending_tasks"):
            BriefSummaryBuilder(max_pending_tasks=-3)
        with pytest.raises(ValueError, match="max_pending_tasks"):
            BriefSummaryBuilder(max_pending_tasks="5")  # type: ignore[arg-type]


class TestBriefSummaryBuilderOutput:
    def test_headline_contains_agent_session_tick(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot())
        assert "[brief] agent=agent1 session=sess-abc tick=#7" in out

    def test_pending_none(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot(pending_tasks=()))
        assert "pending: (none)" in out

    def test_pending_listed(self) -> None:
        snap = _snapshot(pending_tasks=("task-a", "task-b"))
        out = BriefSummaryBuilder().build(snap)
        assert "pending:" in out
        assert "  - task-a" in out
        assert "  - task-b" in out

    def test_pending_truncated_with_overflow(self) -> None:
        snap = _snapshot(pending_tasks=tuple(f"task-{i}" for i in range(7)))
        out = BriefSummaryBuilder(max_pending_tasks=3).build(snap)
        assert "  - task-0" in out
        assert "  - task-2" in out
        # task-3..6 should be summarized as +4 more.
        assert "…(+4 more)" in out
        assert "  - task-6" not in out

    def test_last_action_rendered(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot(last_action="approved plan"))
        assert "last: approved plan" in out

    def test_no_last_action_omitted(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot(last_action=None))
        assert "last:" not in out

    def test_metadata_hidden_by_default(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot(metadata={"plan_id": "P-86"}))
        assert "plan_id" not in out
        assert "meta:" not in out

    def test_metadata_rendered_when_enabled(self) -> None:
        out = BriefSummaryBuilder(include_metadata=True).build(
            _snapshot(metadata={"plan_id": "P-86", "agent": "claude"})
        )
        assert "meta:" in out
        assert "plan_id: P-86" in out
        assert "agent: claude" in out

    def test_captured_at_uses_local_iso(self) -> None:
        out = BriefSummaryBuilder().build(_snapshot(captured_at=1749696000.0))
        assert "captured:" in out
        # ISO 8601 starts with the year.
        assert "2026" in out or "2025" in out  # TZ dependent

    def test_deterministic_for_same_snapshot(self) -> None:
        snap = _snapshot()
        out_a = BriefSummaryBuilder().build(snap)
        out_b = BriefSummaryBuilder().build(snap)
        assert out_a == out_b


class TestBriefSummaryBuilderValidation:
    def test_build_rejects_non_snapshot(self) -> None:
        with pytest.raises(BriefGenerationError, match="BriefSummarySnapshot"):
            BriefSummaryBuilder().build("not a snapshot")  # type: ignore[arg-type]
        with pytest.raises(BriefGenerationError, match="BriefSummarySnapshot"):
            BriefSummaryBuilder().build(None)  # type: ignore[arg-type]

    def test_backwards_compatible_generate_alias(self) -> None:
        snap = _snapshot()
        b = BriefSummaryBuilder()
        # Both methods should produce the same output during the rename.
        assert b.build(snap) == b.generate(snap)


class TestBriefSummaryBuilderNaming:
    def test_old_brief_generator_name_no_longer_exported(self) -> None:
        # Sanity: ensure the rename is observable through the public API.
        import src.services.kairos as k

        assert hasattr(k, "BriefSummaryBuilder")
        assert not hasattr(k, "BriefGenerator")
