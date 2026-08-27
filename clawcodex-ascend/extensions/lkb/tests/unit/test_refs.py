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

"""Unit tests for lkb.refs.NodeRef.

Spec §5.3 — NodeRef identity and canonical form.
LKB-REF-001..005 characterization tests.
"""

from __future__ import annotations

import pytest

from lkb.refs import NodeRef


# ── LKB-REF-001: plan-task round-trip stable serialization ──────────


def test_plan_task_round_trip() -> None:
    """Plan-task NodeRef must round-trip through canonical string form."""
    ref = NodeRef("plan", "task", "T-001")
    assert ref.graph == "plan"
    assert ref.kind == "task"
    assert ref.id == "T-001"

    s = ref.to_str()
    assert s == "plan:task:T-001"
    assert str(ref) == s

    parsed = NodeRef.from_str(s)
    assert parsed == ref
    assert hash(parsed) == hash(ref)


def test_plan_agent_round_trip() -> None:
    """Agent-node refs round-trip as well (not just tasks)."""
    ref = NodeRef("plan", "agent", "agent-a")
    assert ref.to_str() == "plan:agent:agent-a"
    assert NodeRef.from_str("plan:agent:agent-a") == ref


# ── LKB-REF-003: invalid graph / kind / id rejected ────────────────


def test_empty_components_rejected() -> None:
    """Empty graph, kind, or id must raise ValueError."""
    with pytest.raises(ValueError, match="graph.*empty"):
        NodeRef("", "task", "T-1")
    with pytest.raises(ValueError, match="kind.*empty"):
        NodeRef("plan", "", "T-1")
    with pytest.raises(ValueError, match="id.*empty"):
        NodeRef("plan", "task", "")


def test_graph_kind_character_set() -> None:
    """graph/kind must match [A-Za-z0-9_-]+."""
    # valid
    NodeRef("my-graph_v2", "TaskKind-1", "ok")
    # invalid — spaces are invalid
    with pytest.raises(ValueError, match="graph.*match"):
        NodeRef("my graph", "task", "T-1")
    with pytest.raises(ValueError, match="kind.*match"):
        NodeRef("plan", "task kind", "T-1")
    # colon is the field separator — rejected explicitly
    with pytest.raises(ValueError, match="field separator"):
        NodeRef("a:b", "task", "T-1")


def test_from_str_wrong_column_count() -> None:
    """from_str must reject strings with fewer than 2 colons."""
    with pytest.raises(ValueError, match="at least 2 colons"):
        NodeRef.from_str("plan:task")
    # 3+ colons: first two are delimiters, rest goes into id (then validated)
    # This ensures id can contain slashes and other valid chars but not colons
    with pytest.raises(ValueError, match="field separator"):
        NodeRef.from_str("a:b:c:d")


# ── LKB-REF-004: path-traversal chars rejected ─────────────────────


def test_path_traversal_chars_rejected_in_id() -> None:
    """'..' components and backslash must be rejected in id."""
    with pytest.raises(ValueError, match="escape storage path"):
        NodeRef("legacy", "node", "../escape")
    with pytest.raises(ValueError, match="escape storage path"):
        NodeRef("legacy", "node", "a/../b")
    with pytest.raises(ValueError, match="path traversal"):
        NodeRef("legacy", "node", "a\\b")


def test_dot_dot_rejected_as_id() -> None:
    """id '.' and '..' must be rejected (would escape storage paths)."""
    with pytest.raises(ValueError, match="escape storage path"):
        NodeRef("plan", "task", "..")
    with pytest.raises(ValueError, match="path component"):
        NodeRef("plan", "task", ".")


def test_slash_rejected_in_graph_and_kind() -> None:
    """'/' must be rejected in graph and kind too (not just id)."""
    with pytest.raises(ValueError, match="match"):
        NodeRef("a/b", "task", "T-1")
    with pytest.raises(ValueError, match="match"):
        NodeRef("plan", "a/b", "T-1")


def test_control_chars_rejected() -> None:
    """Control characters must be rejected in every component."""
    with pytest.raises(ValueError, match="control"):
        NodeRef("plan\x00", "task", "T-1")
    with pytest.raises(ValueError, match="control"):
        NodeRef("plan", "task\x01", "T-1")
    with pytest.raises(ValueError, match="control"):
        NodeRef("plan", "task", "T-1\x1b")


# ── LKB-REF-005: custom node kind usable after registration ────────


def test_custom_kind_open_set() -> None:
    """New kinds can be introduced without modifying refs.py (open set)."""
    # Risk kind — invented here for this test, works fine
    ref = NodeRef("plan", "risk", "R-042")
    assert ref.kind == "risk"
    assert ref.to_str() == "plan:risk:R-042"
    assert NodeRef.from_str("plan:risk:R-042") == ref


def test_custom_graph_open_set() -> None:
    """New graph names can be introduced (open set, not closed Literal)."""
    ref = NodeRef("deployment", "service", "api-gateway")
    assert ref.graph == "deployment"
    assert ref.to_str() == "deployment:service:api-gateway"


# ── backward-compat: task_id property ──────────────────────────────


def test_task_id_property_for_tasks() -> None:
    """task_id returns self.id when kind == 'task'."""
    ref = NodeRef("plan", "task", "T-001")
    assert ref.task_id == "T-001"


def test_task_id_property_for_non_tasks() -> None:
    """task_id returns None for non-task kinds."""
    ref = NodeRef("plan", "agent", "agent-a")
    assert ref.task_id is None

    ref2 = NodeRef("artifact", "file", "src/x.py")
    assert ref2.task_id is None


# ── equality / hashing ──────────────────────────────────────────────


def test_equality_and_hashing() -> None:
    """Same triple → equal + same hash; different triple → not equal."""
    a = NodeRef("plan", "task", "T-1")
    b = NodeRef("plan", "task", "T-1")
    c = NodeRef("plan", "task", "T-2")

    assert a == b
    assert hash(a) == hash(b)
    assert a != c

    # usable in sets and dict keys
    s = {a, b, c}
    assert len(s) == 2
    d = {a: "first", c: "second"}
    assert d[b] == "first"
