#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

# pylint: disable=use-implicit-booleaness-not-comparison
"""Tests for clawcodex_ext.community_radar.models."""

from __future__ import annotations

from clawcodex_ext.community_radar.models import (
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    Release,
    WatchSource,
    get_level,
    get_path,
    get_root,
    get_subtree,
    is_leaf,
    make_feature_id,
)


def test_watch_source_roundtrip() -> None:
    raw = {
        "name": "aider",
        "repo": "paul-gauthier/aider",
        "track_releases": True,
        "track_commits": True,
        "track_prs": False,
        "release_tag_filter": r"\d+\.\d+\.\d+",
        "changelog_path": "aider/CHANGELOG.md",
        "notes": "Python 生态最活跃的编码 Agent",
        "roadmap_keywords": ["lint", "edit"],
    }
    src = WatchSource.from_dict(raw)
    assert src.name == "aider"
    assert src.track_commits is True
    assert src.roadmap_keywords == ["lint", "edit"]
    again = WatchSource.from_dict(src.to_dict())
    assert again.repo == src.repo
    assert again.notes == src.notes


def test_watch_source_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        WatchSource.from_dict({"name": "", "repo": "x/y"})
    with pytest.raises(ValueError):
        WatchSource.from_dict({"name": "ok", "repo": "missing-slash"})


def test_make_feature_id_stable() -> None:
    a = make_feature_id("aider", "Add lint auto-fix", "new")
    b = make_feature_id("aider", "Add lint auto-fix", "new")
    assert a == b
    c = make_feature_id("aider", "add lint auto-fix", "new")
    assert c == a  # case-insensitive on title
    d = make_feature_id("aider", "Add lint auto-fix", "enhancement")
    assert d != a


def test_feature_record_roundtrip() -> None:
    record = FeatureRecord(
        id="abc123",
        source="aider",
        title="Lint auto-fix",
        description="Aider now auto-fixes lint errors",
        category=FeatureCategory.TOOL_SYSTEM,
        feature_type=FeatureType.NEW,
        released_at="2026-06-15T00:00:00Z",
        url="https://example.com/aider/v1.2.3",
        related_projects=["claude-code"],
        tags=["lint"],
    )
    again = FeatureRecord.from_dict(record.to_dict())
    assert again.category == FeatureCategory.TOOL_SYSTEM
    assert again.feature_type == FeatureType.NEW
    assert again.related_projects == ["claude-code"]


def test_release_to_dict_roundtrip() -> None:
    rel = Release(
        tag="v1.2.3",
        name="Release v1.2.3",
        body="## Added\n- new feature\n",
        published_at="2026-06-15T00:00:00Z",
        url="https://example.com/release/v1.2.3",
        is_prerelease=False,
    )
    again = Release.from_dict(rel.to_dict())
    assert again.tag == "v1.2.3"
    assert again.body.startswith("## Added")
    assert again.is_prerelease is False


def test_feature_score_to_dict() -> None:
    score = FeatureScore(
        record_id="abc",
        overall=72.5,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    payload = score.to_dict()
    assert payload["overall"] == 72.5
    assert payload["dimensions"]["popularity"] == 80.0


# ---------------------------------------------------------------------------
# FeatureCategory hierarchy tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FeatureCategory hierarchy tests (path-based)
# ---------------------------------------------------------------------------


def test_get_path_code_agent_children() -> None:
    """All 12 code-agent leaf categories have path (CODE_AGENT, <self>)."""
    code_agent_children = [
        FeatureCategory.AGENT_LOOP,
        FeatureCategory.TOOL_SYSTEM,
        FeatureCategory.PROVIDER,
        FeatureCategory.PERMISSION,
        FeatureCategory.MEMORY,
        FeatureCategory.MCP,
        FeatureCategory.MULTI_AGENT,
        FeatureCategory.ORCHESTRATOR,
        FeatureCategory.TUI_REPL,
        FeatureCategory.CLI,
        FeatureCategory.OBSERVABILITY,
        FeatureCategory.INFRA,
    ]
    for cat in code_agent_children:
        path = get_path(cat)
        assert len(path) == 2, f"{cat.value} path should have 2 elements"
        assert path[0] == FeatureCategory.CODE_AGENT, f"{cat.value} root should be CODE_AGENT, got {path[0]}"
        assert path[1] == cat


def test_get_path_root_nodes_are_single_element() -> None:
    """CODE_AGENT, EMBODIED_AI, and UNKNOWN are single-element paths."""
    for cat in (FeatureCategory.CODE_AGENT, FeatureCategory.EMBODIED_AI, FeatureCategory.UNKNOWN):
        path = get_path(cat)
        assert len(path) == 1, f"{cat.value} path should be single-element"
        assert path[0] == cat


def test_get_path_spatial_under_embodied() -> None:
    """SPATIAL_INTELLIGENCE path is (EMBODIED_AI, SPATIAL_INTELLIGENCE)."""
    path = get_path(FeatureCategory.SPATIAL_INTELLIGENCE)
    assert path == (FeatureCategory.EMBODIED_AI, FeatureCategory.SPATIAL_INTELLIGENCE)


def test_get_root_returns_root_category() -> None:
    """get_root returns the first element of the path."""
    assert get_root(FeatureCategory.AGENT_LOOP) == FeatureCategory.CODE_AGENT
    assert get_root(FeatureCategory.TOOL_SYSTEM) == FeatureCategory.CODE_AGENT
    assert get_root(FeatureCategory.CODE_AGENT) == FeatureCategory.CODE_AGENT
    assert get_root(FeatureCategory.SPATIAL_INTELLIGENCE) == FeatureCategory.EMBODIED_AI
    assert get_root(FeatureCategory.EMBODIED_AI) == FeatureCategory.EMBODIED_AI
    assert get_root(FeatureCategory.UNKNOWN) == FeatureCategory.UNKNOWN


def test_get_root_value_for_aggregation() -> None:
    """get_root().value gives stable aggregation keys."""
    assert get_root(FeatureCategory.AGENT_LOOP).value == "code_agent"
    assert get_root(FeatureCategory.SPATIAL_INTELLIGENCE).value == "embodied_ai"
    assert get_root(FeatureCategory.EMBODIED_AI).value == "embodied_ai"
    assert get_root(FeatureCategory.UNKNOWN).value == "unknown"


def test_get_level_root_nodes() -> None:
    """Root-only nodes are level 0."""
    for cat in (FeatureCategory.CODE_AGENT, FeatureCategory.EMBODIED_AI, FeatureCategory.UNKNOWN):
        assert get_level(cat) == 0, f"{cat.value} should be level 0"


def test_get_level_children() -> None:
    """Direct children are level 1."""
    assert get_level(FeatureCategory.AGENT_LOOP) == 1
    assert get_level(FeatureCategory.SPATIAL_INTELLIGENCE) == 1
    assert get_level(FeatureCategory.MCP) == 1


def test_get_subtree_code_agent_has_12() -> None:
    """CODE_AGENT subtree should have 12 descendants, not including itself."""
    subtree = get_subtree(FeatureCategory.CODE_AGENT)
    assert len(subtree) == 12
    assert FeatureCategory.AGENT_LOOP in subtree
    assert FeatureCategory.INFRA in subtree
    assert FeatureCategory.CODE_AGENT not in subtree  # root excluded


def test_get_subtree_embodied_ai_has_1() -> None:
    """EMBODIED_AI subtree has 1 descendant: SPATIAL_INTELLIGENCE (not itself)."""
    subtree = get_subtree(FeatureCategory.EMBODIED_AI)
    assert len(subtree) == 1
    assert FeatureCategory.SPATIAL_INTELLIGENCE in subtree
    assert FeatureCategory.EMBODIED_AI not in subtree  # root excluded


def test_get_subtree_unknown_empty() -> None:
    """UNKNOWN has no descendants."""
    assert get_subtree(FeatureCategory.UNKNOWN) == []


def test_is_leaf_code_agent_is_false() -> None:
    """CODE_AGENT is a pure aggregator, not a leaf."""
    assert is_leaf(FeatureCategory.CODE_AGENT) is False


def test_is_leaf_real_categories_are_true() -> None:
    """All actual feature categories (including EMBODIED_AI) are leaves."""
    for cat in FeatureCategory:
        if cat == FeatureCategory.CODE_AGENT:
            continue
        assert is_leaf(cat) is True, f"{cat.value} should be a leaf"
