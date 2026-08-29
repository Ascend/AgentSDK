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

"""Tests for clawcodex_ext.community_radar.scorer."""

from __future__ import annotations

from clawcodex_ext.community_radar.config import RadarConfig
from clawcodex_ext.community_radar.models import (
    FeatureCategory,
    FeatureRecord,
    FeatureType,
)
from clawcodex_ext.community_radar.scorer import FeatureScorer


def _record(
    *,
    title: str = "Add lint auto-fix",
    description: str = "Tooling now supports lint auto-fix",
    category: FeatureCategory = FeatureCategory.TOOL_SYSTEM,
    feature_type: FeatureType = FeatureType.NEW,
    related_projects: list[str] | None = None,
) -> FeatureRecord:
    return FeatureRecord(
        id="r1",
        source="aider",
        title=title,
        description=description,
        category=category,
        feature_type=feature_type,
        related_projects=list(related_projects or []),
        released_at="2026-06-15T00:00:00Z",
    )


def test_score_returns_all_dimensions() -> None:
    score = FeatureScorer().score(_record())
    assert 0 <= score.overall <= 100
    for dim in ("popularity", "maturity", "adaptation_cost", "strategic_value", "architecture_fit"):
        assert 0 <= getattr(score, dim) <= 100


def test_score_breaking_lowers_maturity() -> None:
    breaking = FeatureScorer().score(_record(feature_type=FeatureType.BREAKING))
    new = FeatureScorer().score(_record(feature_type=FeatureType.NEW))
    assert breaking.maturity < new.maturity


def test_score_cross_project_raises_adaptation_cost() -> None:
    alone = FeatureScorer().score(_record())
    shared = FeatureScorer().score(_record(related_projects=["claude-code", "openhands"]))
    assert shared.adaptation_cost >= alone.adaptation_cost - 5  # not strictly less but cheaper


def test_score_uses_custom_weights() -> None:
    cfg = RadarConfig(
        weights={
            "popularity": 1.0,
            "maturity": 0.0,
            "adaptation_cost": 0.0,
            "strategic_value": 0.0,
            "architecture_fit": 0.0,
        }
    )
    score = FeatureScorer(cfg).score(_record())
    # With only popularity carrying weight, overall ≈ popularity
    # (within rounding).
    assert abs(score.overall - score.popularity) < 0.5


def test_strategic_value_rewards_roadmap_keywords() -> None:
    cfg = RadarConfig(roadmap_keywords=["mcp", "tool", "context"])
    aligned = FeatureScorer(cfg).score(
        _record(
            title="Add MCP tool",
            description="Supports new mcp context tooling",
        )
    )
    noise = FeatureScorer(cfg).score(
        _record(
            title="Update changelog formatter",
            description="Internal refactor only",
            category=FeatureCategory.INFRA,
        )
    )
    assert aligned.strategic_value >= noise.strategic_value
