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
"""Tests for clawcodex_ext.community_radar.reporter.render_proposals (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.community_radar.config import RadarConfig
from clawcodex_ext.community_radar.models import (
    CommunityDigest,
    DigestStats,
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    ScoredFeature,
    utc_now_iso,
)
from clawcodex_ext.community_radar.reporter import (
    CommunityReporter,
    _candidate_action,
    render_proposals,
)


def _record(
    *,
    title: str = "Sample",
    description: str = "A description",
    source: str = "aider",
    category: FeatureCategory = FeatureCategory.TOOL_SYSTEM,
    feature_type: FeatureType = FeatureType.NEW,
    related: list[str] | None = None,
) -> FeatureRecord:
    return FeatureRecord(
        id=f"{source}|{title}|NEW",
        source=source,
        title=title,
        description=description,
        category=category,
        feature_type=feature_type,
        related_projects=list(related or []),
        released_at="2026-06-15T00:00:00Z",
        url=f"https://example.com/{source}",
        tags=["tool", "ci"],
    )


def _score(
    record_id: str,
    *,
    overall: float = 70.0,
    strategic_value: float = 75.0,
    architecture_fit: float = 70.0,
    popularity: float = 80.0,
    maturity: float = 70.0,
    adaptation_cost: float = 65.0,
) -> FeatureScore:
    return FeatureScore(
        record_id=record_id,
        overall=overall,
        popularity=popularity,
        maturity=maturity,
        adaptation_cost=adaptation_cost,
        strategic_value=strategic_value,
        architecture_fit=architecture_fit,
    )


def test_render_proposals_basic_shape() -> None:
    record = _record()
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="ok",
        new_features=[record],
        trending=[ScoredFeature(record=record, score=_score(record.id))],
        breaking_changes=[],
        stats=DigestStats(total_versions=1, total_features=1),
        sources_used=["aider"],
    )
    payload = render_proposals(digest)
    assert payload["schema_version"] == "1.0"
    assert payload["period"] == "weekly"
    assert payload["generated_at"] == digest.generated_at
    assert len(payload["proposals"]) == 1
    proposal = payload["proposals"][0]
    assert proposal["id"] == record.id
    assert proposal["title"] == "Sample"
    assert proposal["category"] == "tool_system"
    assert proposal["feature_type"] == "new"
    assert proposal["source_projects"] == ["aider"]
    assert proposal["candidate_action"] in {"adopt", "observe", "skip"}
    assert proposal["score"]["overall"] == 70.0
    assert proposal["tags"] == ["tool", "ci"]
    assert proposal["url"] == "https://example.com/aider"


def test_render_proposals_empty_digest() -> None:
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="empty",
        new_features=[],
        trending=[],
        breaking_changes=[],
        stats=DigestStats(total_versions=0, total_features=0),
        sources_used=[],
    )
    payload = render_proposals(digest)
    assert payload["proposals"] == []
    assert payload["schema_version"] == "1.0"


def test_candidate_action_adopt_when_high_alignment() -> None:
    record = _record()
    score = _score(record.id, strategic_value=80.0, architecture_fit=80.0)
    assert _candidate_action(record, score) == "adopt"


def test_candidate_action_observe_when_decent() -> None:
    record = _record()
    score = _score(record.id, overall=70.0, strategic_value=40.0, architecture_fit=40.0)
    assert _candidate_action(record, score) == "observe"


def test_candidate_action_skip_when_breaking_and_low() -> None:
    record = _record(feature_type=FeatureType.BREAKING)
    score = _score(record.id, strategic_value=20.0, architecture_fit=30.0)
    assert _candidate_action(record, score) == "skip"


def test_candidate_action_skip_when_low_overall() -> None:
    record = _record()
    # Override strategic_value/architecture_fit so the high-alignment
    # branch doesn't fire — only overall is dragged below 60.
    score = _score(
        record.id,
        overall=20.0,
        strategic_value=10.0,
        architecture_fit=10.0,
    )
    assert _candidate_action(record, score) == "skip"


def test_render_proposals_includes_related_projects() -> None:
    record = _record(related=["claude-code", "openclaw"])
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="ok",
        new_features=[record],
        trending=[ScoredFeature(record=record, score=_score(record.id))],
        breaking_changes=[],
        stats=DigestStats(total_versions=1, total_features=1),
        sources_used=["aider"],
    )
    payload = render_proposals(digest)
    proposal = payload["proposals"][0]
    assert set(proposal["source_projects"]) == {"aider", "claude-code", "openclaw"}


def test_reporter_write_emits_proposals_file(tmp_path: Path) -> None:
    record = _record()
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = reporter.build_digest(
        period="weekly",
        features=[record],
        scored=[ScoredFeature(record=record, score=_score(record.id))],
        sources_used=["aider"],
        versions_total=1,
        summary="x",
    )
    result = reporter.write(digest, tmp_path)
    assert result.proposals_path is not None
    assert result.proposals_path.exists()
    import json

    payload = json.loads(result.proposals_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert len(payload["proposals"]) == 1


def test_reporter_write_can_skip_proposals(tmp_path: Path) -> None:
    record = _record()
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = reporter.build_digest(
        period="weekly",
        features=[record],
        scored=[ScoredFeature(record=record, score=_score(record.id))],
        sources_used=["aider"],
        versions_total=1,
    )
    result = reporter.write(digest, tmp_path, write_proposals=False)
    assert result.proposals_path is None
