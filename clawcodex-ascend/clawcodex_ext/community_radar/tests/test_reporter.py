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

"""Tests for clawcodex_ext.community_radar.reporter."""

from __future__ import annotations

import json
from pathlib import Path

from clawcodex_ext.community_radar.config import RadarConfig
from clawcodex_ext.community_radar.models import (
    CommunityDigest,
    DigestStats,
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    HistoryComparison,
    ScoredFeature,
    make_feature_id,
    utc_now_iso,
)
from clawcodex_ext.community_radar.reporter import (
    CommunityReporter,
    _load_digest_from_json,
    _render_comparison_section,
    _render_inline_markdown,
    compare_digests,
    find_previous_digest_path,
)


def _record(
    *,
    title: str = "Sample",
    description: str = "A description",
    source: str = "aider",
    category: FeatureCategory = FeatureCategory.TOOL_SYSTEM,
    feature_type: FeatureType = FeatureType.NEW,
    related: list[str] | None = None,
    rid: str | None = None,
) -> FeatureRecord:
    return FeatureRecord(
        id=rid or make_feature_id(source, title, feature_type.value),
        source=source,
        title=title,
        description=description,
        category=category,
        feature_type=feature_type,
        related_projects=list(related or []),
        released_at="2026-06-15T00:00:00Z",
    )


def test_render_markdown_includes_table() -> None:
    record = _record()
    score = FeatureScore(
        record_id=record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="本期内容。",
        new_features=[record],
        trending=[ScoredFeature(record=record, score=score)],
        breaking_changes=[],
        stats=DigestStats(total_versions=1, total_features=1, by_category={"tool_system": 1}),
        sources_used=["aider"],
    )
    md = _render_inline_markdown(digest)
    assert "# ClawCodex 社区动态报告" in md
    assert "| Sample" in md
    assert "tool_system" in md
    assert "本期内容。" in md


def test_reporter_write_produces_md_and_json(tmp_path: Path) -> None:
    record = _record(title="Write me", description="writes")
    score = FeatureScore(
        record_id=record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = reporter.build_digest(
        period="weekly",
        features=[record],
        scored=[ScoredFeature(record=record, score=score)],
        sources_used=["aider"],
        versions_total=1,
        summary="hello",
    )
    write_result = reporter.write(digest, tmp_path)
    assert write_result.markdown_path.exists()
    assert write_result.json_path.exists()

    payload = json.loads(write_result.json_path.read_text(encoding="utf-8"))
    assert payload["period"] == "weekly"
    assert payload["stats"]["total_features"] == 1
    assert payload["trending"][0]["record"]["title"] == "Write me"


def test_reporter_handles_empty_input(tmp_path: Path) -> None:
    reporter = CommunityReporter(RadarConfig())
    digest = reporter.build_digest(
        period="monthly",
        features=[],
        scored=[],
        sources_used=[],
        versions_total=0,
    )
    assert digest.stats.total_features == 0
    md = _render_inline_markdown(digest)
    assert "（无）" in md


def test_breaking_changes_surface_in_digest(tmp_path: Path) -> None:
    breaking = _record(
        title="StateGraph refactor",
        description="API refactor",
        feature_type=FeatureType.BREAKING,
        source="langgraph",
    )
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="breaking",
        new_features=[breaking],
        trending=[
            ScoredFeature(
                record=breaking,
                score=FeatureScore(
                    record_id=breaking.id,
                    overall=50.0,
                    popularity=50.0,
                    maturity=30.0,
                    adaptation_cost=40.0,
                    strategic_value=50.0,
                    architecture_fit=50.0,
                ),
            )
        ],
        breaking_changes=[breaking],
        stats=DigestStats(total_versions=1, total_features=1),
        sources_used=["langgraph"],
    )
    md = _render_inline_markdown(digest)
    assert "## 破坏性变更预警" in md
    assert "StateGraph refactor" in md


# ---------------------------------------------------------------------------
# History comparison tests
# ---------------------------------------------------------------------------


def _make_digest(
    *,
    period: str = "weekly",
    features: list[FeatureRecord] | None = None,
    scored: list[ScoredFeature] | None = None,
) -> CommunityDigest:
    features = features or []
    scored = scored or []
    return CommunityDigest(
        period=period,
        generated_at=utc_now_iso(),
        summary="test digest",
        new_features=features,
        trending=scored,
        stats=DigestStats(
            total_versions=1,
            total_features=len(features),
        ),
    )


def test_find_previous_digest_path_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert find_previous_digest_path(tmp_path, "weekly", "current-stem") is None


def test_find_previous_digest_path_finds_prior_file(tmp_path: Path) -> None:
    (tmp_path / "community-digest-weekly-20260622T080000Z.json").write_text("{}")
    (tmp_path / "community-digest-weekly-20260629T080000Z.json").write_text("{}")
    result = find_previous_digest_path(tmp_path, "weekly", "community-digest-weekly-20260629T080000Z")
    assert result is not None
    assert result.stem == "community-digest-weekly-20260622T080000Z"


def test_find_previous_digest_path_skips_current_stem(tmp_path: Path) -> None:
    (tmp_path / "community-digest-weekly-20260629T080000Z.json").write_text("{}")
    result = find_previous_digest_path(tmp_path, "weekly", "community-digest-weekly-20260629T080000Z")
    assert result is None


def test_load_digest_from_json_round_trips(tmp_path: Path) -> None:
    record = _record(title="Test", source="aider")
    score = FeatureScore(
        record_id=record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    digest = CommunityDigest(
        period="weekly",
        generated_at="2026-06-29T08:00:00Z",
        summary="round trip",
        new_features=[record],
        trending=[ScoredFeature(record=record, score=score)],
        stats=DigestStats(total_versions=1, total_features=1),
    )
    json_path = tmp_path / "test.json"
    json_path.write_text(
        json.dumps(digest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = _load_digest_from_json(json_path)
    assert loaded is not None
    assert loaded.period == "weekly"
    assert loaded.summary == "round trip"
    assert len(loaded.new_features) == 1
    assert loaded.new_features[0].title == "Test"


def test_load_digest_from_json_returns_none_for_bad_file(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert _load_digest_from_json(tmp_path / "bad.json") is None


def test_load_digest_from_json_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert _load_digest_from_json(tmp_path / "missing.json") is None


def test_compare_digests_detects_new_features() -> None:
    prev = _make_digest(features=[_record(title="old feature", source="aider")])
    curr = _make_digest(
        features=[
            _record(title="old feature", source="aider"),
            _record(title="new feature", source="claude-code"),
        ]
    )
    comparison = compare_digests(prev, curr)
    assert comparison.new_count == 1
    assert comparison.disappeared_count == 0
    assert len(comparison.new_features) == 1
    assert comparison.new_features[0].title == "new feature"


def test_compare_digests_detects_disappeared_features() -> None:
    prev = _make_digest(
        features=[
            _record(title="old feature", source="aider"),
            _record(title="gone feature", source="langgraph"),
        ]
    )
    curr = _make_digest(
        features=[
            _record(title="old feature", source="aider"),
        ]
    )
    comparison = compare_digests(prev, curr)
    assert comparison.new_count == 0
    assert comparison.disappeared_count == 1
    assert len(comparison.disappeared_features) == 1
    assert comparison.disappeared_features[0].title == "gone feature"


def test_compare_digests_detects_score_changes() -> None:
    record1 = _record(title="same feature", source="aider")
    record2 = _record(title="same feature", source="aider")
    prev = _make_digest(
        scored=[
            ScoredFeature(
                record=record1,
                score=FeatureScore(
                    record_id=record1.id,
                    overall=50.0,
                    popularity=50.0,
                    maturity=50.0,
                    adaptation_cost=50.0,
                    strategic_value=50.0,
                    architecture_fit=50.0,
                ),
            ),
        ]
    )
    curr = _make_digest(
        scored=[
            ScoredFeature(
                record=record2,
                score=FeatureScore(
                    record_id=record2.id,
                    overall=72.0,
                    popularity=80.0,
                    maturity=70.0,
                    adaptation_cost=65.0,
                    strategic_value=75.0,
                    architecture_fit=70.0,
                ),
            ),
        ]
    )
    comparison = compare_digests(prev, curr)
    assert len(comparison.score_changed) == 1
    assert comparison.score_changed[0]["old_score"] == 50.0
    assert comparison.score_changed[0]["new_score"] == 72.0
    assert comparison.score_changed[0]["delta"] == 22.0


def test_compare_digests_ignores_small_score_deltas() -> None:
    record1 = _record(title="same feature", source="aider")
    record2 = _record(title="same feature", source="aider")
    prev = _make_digest(
        scored=[
            ScoredFeature(
                record=record1,
                score=FeatureScore(
                    record_id=record1.id,
                    overall=50.0,
                    popularity=50.0,
                    maturity=50.0,
                    adaptation_cost=50.0,
                    strategic_value=50.0,
                    architecture_fit=50.0,
                ),
            ),
        ]
    )
    curr = _make_digest(
        scored=[
            ScoredFeature(
                record=record2,
                score=FeatureScore(
                    record_id=record2.id,
                    overall=52.0,
                    popularity=50.0,
                    maturity=50.0,
                    adaptation_cost=50.0,
                    strategic_value=50.0,
                    architecture_fit=55.0,
                ),
            ),
        ]
    )
    comparison = compare_digests(prev, curr, score_delta_threshold=5.0)
    assert len(comparison.score_changed) == 0


def test_render_comparison_section_includes_new_and_disappeared() -> None:
    comparison = HistoryComparison(
        previous_period="weekly",
        previous_generated_at="2026-06-22T08:00:00Z",
        previous_stem="prev-stem",
        new_count=1,
        disappeared_count=1,
        score_changed=[],
        new_features=[_record(title="New Thing", source="aider")],
        disappeared_features=[_record(title="Old Thing", source="claude-code")],
    )
    md = _render_comparison_section(comparison)
    assert "## 变化对比 (vs 上期)" in md
    assert "### 新增特性 (1)" in md
    assert "New Thing" in md
    assert "### 消失特性 (1)" in md
    assert "Old Thing" in md
    assert "### 评分变化 (0)" in md


def test_render_comparison_section_includes_score_changes() -> None:
    comparison = HistoryComparison(
        previous_period="weekly",
        previous_generated_at="2026-06-22T08:00:00Z",
        previous_stem="prev-stem",
        new_count=0,
        disappeared_count=0,
        score_changed=[
            {
                "id": "abc",
                "title": "Evolving Feature",
                "old_score": 60.0,
                "new_score": 75.0,
                "delta": 15.0,
            },
        ],
    )
    md = _render_comparison_section(comparison)
    assert "Evolving Feature" in md
    assert "60.0" in md
    assert "75.0" in md
    assert "↑" in md
    assert "15.0" in md


def test_render_inline_markdown_includes_comparison() -> None:
    record = _record(title="Current Feature", source="aider")
    score = FeatureScore(
        record_id=record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    digest = CommunityDigest(
        period="weekly",
        generated_at=utc_now_iso(),
        summary="test",
        new_features=[record],
        trending=[ScoredFeature(record=record, score=score)],
        stats=DigestStats(total_versions=1, total_features=1),
        sources_used=["aider"],
    )
    comparison = HistoryComparison(
        previous_period="weekly",
        previous_generated_at="2026-06-22T08:00:00Z",
        previous_stem="prev",
        new_count=1,
        disappeared_count=0,
        score_changed=[],
        new_features=[_record(title="New Feature", source="claude-code")],
    )
    md = _render_inline_markdown(digest, comparison=comparison)
    assert "## 变化对比 (vs 上期)" in md
    assert "New Feature" in md
    assert "Current Feature" in md  # still has original content


def test_reporter_write_with_compare(tmp_path: Path) -> None:
    # Write a previous digest
    prev_record = _record(title="Prev Feature", source="aider")
    prev_score = FeatureScore(
        record_id=prev_record.id,
        overall=65.0,
        popularity=70.0,
        maturity=60.0,
        adaptation_cost=60.0,
        strategic_value=65.0,
        architecture_fit=65.0,
    )
    prev_reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    prev_digest = prev_reporter.build_digest(
        period="weekly",
        features=[prev_record],
        scored=[ScoredFeature(record=prev_record, score=prev_score)],
        sources_used=["aider"],
        versions_total=1,
        generated_at="2026-06-22T08:00:00Z",
    )
    prev_reporter.write(prev_digest, tmp_path)

    # Now write a new digest with compare=True
    curr_record = _record(title="Current Feature", source="aider")
    curr_score = FeatureScore(
        record_id=curr_record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = reporter.build_digest(
        period="weekly",
        features=[curr_record],
        scored=[ScoredFeature(record=curr_record, score=curr_score)],
        sources_used=["aider"],
        versions_total=1,
        generated_at="2026-06-29T08:00:00Z",
    )
    write_result = reporter.write(digest, tmp_path, compare=True)
    md_text = write_result.markdown_path.read_text(encoding="utf-8")
    assert "## 变化对比 (vs 上期)" in md_text


def test_reporter_write_with_compare_no_previous(tmp_path: Path) -> None:
    """When there is no previous digest, compare=True should not crash."""
    record = _record(title="Only Feature", source="aider")
    score = FeatureScore(
        record_id=record.id,
        overall=72.0,
        popularity=80.0,
        maturity=70.0,
        adaptation_cost=65.0,
        strategic_value=75.0,
        architecture_fit=70.0,
    )
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = reporter.build_digest(
        period="weekly",
        features=[record],
        scored=[ScoredFeature(record=record, score=score)],
        sources_used=["aider"],
        versions_total=1,
    )
    write_result = reporter.write(digest, tmp_path, compare=True)
    md_text = write_result.markdown_path.read_text(encoding="utf-8")
    # Should still produce a valid digest, just without comparison section
    assert "# ClawCodex 社区动态报告" in md_text
    assert "Only Feature" in md_text
    # No comparison section since there's no previous digest
    assert "## 变化对比" not in md_text
