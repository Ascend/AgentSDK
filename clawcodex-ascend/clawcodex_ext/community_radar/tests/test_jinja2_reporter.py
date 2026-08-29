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

"""Tests for Jinja2 rendering in clawcodex_ext.community_radar.reporter (Phase 3)."""

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
    _render_jinja_markdown,
    _resolve_template_dir,
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
        id="r1",
        source=source,
        title=title,
        description=description,
        category=category,
        feature_type=feature_type,
        related_projects=list(related or []),
        released_at="2026-06-15T00:00:00Z",
    )


def _digest(
    *,
    period: str = "weekly",
    records: list[FeatureRecord] | None = None,
    breaking: list[FeatureRecord] | None = None,
    summary: str = "本期内容。",
) -> CommunityDigest:
    records = list(records or [_record()])
    scored = [
        ScoredFeature(
            record=r,
            score=FeatureScore(
                record_id=r.id,
                overall=70.0,
                popularity=80.0,
                maturity=70.0,
                adaptation_cost=65.0,
                strategic_value=75.0,
                architecture_fit=70.0,
            ),
        )
        for r in records
    ]
    return CommunityDigest(
        period=period,
        generated_at=utc_now_iso(),
        summary=summary,
        new_features=records,
        trending=scored,
        breaking_changes=list(breaking or []),
        stats=DigestStats(
            total_versions=1,
            total_features=len(records),
            by_category={r.category.value: 1 for r in records},
        ),
        sources_used=["aider"],
    )


# ---------------------------------------------------------------------------
# _resolve_template_dir
# ---------------------------------------------------------------------------


def test_resolve_template_dir_finds_default(tmp_path: Path) -> None:
    found = _resolve_template_dir(None)
    # The package ships with the templates/ directory.
    assert found is not None
    assert (found / "weekly_digest.md.j2").exists()


def test_resolve_template_dir_explicit_wins(tmp_path: Path) -> None:
    custom = tmp_path / "templates"
    custom.mkdir()
    (custom / "weekly_digest.md.j2").write_text("# {{ title }}\n", encoding="utf-8")
    found = _resolve_template_dir(custom)
    assert found == custom


def test_resolve_template_dir_falls_back_to_package_default(tmp_path: Path) -> None:
    # When the explicit directory is missing, the package's default
    # templates dir wins so callers always have *some* template set.
    found = _resolve_template_dir(tmp_path / "nope")
    assert found is not None
    assert (found / "weekly_digest.md.j2").exists()


# ---------------------------------------------------------------------------
# _render_jinja_markdown
# ---------------------------------------------------------------------------


def test_render_jinja_markdown_weekly(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_dir.joinpath("weekly_digest.md.j2").write_text(
        "# {{ title }} ({{ period_label }})\n"
        "生成时间: {{ generated_at }}\n"
        "摘要: {{ summary }}\n"
        "{% for item in trending %}- {{ item.title }} | {{ item.category }}\n{% endfor %}\n"
        "{% if errors %}错误: {{ errors | length }}{% endif %}\n",
        encoding="utf-8",
    )
    digest = _digest(summary="hello")
    rendered = _render_jinja_markdown(digest, template_dir)
    assert rendered is not None
    assert "ClawCodex 社区动态报告 (周报)" in rendered
    assert "摘要: hello" in rendered
    assert "- Sample | tool_system" in rendered


def test_render_jinja_markdown_monthly(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_dir.joinpath("monthly_digest.md.j2").write_text(
        "# monthly {{ period_label }}\n",
        encoding="utf-8",
    )
    digest = _digest(period="monthly")
    rendered = _render_jinja_markdown(digest, template_dir)
    assert rendered is not None
    assert "monthly 月报" in rendered


def test_render_jinja_markdown_returns_none_when_missing(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    digest = _digest(period="weekly")
    # No template files ⇒ None (caller falls back to inline renderer).
    assert _render_jinja_markdown(digest, template_dir) is None


def test_render_jinja_markdown_strict_undefined_missing_var(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_dir.joinpath("weekly_digest.md.j2").write_text(
        "{{ undefined_var }}",
        encoding="utf-8",
    )
    digest = _digest()
    # StrictUndefined should fail and the helper should fall back to None.
    assert _render_jinja_markdown(digest, template_dir) is None


# ---------------------------------------------------------------------------
# CommunityReporter.write — Phase 3 wiring
# ---------------------------------------------------------------------------


def test_reporter_write_uses_jinja_when_available(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_dir.joinpath("weekly_digest.md.j2").write_text(
        "# CUSTOM {{ title }} {{ summary }}",
        encoding="utf-8",
    )
    reporter = CommunityReporter(
        RadarConfig(max_features_per_report=5),
        template_dir=template_dir,
    )
    digest = _digest(summary="abc")
    result = reporter.write(digest, tmp_path)
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "CUSTOM ClawCodex 社区动态报告 abc" in text


def test_reporter_write_falls_back_to_inline_when_no_template(tmp_path: Path) -> None:
    reporter = CommunityReporter(
        RadarConfig(max_features_per_report=5),
        template_dir=tmp_path / "missing",
    )
    digest = _digest(summary="plain")
    result = reporter.write(digest, tmp_path)
    text = result.markdown_path.read_text(encoding="utf-8")
    # Inline renderer heading is preserved as fallback marker.
    assert "# ClawCodex 社区动态报告" in text
    assert "plain" in text


def test_reporter_write_includes_llm_marker(tmp_path: Path) -> None:
    """Templates may include a conditional block for LLM-assisted digests."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_dir.joinpath("weekly_digest.md.j2").write_text(
        "[regular]\n{% if llm_assisted %}-- LLM 标记 --\n{% endif %}",
        encoding="utf-8",
    )
    reporter = CommunityReporter(
        RadarConfig(max_features_per_report=5),
        template_dir=template_dir,
    )
    digest = _digest(summary="本期内容 (LLM-assisted)")
    result = reporter.write(digest, tmp_path)
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "-- LLM 标记 --" in text


def test_reporter_default_uses_shipped_templates(tmp_path: Path) -> None:
    """Without explicit template_dir, the package's templates/ is used."""
    reporter = CommunityReporter(RadarConfig(max_features_per_report=5))
    digest = _digest()
    result = reporter.write(digest, tmp_path)
    text = result.markdown_path.read_text(encoding="utf-8")
    # Shipped template renders the trending table.
    assert "| Sample |" in text or "Sample" in text
