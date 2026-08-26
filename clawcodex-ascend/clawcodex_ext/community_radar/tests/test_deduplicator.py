"""Tests for clawcodex_ext.community_radar.deduplicator."""

from __future__ import annotations

from clawcodex_ext.community_radar.deduplicator import (
    FeatureDeduplicator,
    default_scorer,
)
from clawcodex_ext.community_radar.models import FeatureRecord


def _record(source: str, title: str, description: str = "") -> FeatureRecord:
    return FeatureRecord(
        id=f"{source}-{title[:10]}",
        source=source,
        title=title,
        description=description,
    )


def test_empty_input_returns_empty() -> None:
    assert not FeatureDeduplicator().deduplicate([])


def test_single_record_passes_through() -> None:
    r = _record("aider", "Add lint auto-fix")
    out = FeatureDeduplicator().deduplicate([r])
    assert out == [r]


def test_dedup_merges_similar_records() -> None:
    r1 = _record("aider", "Add lint auto-fix mode", "Aider now supports lint auto-fix")
    r2 = _record("claude-code", "Lint auto-fix", "Claude Code adds lint auto-fix")
    out = FeatureDeduplicator(threshold=0.4).deduplicate([r1, r2])
    assert len(out) == 1
    canonical = out[0]
    # Only the merged (non-canonical) source lands in related_projects;
    # the canonical record never lists itself.
    assert canonical.related_projects == ["claude-code"]


def test_dedup_keeps_distinct_records() -> None:
    r1 = _record("aider", "Add lint auto-fix")
    r2 = _record("crewai", "Multi-agent crew routing")
    out = FeatureDeduplicator().deduplicate([r1, r2])
    assert len(out) == 2


def test_dedup_prefix_fallback() -> None:
    # TF-IDF can be degenerate if both titles share only a long prefix
    # and no body keywords. The prefix fallback should still merge them.
    r1 = _record("a", "Add ultra-specific long-feature-name goes here")
    r2 = _record("b", "Add ultra-specific long-feature-name also here")
    out = FeatureDeduplicator().deduplicate([r1, r2])
    assert len(out) == 1


def test_dedup_invalid_threshold_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        FeatureDeduplicator(threshold=1.5)


def test_dedup_combines_tags() -> None:
    r1 = FeatureRecord(
        id="r1",
        source="aider",
        title="Add cron task",
        description="Add cron task scheduling",
        tags=["cron"],
    )
    r2 = FeatureRecord(
        id="r2",
        source="claude-code",
        title="Add cron task scheduling",
        description="Claude Code cron tasks",
        tags=["scheduling"],
    )
    out = FeatureDeduplicator(threshold=0.4).deduplicate([r1, r2])
    assert len(out) == 1
    assert set(out[0].tags) >= {"cron", "scheduling"}


def test_default_scorer_returns_float_in_range() -> None:
    scorer = default_scorer()
    score = scorer(_record("a", "Add MCP tool"), _record("b", "Add MCP server"))
    assert 0.0 <= score <= 1.0
