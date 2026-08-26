"""Tests for clawcodex_ext.community_radar.extractor."""

from __future__ import annotations

from clawcodex_ext.community_radar.extractor import FeatureExtractor
from clawcodex_ext.community_radar.models import FeatureType, Release


def _release(body: str) -> Release:
    return Release(
        tag="v1.0.0",
        name="v1.0.0",
        body=body,
        published_at="2026-06-15T00:00:00Z",
        url="https://example.com/release",
    )


def test_extract_added_section() -> None:
    body = "# v1.2.3\n\n## Added\n- New `--lint` auto-fix mode\n- MCP server hot-reload\n"
    records = FeatureExtractor().extract(_release(body), source="aider")
    titles = [r.title for r in records]
    assert any("lint" in t.lower() for t in titles)
    assert any("mcp" in t.lower() for t in titles)
    assert all(r.feature_type == FeatureType.NEW for r in records)


def test_extract_breaking_section() -> None:
    body = "## Breaking Changes\n- StateGraph API has been refactored\n"
    records = FeatureExtractor().extract(_release(body), source="langgraph")
    assert records
    assert records[0].feature_type == FeatureType.BREAKING


def test_extract_checkbox_items() -> None:
    body = "## Added\n- [x] Add OpenTelemetry exporter\n- [x] Add hook for post-edit commands\n"
    records = FeatureExtractor().extract(_release(body), source="claude-code")
    assert len(records) == 2
    assert any("OpenTelemetry" in r.title for r in records)


def test_extract_ignores_empty_body() -> None:
    records = FeatureExtractor().extract(_release(""), source="x")
    assert records == []


def test_extract_unknown_sections_use_default_kind() -> None:
    body = "## Documentation\n- Documented new CLI flags\n"
    records = FeatureExtractor().extract(_release(body), source="x")
    # Unknown sections still produce records so the classifier can
    # decide what to do with them.
    assert records
    assert records[0].feature_type == FeatureType.NEW  # default kind


def test_extract_dedups_repeated_bullets() -> None:
    body = "## Added\n- Add MCP hot-reload\n- Add MCP hot-reload\n"
    records = FeatureExtractor().extract(_release(body), source="x")
    assert len(records) == 1


def test_extract_llm_hook_overrides_records() -> None:
    body = "## Added\n- Generic feature\n"

    def hook(records, body_text):  # type: ignore[no-untyped-def]
        for r in records:
            r.title = "[LLM] " + r.title
        return records

    records = FeatureExtractor(llm_hook=hook).extract(_release(body), source="x")
    assert records[0].title.startswith("[LLM]")


def test_extract_assigns_source_metadata() -> None:
    body = "## Added\n- A nice feature\n"
    records = FeatureExtractor().extract(_release(body), source="aider")
    assert records[0].source == "aider"
    assert records[0].released_at == "2026-06-15T00:00:00Z"
    assert records[0].url == "https://example.com/release"
