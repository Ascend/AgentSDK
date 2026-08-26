"""Tests for clawcodex_ext.community_radar.classifier."""

from __future__ import annotations

from clawcodex_ext.community_radar.classifier import FeatureClassifier
from clawcodex_ext.community_radar.models import FeatureCategory, FeatureRecord


def _record(title: str, description: str = "") -> FeatureRecord:
    return FeatureRecord(
        id=f"r-{title[:6]}",
        source="aider",
        title=title,
        description=description,
    )


def test_classify_agent_loop() -> None:
    record = _record("Self-correction loop", "Agent can now retry on failure")
    out = FeatureClassifier().classify(record)
    assert out.category == FeatureCategory.AGENT_LOOP
    assert "self_correction" in out.tags


def test_classify_memory() -> None:
    record = _record("Context collapse", "Auto-compact long histories")
    out = FeatureClassifier().classify(record)
    assert out.category == FeatureCategory.MEMORY


def test_classify_tool_system() -> None:
    record = _record("Add MCP tool", "Supports new MCP tool definition")
    out = FeatureClassifier().classify(record)
    assert out.category in {FeatureCategory.TOOL_SYSTEM, FeatureCategory.MCP}
    assert "mcp_extension" in out.tags or "mcp" in out.tags


def test_classify_observability() -> None:
    record = _record("Add telemetry pipeline", "Langfuse exporter for traces")
    out = FeatureClassifier().classify(record)
    assert out.category == FeatureCategory.OBSERVABILITY


def test_classify_permission() -> None:
    record = _record("YOLO mode toggle", "New approval gate for shell tools")
    out = FeatureClassifier().classify(record)
    assert out.category == FeatureCategory.PERMISSION


def test_classify_multi_agent() -> None:
    record = _record("Agent2Agent handoff", "subagent can transfer context")
    out = FeatureClassifier().classify(record)
    assert out.category == FeatureCategory.MULTI_AGENT


def test_classify_unknown_falls_back() -> None:
    record = _record("Generic changelog entry", "internal refactor")
    out = FeatureClassifier().classify(record)
    assert out.category in {FeatureCategory.UNKNOWN, FeatureCategory.INFRA}


def test_classify_roadmap_keywords_become_tags() -> None:
    record = _record("Improve cron scheduling", "more deterministic jitter")
    out = FeatureClassifier().classify(record)
    assert "cron" in out.tags


def test_classifier_llm_hook_refines_unknown() -> None:
    def hook(record: FeatureRecord) -> FeatureCategory:
        return FeatureCategory.OBSERVABILITY

    record = _record("Mystery feature", "no keywords match")
    out = FeatureClassifier(llm_hook=hook).classify(record)
    assert out.category == FeatureCategory.OBSERVABILITY


def test_classify_many_returns_list() -> None:
    records = [
        _record("Add telemetry"),
        _record("Add cron task"),
        _record("Add memory compact"),
    ]
    out = FeatureClassifier().classify_many(records)
    assert len(out) == 3
    assert all(r.category != FeatureCategory.UNKNOWN for r in out)
