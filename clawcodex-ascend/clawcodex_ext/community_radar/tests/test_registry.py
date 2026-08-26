# pylint: disable=use-implicit-booleaness-not-comparison
"""Tests for clawcodex_ext.community_radar.registry."""

from __future__ import annotations

import json
from pathlib import Path

from clawcodex_ext.community_radar.models import WatchSource
from clawcodex_ext.community_radar.registry import (
    DEFAULT_SOURCES,
    PHASE1_SOURCES,
    PHASE2_SOURCES,
    SourceRegistry,
    default_registry_path,
)


def test_with_defaults_seeds_phase1_projects() -> None:
    reg = SourceRegistry.with_defaults()
    names = {s.name for s in reg.list()}
    assert "claude-code" in names
    assert "aider" in names
    assert "swe-agent" in names
    assert "openhands" in names
    assert "autogen" in names
    assert "crewai" in names
    assert "langgraph" in names


def test_load_missing_file_yields_empty(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path / "sources.yaml")
    assert reg.load() == {}
    assert reg.list() == []


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    reg = SourceRegistry(path)
    reg.add(WatchSource.from_dict({"name": "demo", "repo": "foo/bar"}))
    reg.add(WatchSource.from_dict({"name": "demo2", "repo": "x/y", "track_commits": True}))
    reg.save()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert {item["name"] for item in payload} == {"demo", "demo2"}

    again = SourceRegistry(path)
    again.load()
    assert {s.name for s in again.list()} == {"demo", "demo2"}
    demo2 = again.get("demo2")
    assert demo2 is not None and demo2.track_commits is True


def test_remove_returns_bool(tmp_path: Path) -> None:
    reg = SourceRegistry.with_defaults(tmp_path / "sources.yaml")
    assert reg.remove("aider") is True
    assert reg.remove("does-not-exist") is False


def test_default_registry_path_respects_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    assert default_registry_path() == tmp_path / "community-radar" / "sources.yaml"


def test_default_sources_shape() -> None:
    # Defensive: every default entry must parse cleanly.
    for item in DEFAULT_SOURCES:
        WatchSource.from_dict(item)


def test_phase1_sources_shape() -> None:
    for item in PHASE1_SOURCES:
        WatchSource.from_dict(item)
    # Exactly 7 Phase-1 sources.
    assert len(PHASE1_SOURCES) == 7


def test_phase2_sources_shape() -> None:
    names = set()
    for item in PHASE2_SOURCES:
        WatchSource.from_dict(item)
        names.add(item["name"])
    # Phase 2 should cover the named projects.
    for required in ("cline", "continue", "goose", "openclaw"):
        assert required in names, f"missing Phase 2 source: {required}"


def test_with_defaults_include_phase2() -> None:
    reg = SourceRegistry.with_defaults(include_phase2=True)
    names = {s.name for s in reg.list()}
    # Phase 1 names still present.
    assert "claude-code" in names
    assert "langgraph" in names
    # Phase 2 names added.
    assert "cline" in names
    assert "continue" in names
    assert "goose" in names


def test_with_defaults_phase2_off_by_default() -> None:
    reg = SourceRegistry.with_defaults()
    names = {s.name for s in reg.list()}
    # Backward-compat: the default constructor still yields Phase 1 only.
    assert "cline" not in names
    assert "goose" not in names
    # And matches DEFAULT_SOURCES exactly.
    assert len(reg) == len(DEFAULT_SOURCES)
