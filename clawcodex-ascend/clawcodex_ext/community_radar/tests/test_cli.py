"""Tests for clawcodex_ext.community_radar.cli."""

from __future__ import annotations

import json
from pathlib import Path

from clawcodex_ext.community_radar.cli import run
from clawcodex_ext.community_radar.models import (
    FetchResult,
    Release,
)


class _FakeFetcher:
    def __init__(self, results: list[FetchResult]) -> None:
        self._results = results
        self.closed = False

    def fetch(self, source, *, incremental: bool = False, since: str | None = None):  # type: ignore[no-untyped-def]
        return next((r for r in self._results if r.source == source.name), FetchResult(source=source.name))

    def fetch_all(self, sources, *, incremental: bool = False):  # type: ignore[no-untyped-def]
        return list(self._results)

    def close(self) -> None:
        self.closed = True


def _release(body: str) -> Release:
    return Release(
        tag="v1.0.0",
        name="v1.0.0",
        body=body,
        published_at="2026-06-15T00:00:00Z",
        url="https://example.com/r",
    )


def _patch_registry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))


def test_help_prints_usage(capsys) -> None:  # type: ignore[no-untyped-def]
    code = run(["help"])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "community-radar" in out


def test_source_list_seeds_defaults(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(["source", "list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "aider" in out


def test_source_add_then_remove(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(
        [
            "source",
            "add",
            "demo",
            "--repo",
            "foo/bar",
            "--notes",
            "test",
            "--roadmap-keyword",
            "demo",
            "--roadmap-keyword",
            "test",
        ]
    )
    assert code == 0
    capsys.readouterr()  # discard "Added source ..." output

    code = run(["source", "show", "demo"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "demo"
    assert payload["roadmap_keywords"] == ["demo", "test"]

    code = run(["source", "remove", "demo"])
    assert code == 0
    capsys.readouterr()  # discard "Removed source ..." output

    code = run(["source", "show", "demo"])
    assert code == 1


def test_source_add_validation(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(["source", "add", "bad", "--repo", "missing-slash"])
    assert code == 2


def test_status_prints_paths(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(["status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Registry path:" in out
    assert "Config path:" in out


def test_config_init_writes_default(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(["config", "init"])
    assert code == 0
    config_path = tmp_path / "community-radar" / "config.yaml"
    # When yaml is unavailable the CLI falls back to .json; accept either.
    assert config_path.exists() or (tmp_path / "community-radar" / "config.json").exists()


def test_config_show_uses_defaults(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    code = run(["config", "show"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["max_features_per_report"] >= 1


def test_scan_invokes_pipeline(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _patch_registry(monkeypatch, tmp_path)
    # Replace the fetcher the pipeline builds with our fake.
    release = _release("## Added\n- New telemetry pipeline\n")
    fake_results = [FetchResult(source="claude-code", releases=[release])]

    from clawcodex_ext.community_radar import pipeline as pipeline_module

    original_init = pipeline_module.CommunityRadarPipeline.__init__

    def patched_init(self, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, **kwargs)
        self.fetcher = _FakeFetcher(fake_results)  # type: ignore[assignment]
        self._owns_fetcher = True

    monkeypatch.setattr(
        pipeline_module.CommunityRadarPipeline,
        "__init__",
        patched_init,
    )

    code = run(
        [
            "scan",
            "--period",
            "weekly",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Scan complete" in out
    md_files = list((tmp_path / "out").glob("*.md"))
    assert md_files, "scan should have written a markdown digest"
