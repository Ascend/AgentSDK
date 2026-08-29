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
"""Tests for clawcodex_ext.community_radar.pipeline."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.community_radar.config import RadarConfig
from clawcodex_ext.community_radar.models import (
    FetchResult,
    Release,
    WatchSource,
)
from clawcodex_ext.community_radar.pipeline import CommunityRadarPipeline
from clawcodex_ext.community_radar.registry import SourceRegistry


class _FakeFetcher:
    def __init__(self, results: list[FetchResult]) -> None:
        self._results = results
        self.closed = False

    def fetch(self, source, *, incremental: bool = False, since: str | None = None):  # type: ignore[no-untyped-def]
        return self._results[0] if self._results else FetchResult(source=source.name)

    def fetch_all(self, sources, *, incremental: bool = False, since: str | None = None):  # type: ignore[no-untyped-def]
        return list(self._results)

    def close(self) -> None:
        self.closed = True


def _release(body: str) -> Release:
    return Release(
        tag="v1.0.0",
        name="v1.0.0",
        body=body,
        published_at="2026-06-15T00:00:00Z",
        url="https://example.com/r1",
    )


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    releases = [_release("## Added\n- Add lint auto-fix\n- Add MCP server hot-reload\n")]
    fetcher = _FakeFetcher(
        [
            FetchResult(source="aider", releases=releases),
        ]
    )

    registry_path = tmp_path / "sources.yaml"
    registry = SourceRegistry.with_defaults(registry_path)

    output_dir = tmp_path / "out"
    pipeline = CommunityRadarPipeline(
        config=RadarConfig(output_dir=str(output_dir)),
        registry=registry,
        fetcher=fetcher,  # type: ignore[arg-type]
    )
    result = pipeline.run_scan(period="weekly", write=True)

    assert result.digest.stats.total_features >= 1
    assert result.write_result is not None
    assert result.write_result.markdown_path.exists()
    assert result.write_result.json_path.exists()
    # Fetcher.close must run even when the pipeline owns it; here we
    # pass our own so we check the contract differently:
    assert fetcher.closed is False


def test_pipeline_swallows_fetcher_errors(tmp_path: Path) -> None:
    fetcher = _FakeFetcher(
        [
            FetchResult(
                source="aider",
                releases=[_release("## Added\n- Some feature\n")],
                errors=["network down"],
            ),
        ]
    )
    registry = SourceRegistry.with_defaults(tmp_path / "sources.yaml")
    pipeline = CommunityRadarPipeline(
        config=RadarConfig(output_dir=str(tmp_path / "out")),
        registry=registry,
        fetcher=fetcher,  # type: ignore[arg-type]
    )
    result = pipeline.run_scan(period="weekly", write=False)
    assert any("network down" in e for e in result.digest.errors)


def test_pipeline_uses_explicit_sources(tmp_path: Path) -> None:
    fetcher = _FakeFetcher([])
    pipeline = CommunityRadarPipeline(
        config=RadarConfig(output_dir=str(tmp_path / "out")),
        registry=None,
        fetcher=fetcher,  # type: ignore[arg-type]
    )
    sources = [WatchSource.from_dict({"name": "demo", "repo": "foo/bar"})]
    result = pipeline.run_scan(
        period="weekly",
        write=False,
        sources=sources,
    )
    assert result.digest.sources_used == ["demo"]


def test_pipeline_no_sources(tmp_path: Path) -> None:
    pipeline = CommunityRadarPipeline(
        config=RadarConfig(output_dir=str(tmp_path / "out")),
        registry=None,
        fetcher=_FakeFetcher([]),  # type: ignore[arg-type]
    )
    result = pipeline.run_scan(period="weekly", write=False, sources=[])
    assert result.digest.sources_used == []
    assert result.digest.stats.total_features == 0


def test_pipeline_owns_fetcher_closes_it(tmp_path: Path) -> None:
    class _AutoCloseFetcher(_FakeFetcher):
        def __init__(self) -> None:
            super().__init__([])
            self.close_called = False

        def close(self) -> None:
            self.close_called = True

    fake = _AutoCloseFetcher()
    pipeline = CommunityRadarPipeline(
        config=RadarConfig(output_dir=str(tmp_path / "out")),
        registry=None,
        fetcher=fake,  # type: ignore[arg-type]
    )
    # Override _owns_fetcher to True so the pipeline calls close().
    pipeline._owns_fetcher = True  # type: ignore[attr-defined]
    pipeline.run_scan(period="weekly", write=False, sources=[])
    assert fake.close_called is True
