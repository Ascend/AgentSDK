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

# pylint: disable=relative-beyond-top-level
"""Source registry for SR-5.1 Community Feature Radar.

Implements the ``SourceRegistry`` sketched in the feature design.
The registry owns the set of :class:`WatchSource` records the radar
tracks, persists them as YAML, and ships with a sensible default set
derived from the Phase-1 project list.

Design decisions:
* Pure stdlib (``json`` + optional ``yaml``). ``PyYAML`` is already a
  required dep of ClawCodex, so we use it for ergonomic round-trips,
  but fall back to JSON if the import fails (e.g. on a slim CI image).
* ``with_defaults()`` is the factory used by the CLI when no
  ``sources.yaml`` exists yet. The default list mirrors the Phase-1
  table in the original design.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .fetcher import detect_repo_domain
from .models import SourceDomain, WatchSource

_log = logging.getLogger(__name__)


# Default sources — keep in sync with the original design's phase lists.
PHASE1_SOURCES: list[dict[str, Any]] = [
    {
        "name": "claude-code",
        "repo": "anthropics/claude-code",
        "track_releases": True,
        "changelog_path": "CHANGELOG.md",
        "notes": "上游源，核心对齐目标",
        "roadmap_keywords": ["agent", "tool", "permission", "memory", "compact"],
    },
    {
        "name": "aider",
        "repo": "paul-gauthier/aider",
        "track_releases": True,
        "release_tag_filter": r"\d+\.\d+\.\d+",
        "notes": "Python 生态最活跃的编码 Agent",
        "roadmap_keywords": ["lint", "edit", "commit", "test", "refactor"],
    },
    {
        "name": "swe-agent",
        "repo": "princeton-nlp/SWE-agent",
        "track_releases": True,
        "notes": "自动修复 GitHub issue 的标杆项目",
        "roadmap_keywords": ["issue", "patch", "test", "verify"],
    },
    {
        "name": "openhands",
        "repo": "All-Hands-AI/OpenHands",
        "track_releases": True,
        "notes": "通用 AI 软件工程 Agent",
        "roadmap_keywords": ["workspace", "agent", "sandbox", "browser"],
    },
    {
        "name": "autogen",
        "repo": "microsoft/autogen",
        "track_releases": True,
        "notes": "多 Agent 对话框架",
        "roadmap_keywords": ["multi-agent", "conversation", "group", "router"],
    },
    {
        "name": "crewai",
        "repo": "crewAIInc/crewAI",
        "track_releases": True,
        "notes": "多 Agent 编排框架",
        "roadmap_keywords": ["crew", "role", "task", "process"],
    },
    {
        "name": "langgraph",
        "repo": "langchain-ai/langgraph",
        "track_releases": True,
        "notes": "Agent 图状工作流引擎",
        "roadmap_keywords": ["graph", "state", "node", "edge", "workflow"],
    },
]


PHASE2_SOURCES: list[dict[str, Any]] = [
    {
        "name": "cline",
        "repo": "cline/cline",
        "track_releases": True,
        "notes": "VS Code 编码 Agent（Claude Dev 继任）",
        "roadmap_keywords": ["vscode", "extension", "browser", "mcp"],
    },
    {
        "name": "continue",
        "repo": "continuedev/continue",
        "track_releases": True,
        "notes": "VS Code / JetBrains 编码助手",
        "roadmap_keywords": ["vscode", "jetbrains", "autocomplete", "chat"],
    },
    {
        "name": "codegate",
        "repo": "stacklok/codegate",
        "track_releases": True,
        "notes": "AI 编码助手的安全/沙箱网关",
        "roadmap_keywords": ["security", "sandbox", "policy", "guardrail"],
    },
    {
        "name": "goose",
        "repo": "block/goose",
        "track_releases": True,
        "notes": "Block 出品的本地 AI Agent",
        "roadmap_keywords": ["desktop", "extension", "tool", "recipe"],
    },
    {
        "name": "eliza",
        "repo": "elizaOS/eliza",
        "track_releases": True,
        "notes": "多模态 Agent 运行时框架",
        "roadmap_keywords": ["agent", "runtime", "character", "plugin"],
    },
    {
        "name": "openclaw",
        "repo": "openclaw/openclaw",
        "track_releases": True,
        "notes": "Node.js 编码 Agent 同类产品",
        "roadmap_keywords": ["nodejs", "agent", "cli", "tool"],
    },
]


# Phase 1 remains the default seed; Phase 2 sources can be layered on
# top via the ``--include-phase2`` CLI flag or ``Registry.with_defaults(
# include_phase2=True)``. ``DEFAULT_SOURCES`` keeps pointing at Phase 1
# so existing installs and tests see no churn.
DEFAULT_SOURCES: list[dict[str, Any]] = list(PHASE1_SOURCES)


def _load_yaml_or_json(path: Path) -> Any:
    """Load a YAML or JSON document; YAML is preferred when present."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            _log.warning(
                "PyYAML not available; cannot parse %s as YAML — aborting load",
                path,
            )
            raise
        return yaml.safe_load(text)
    return json.loads(text)


def _dump_yaml_or_json(data: Any, path: Path) -> None:
    """Persist ``data`` as YAML or JSON based on suffix."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            _log.warning("PyYAML not available; falling back to JSON for %s", path)
            json_path = path.with_suffix(".json")
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class SourceRegistry:
    """Manage :class:`WatchSource` records persisted on disk.

    The registry is intentionally simple: a name→WatchSource dict backed
    by a single YAML file. ``load()`` returns the populated dict for
    fluent chaining.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._sources: dict[str, WatchSource] = {}
        self._load_failed = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> dict[str, WatchSource]:
        """Load sources from ``self.path``.

        Returns the populated dict. If the file is missing, the registry
        stays empty; callers should use :meth:`with_defaults` or
        :meth:`add` to populate it.
        """
        self._sources = {}
        self._load_failed = False
        if not self.path.exists():
            return self._sources
        try:
            raw = _load_yaml_or_json(self.path)
        except Exception as exc:  # noqa: BLE001 — user-data, surface in CLI
            _log.warning("failed to read %s: %s", self.path, exc)
            self._load_failed = True
            return self._sources

        items: Iterable[dict[str, Any]]
        if isinstance(raw, dict) and isinstance(raw.get("sources"), list):
            items = raw["sources"]
        elif isinstance(raw, list):
            items = raw
        else:
            _log.warning("unexpected sources file shape in %s; expected list", self.path)
            self._load_failed = True
            return self._sources

        for item in items:
            try:
                src = WatchSource.from_dict(item)
            except ValueError as exc:
                _log.warning("skipping invalid source entry %r: %s", item, exc)
                continue
            self._sources[src.name] = src
        return self._sources

    def save(self) -> None:
        """Persist the registry to ``self.path``.

        Creates parent directories on demand. Writes ``{"sources": [...]}``
        for YAML and a bare list for JSON, matching what :meth:`load`
        accepts.
        """
        if self._load_failed:
            raise RuntimeError(f"refusing to overwrite {self.path} after a failed load")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = [src.to_dict() for src in self._sources.values()]
        suffix = self.path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            payload: Any = {"sources": items}
        else:
            payload = items
        _dump_yaml_or_json(payload, self.path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, source: WatchSource) -> None:
        """Insert or replace a source by name."""
        self._sources[source.name] = source

    def remove(self, name: str) -> bool:
        """Remove a source by name. Returns True when present."""
        return self._sources.pop(name, None) is not None

    def get(self, name: str) -> WatchSource | None:
        return self._sources.get(name)

    def list(self) -> list[WatchSource]:
        return list(self._sources.values())

    def names(self) -> list[str]:
        return list(self._sources.keys())

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @classmethod
    def with_defaults(
        cls,
        path: Path | str | None = None,
        *,
        extras: Iterable[WatchSource] | None = None,
        include_phase2: bool = False,
    ) -> SourceRegistry:
        """Build a registry pre-populated with the Phase-1 default sources.

        ``extras`` lets callers layer local sources on top without
        overriding the defaults. ``path`` is optional; passing it lets
        the same registry be saved later without rewiring. Pass
        ``include_phase2=True`` to also seed the Phase-2 expansion
        projects (Cline / Continue.dev / Goose / OpenClaw / etc.).
        """
        reg = cls(path if path is not None else Path("sources.yaml"))
        seed = list(PHASE1_SOURCES)
        if include_phase2:
            seed.extend(PHASE2_SOURCES)
        for item in seed:
            try:
                reg.add(WatchSource.from_dict(item))
            except ValueError as exc:
                _log.warning("default source rejected: %s", exc)
        for extra in extras or ():
            reg.add(extra)
        return reg

    # ------------------------------------------------------------------
    # Domain auto-detection
    # ------------------------------------------------------------------

    def auto_detect_domains(self, cache_dir: Path | str, *, github_token: str | None = None) -> int:
        """Detect domains for sources still marked ``general``.

        Calls :func:`detect_repo_domain` (which hits the GitHub API and
        caches results for 24 h) for every source whose ``domain`` is
        ``"general"``.  Only ``embodied_ai`` / ``spatial_intelligence``
        results update the source (``code_agent`` is the general default);
        the registry is persisted to disk after any update.

        Returns the number of sources whose domain was updated.
        """
        updated = 0
        for source in self._sources.values():
            if source.domain != SourceDomain.GENERAL.value:
                continue
            try:
                detected = detect_repo_domain(source.repo, cache_dir, github_token=github_token)
            except Exception as exc:  # noqa: BLE001
                _log.warning("domain detection failed for %s: %s", source.name, exc)
                continue
            if detected is not None and detected in {
                SourceDomain.EMBODIED_AI.value,
                SourceDomain.SPATIAL_INTELLIGENCE.value,
            }:
                source.domain = detected
                updated += 1
                _log.info("auto-detected domain=%s for source=%s", detected, source.name)
        if updated:
            self.save()
        return updated


# ---------------------------------------------------------------------------
# Convenience: locate the user config directory
# ---------------------------------------------------------------------------


def default_registry_path() -> Path:
    """Return ``~/.clawcodex/community-radar/sources.yaml``.

    Honours ``$CLAWCODEX_HOME`` when set (matches the convention used
    elsewhere in ClawCodex so tests and containers can override it).
    """
    base = os.environ.get("CLAWCODEX_HOME")
    root = Path(base) if base else Path.home() / ".clawcodex"
    return root / "community-radar" / "sources.yaml"
