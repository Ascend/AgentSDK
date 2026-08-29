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

# pylint: disable=ungrouped-imports
"""Markdown-driven agent discovery for ``clawcodex_ext`` and ``extensions/``.

Two root layouts are walked:

* ``clawcodex_ext/agent/agents/*.md`` — first-party decoupled agents.
* ``extensions/<name>/agents/*.md`` — third-party extension packages.

Each ``*.md`` file is parsed by :func:`src.agent.parse_agent_markdown.parse_agent_from_markdown`
(re-using the existing frontmatter schema) and tagged with
``source="clawcodex_ext"`` or ``source="extensions"`` so the merge
order in :func:`src.agent.load_agents_dir.get_agent_definitions_with_overrides`
can apply the correct priority.

The discovery functions are **side-effect free** — they do not register
anything into :class:`clawcodex_ext.agent.registry.AgentRegistry`. The
caller (typically ``get_agent_definitions_with_overrides``) is
responsible for routing the returned list into the merge pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from clawcodex_ext.agent.parse_agent_markdown import parse_agent_from_markdown
from src.skills.frontmatter import parse_frontmatter
from clawcodex_ext.agent.registry import SOURCE_CLAWCODEX_EXT, SOURCE_EXTENSIONS

if TYPE_CHECKING:
    from clawcodex_ext.agent.agent_definitions import AgentDefinition

logger = logging.getLogger(__name__)


# Default roots. Callers may override for tests or for vendored trees.
DEFAULT_CLAWCODEX_EXT_AGENTS_DIR: Path = Path(__file__).parent / "agents"
DEFAULT_EXTENSIONS_ROOT: Path = Path(__file__).parent.parent.parent / "extensions"


def _walk_markdown(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        return sorted(p for p in directory.rglob("*.md") if p.is_file())
    except OSError as exc:
        logger.debug("markdown walk failed for %s: %s", directory, exc)
        return []


def _load_directory(directory: Path, *, source: str) -> Iterable[AgentDefinition]:
    for md_path in _walk_markdown(directory):
        try:
            content = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("skip unreadable agent file %s: %s", md_path, exc)
            continue
        try:
            parsed = parse_frontmatter(content)
        except Exception as exc:  # yaml.safe_load is permissive but be defensive
            logger.debug("skip unparseable frontmatter in %s: %s", md_path, exc)
            continue
        try:
            agent = parse_agent_from_markdown(
                file_path=str(md_path),
                frontmatter=parsed.frontmatter,
                body=parsed.body,
                source=source,  # type: ignore[arg-type]
                base_dir=str(directory),
            )
        except Exception as exc:
            logger.debug("skip agent file %s due to parser error: %s", md_path, exc)
            continue
        if agent is None:
            # parse_agent_from_markdown already logged the reason.
            continue
        yield agent


def discover_clawcodex_ext_agents(
    root: Path | None = None,
) -> list[AgentDefinition]:
    """Discover agents under ``clawcodex_ext/agent/agents/`` (or ``root``)."""
    directory = root or DEFAULT_CLAWCODEX_EXT_AGENTS_DIR
    return list(_load_directory(directory, source=SOURCE_CLAWCODEX_EXT))


def discover_extension_agents(
    extensions_root: Path | None = None,
) -> list[AgentDefinition]:
    """Discover agents under ``extensions/*/agents/``.

    Each direct subdirectory of ``extensions_root`` is treated as an
    independent extension package; only directories containing an
    ``agents/`` subdirectory contribute. This mirrors the layout
    convention used by ``extensions/orchestrator/`` and friends.
    """
    base = extensions_root or DEFAULT_EXTENSIONS_ROOT
    if not base.is_dir():
        return []
    out: list[AgentDefinition] = []
    try:
        ext_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError as exc:
        logger.debug("extensions walk failed for %s: %s", base, exc)
        return out
    for ext_dir in ext_dirs:
        agents_dir = ext_dir / "agents"
        if not agents_dir.is_dir():
            continue
        out.extend(_load_directory(agents_dir, source=SOURCE_EXTENSIONS))
    return out


__all__ = [
    "DEFAULT_CLAWCODEX_EXT_AGENTS_DIR",
    "DEFAULT_EXTENSIONS_ROOT",
    "discover_clawcodex_ext_agents",
    "discover_extension_agents",
]
