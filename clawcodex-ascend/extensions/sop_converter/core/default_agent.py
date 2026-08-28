#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Default Agent replacement — auto-detect the overview Agent at startup.

Priority (high → low):
  1. ``--agent <agent-type>`` explicitly set via CLI
  2. ``.claude/agents/clawcodex-overview.md`` auto-detection
  3. ``GENERAL_PURPOSE_AGENT`` current default behavior (fallback)

The overview Agent's system prompt is injected via ``append_system_prompt``,
retaining all standard sections from ``build_full_system_prompt()``.

When ``--agent <bundle_dir>`` points to a POS bundle directory, the main loop
also switches to the overview's ``AgentDefinition`` and restricts the tool set
to ``POS_PROXY_BASE_TOOLS``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Name convention for the overview agent
OVERVIEW_AGENT_NAME = "clawcodex-overview"


def resolve_default_agent(cwd: str | Path = ".") -> dict[str, Any] | None:
    """Scan the working directory for the default overview Agent.

    Detects whether ``.claude/agents/clawcodex-overview.md`` exists and, if so,
    parses and returns its frontmatter + body.

    Parameters
    ----------
    cwd : str | Path
        Working directory (defaults to the current directory).

    Returns
    -------
    dict | None
        ``{"name", "description", "model", "tools", "skills",
        "system_prompt_body"}``, or None when not found.
    """
    agents_dir = Path(cwd).resolve() / ".claude" / "agents"
    overview_file = agents_dir / f"{OVERVIEW_AGENT_NAME}.md"

    if not overview_file.is_file():
        return None

    return _parse_agent_file(overview_file)


def resolve_agent_by_type(
    cwd: str | Path,
    agent_type: str,
    agent_dir_override: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find an agent definition by agent type (frontmatter name).

    Scans ``.claude/agents/*.md`` and returns the first matching agent.

    If ``agent_type`` is an existing directory path, looks for the overview
    agent under that directory's ``.claude/agents/``. If ``agent_dir_override``
    is given, searches that directory first.

    Parameters
    ----------
    cwd : str | Path
        Working directory.
    agent_type : str
        The frontmatter ``name`` value. If it points to an existing directory,
        also tries to load the overview agent from that directory's
        ``.claude/agents/``.
    agent_dir_override : str | Path | None
        If given, searches this directory first.

    Returns
    -------
    dict | None
        The matching agent definition, or None.
    """
    for agents_dir in _agent_search_dirs(cwd, agent_type, agent_dir_override):
        for md_file in _list_markdown_files(agents_dir):
            try:
                frontmatter, body = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
                if frontmatter.get("name") == agent_type:
                    return {
                        "name": frontmatter.get("name", agent_type),
                        "description": frontmatter.get("description", ""),
                        "model": frontmatter.get("model"),
                        "tools": frontmatter.get("tools", []),
                        "skills": frontmatter.get("skills", []),
                        "system_prompt_body": body,
                    }
            except (ValueError, OSError) as exc:
                logger.warning("Failed to parse %s: %s", md_file, exc)
                continue

    return None


def _agent_search_dirs(
    cwd: str | Path,
    agent_type: str,
    agent_dir_override: str | Path | None,
) -> list[Path]:
    """Collect candidate ``agents`` directories, deduplicated, in search order.

    The override directory (if given) is searched first, followed by the
    directory supplied as *agent_type* when it is an existing directory, and
    finally the ``.claude/agents`` directory under *cwd*.
    """
    bases = _agent_search_bases(cwd, agent_type, agent_dir_override)
    seen: set[Path] = set()
    agents_dirs: list[Path] = []
    for base in bases:
        agents_dir = _agents_dir_for_base(base)
        resolved = agents_dir.resolve()
        if resolved in seen or not agents_dir.is_dir():
            continue
        seen.add(resolved)
        agents_dirs.append(agents_dir)
    return agents_dirs


def _agent_search_bases(
    cwd: str | Path,
    agent_type: str,
    agent_dir_override: str | Path | None,
) -> list[Path]:
    """Collect candidate base directories for agent search, in priority order."""
    bases: list[Path] = []
    if agent_dir_override is not None:
        bases.append(Path(agent_dir_override).resolve())

    agent_type_path = Path(str(agent_type)).resolve()
    if agent_type_path.is_dir():
        bases.append(agent_type_path)

    cwd_agents = Path(cwd).resolve() / ".claude" / "agents"
    if cwd_agents.is_dir():
        bases.append(cwd_agents)
    return bases


def _agents_dir_for_base(base: Path) -> Path:
    """Return the ``agents`` directory inside a search base."""
    return base if base.name == "agents" else base / ".claude" / "agents"


def _parse_agent_file(file_path: Path) -> dict[str, Any]:
    """Parse an agent markdown file, returning frontmatter + body."""
    content = file_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content)
    return {
        "name": frontmatter.get("name", file_path.stem),
        "description": frontmatter.get("description", ""),
        "model": frontmatter.get("model"),
        "tools": frontmatter.get("tools", []),
        "skills": frontmatter.get("skills", []),
        "system_prompt_body": body,
    }


# Public alias
parse_agent_file = _parse_agent_file


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter (between ``---`` delimiters).

    Uses ``yaml.safe_load`` so quoted colons, booleans and nested dicts
    are handled correctly.  Falls back to an empty mapping when the block
    cannot be parsed (e.g. malformed YAML), logging a warning instead of
    raising so callers still receive the body text.

    Parameters
    ----------
    content : str
        Full markdown file content.

    Returns
    -------
    (frontmatter_dict, body_str)
    """
    block, body = _split_frontmatter(content)
    if block is None:
        return {}, content

    try:
        parsed = yaml.safe_load(block)
    except Exception as exc:
        logger.warning("Failed to parse frontmatter: %s", exc)
        parsed = None
    frontmatter: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    return _normalize_frontmatter_lists(frontmatter), body


def _split_frontmatter(content: str) -> tuple[str | None, str]:
    """Return the text between the opening/closing ``---`` markers and the body.

    Returns ``(None, content)`` when the document has no frontmatter block.
    """
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, content

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :]).strip()
    return None, content


def _normalize_frontmatter_lists(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Wrap single-string ``tools``/``skills`` values as one-element lists."""
    for key in ("tools", "skills"):
        if key in frontmatter and isinstance(frontmatter[key], str):
            frontmatter[key] = [frontmatter[key]]
    return frontmatter


def _list_markdown_files(agents_dir: Path) -> list[Path]:
    """List all markdown files under the directory."""
    if not agents_dir.is_dir():
        return []
    return sorted(agents_dir.rglob("*.md"))
