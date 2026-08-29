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

# pylint: disable=redefined-outer-name,reimported
"""Discover and merge custom agent definitions from disk + plugins.

Port of ``getAgentDefinitionsWithOverrides`` in
typescript/src/tools/AgentTool/loadAgentsDir.ts. Combines built-in agents
(``src/agent/agent_definitions.py:get_built_in_agents``), plugin agents
(via ``load_plugin_agents``), and on-disk custom agents from managed /
user / project directories (via ``load_markdown_files_for_subdir``).

Last-wins merge order on duplicate ``agent_type``:
    [built-in, plugin, user, project, managed]

A module-level cache keyed on cwd avoids re-walking the filesystem on
every prompt build. Call ``clear_agent_definitions_cache()`` after a
known on-disk change (e.g., the user edits ``~/.clawcodex/agents/foo.md``)
to force a refresh.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

from clawcodex_ext.agent.agent_definitions import AgentDefinition, get_built_in_agents
from clawcodex_ext.agent.parse_agent_markdown import parse_agent_from_markdown
from src.utils.markdown_config_loader import (
    SOURCE_MANAGED,
    SOURCE_PROJECT,
    SOURCE_USER,
    load_markdown_files_for_subdir,
)

logger = logging.getLogger(__name__)


# Each disk source maps to the matching ``AgentSource`` literal so
# downstream consumers can distinguish a managed-policy agent from a
# user one (e.g., to enforce "managed cannot be overridden by user").
_SOURCE_TO_AGENT_SOURCE: dict[str, str] = {
    SOURCE_MANAGED: "managed",
    SOURCE_USER: "user",
    SOURCE_PROJECT: "project",
}

# Priority order for last-wins merge — earlier entries are overridden by
# later ones if they share an agent_type.
# Order: built-in < plugin < clawcodex_ext < extensions < user < project < managed.
# The two new tiers (``clawcodex_ext`` / ``extensions``) come from the decoupled
# registration API in ``clawcodex_ext/agent/registry.py`` and the markdown
# discovery in ``clawcodex_ext/agent/markdown_discovery.py``.
_MERGE_ORDER: tuple[str, ...] = (
    "built-in",
    "plugin",
    "clawcodex_ext",
    "extensions",
    SOURCE_USER,
    SOURCE_PROJECT,
    SOURCE_MANAGED,
)


# Cache is keyed on ``os.path.realpath(cwd)`` so symlinked / trailing-slash
# variants of the same project collapse into a single entry. The cache is
# session-bound; SDK callers that hop between unrelated projects can grow
# it unboundedly — acceptable for now since per-cwd discovery is cheap.
_agent_dir_cache: dict[str, list[AgentDefinition]] = {}


def _cache_key(cwd: str) -> str:
    try:
        return os.path.realpath(cwd)
    except (OSError, ValueError):
        return cwd


def clear_agent_definitions_cache() -> None:
    """Drop the discovery cache. Call after on-disk agent changes."""
    _agent_dir_cache.clear()


def get_active_agents_from_list(
    agents: Iterable[AgentDefinition],
) -> list[AgentDefinition]:
    """Last-wins dedup by ``agent_type`` while preserving input order.

    Mirrors ``getActiveAgentsFromList`` from loadAgentsDir.ts:192-220.
    Callers are responsible for arranging input order so the desired
    override priority is honoured (lowest priority first, highest last).
    """
    by_type: dict[str, AgentDefinition] = {}
    order: list[str] = []
    for agent in agents:
        if agent.agent_type not in by_type:
            order.append(agent.agent_type)
        by_type[agent.agent_type] = agent
    return [by_type[t] for t in order]


def _load_custom_agents(cwd: str) -> dict[str, list[AgentDefinition]]:
    """Group disk-discovered agents by their disk source label."""
    grouped: dict[str, list[AgentDefinition]] = {
        SOURCE_USER: [],
        SOURCE_PROJECT: [],
        SOURCE_MANAGED: [],
    }
    files = load_markdown_files_for_subdir("agents", cwd)
    for md in files:
        agent_source = _SOURCE_TO_AGENT_SOURCE.get(md.source, "user")
        agent = parse_agent_from_markdown(
            file_path=md.file_path,
            frontmatter=md.frontmatter,
            body=md.body,
            source=agent_source,  # type: ignore[arg-type]
            base_dir=md.base_dir,
        )
        if agent is None:
            continue
        grouped[md.source].append(agent)
    return grouped


def get_agent_definitions_with_overrides(cwd: str) -> list[AgentDefinition]:
    """Return the merged list of agents visible from ``cwd``.

    Cache-keyed on ``cwd``. Built-ins are always included; the user can
    override a built-in by defining an agent with the same ``agent_type``.
    On any unexpected loader error the built-ins are returned alone — a
    broken custom agent file should never disable the model's ability to
    spawn the built-in agents.
    """
    key = _cache_key(cwd)
    cached = _agent_dir_cache.get(key)
    if cached is not None:
        return list(cached)

    try:
        builtins = list(get_built_in_agents())
        try:
            from src.agent.load_plugin_agents import load_plugin_agents
            from src.plugins import get_loaded_plugins

            plugin_agents = load_plugin_agents(get_loaded_plugins())
        except Exception:  # nosec
            logger.exception("plugin agent loading failed; continuing without plugin agents")
            plugin_agents = []

        # Decoupled extension agents: programmatic registry + markdown
        # discovery under clawcodex_ext/agent/agents/ and extensions/*/agents/.
        # Wrapped in its own try/except so a broken extension never disables
        # the model's ability to spawn built-in agents.
        clawcodex_ext_agents: list[AgentDefinition] = []
        extension_agents: list[AgentDefinition] = []
        try:
            from clawcodex_ext.agent import ensure_bundled_agents_registered
            from clawcodex_ext.agent.markdown_discovery import (
                discover_clawcodex_ext_agents,
                discover_extension_agents,
            )
            from clawcodex_ext.agent.registry import (
                AgentRegistry,
                SOURCE_CLAWCODEX_EXT,
                SOURCE_EXTENSIONS,
            )

            ensure_bundled_agents_registered()
            # Programmatic registrations + markdown files together.
            clawcodex_ext_agents = AgentRegistry.by_source(SOURCE_CLAWCODEX_EXT) + discover_clawcodex_ext_agents()
            extension_agents = AgentRegistry.by_source(SOURCE_EXTENSIONS) + discover_extension_agents()
        except Exception:  # nosec
            logger.exception("extension agent loading failed; continuing without extension agents")
            clawcodex_ext_agents = []
            extension_agents = []

        custom = _load_custom_agents(cwd)

        sources_in_order: dict[str, list[AgentDefinition]] = {
            "built-in": builtins,
            "plugin": plugin_agents,
            "clawcodex_ext": clawcodex_ext_agents,
            "extensions": extension_agents,
            SOURCE_USER: custom[SOURCE_USER],
            SOURCE_PROJECT: custom[SOURCE_PROJECT],
            SOURCE_MANAGED: custom[SOURCE_MANAGED],
        }
        flat: list[AgentDefinition] = []
        for source_key in _MERGE_ORDER:
            flat.extend(sources_in_order.get(source_key, []))

        active = get_active_agents_from_list(flat)
        _agent_dir_cache[key] = active
        return list(active)
    except Exception:  # nosec
        logger.exception("agent discovery failed; falling back to built-ins")
        return list(get_built_in_agents())


def resolve_agent_dir_override(
    tool_context: Any | None = None,
    runtime_context: Any | None = None,
) -> Path | None:
    """Return the bundle / ``--agent <dir>`` root that owns ``.claude/agents``."""
    from pathlib import Path

    candidates: list[Any] = []
    if runtime_context is not None:
        opts = getattr(runtime_context, "options", None)
        if opts is not None:
            candidates.append(getattr(opts, "agent_dir_override", None))
            candidates.append(getattr(opts, "bundle_path", None))
    if tool_context is not None:
        candidates.append(getattr(tool_context, "_agent_dir_override", None))
        tc_opts = getattr(tool_context, "options", None)
        if tc_opts is not None:
            candidates.append(getattr(tc_opts, "agent_dir_override", None))
            candidates.append(getattr(tc_opts, "bundle_path", None))

    for raw in candidates:
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        agents_dir = resolved / ".claude" / "agents"
        if agents_dir.is_dir():
            return resolved
    return None


def get_agents_for_mentions(
    workspace_root: str | Path,
    *,
    tool_context: Any | None = None,
    runtime_context: Any | None = None,
) -> list[AgentDefinition]:
    """Agents visible for ``@agent-<type>`` mentions and REPL/TUI completion.

    Merges workspace discovery with bundle agents from ``--agent <dir>``
    (stored on ``runtime_context.options``, ``tool_context._agent_dir_override``,
    or ``bundle_path``).
    """
    from pathlib import Path

    workspace_root = Path(workspace_root)
    agents = list(get_agent_definitions_with_overrides(str(workspace_root)))
    override = resolve_agent_dir_override(tool_context, runtime_context)
    if override is None:
        return agents

    known = {a.agent_type for a in agents}
    for agent in get_agent_definitions_with_overrides(str(override)):
        if agent.agent_type not in known:
            agents.append(agent)
            known.add(agent.agent_type)
    return agents
