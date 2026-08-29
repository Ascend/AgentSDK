#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""SOP Defaults container — dependency injection for SOP converter.

Defines the :class:`SOPDefaults` singleton that carries factory functions
and protocol implementations for all external dependencies the SOP converter
needs.  Populated at import time by :func:`fill_defaults` (called from
``extensions/sop_converter/__init__.py``) so the core algorithm layer never
imports ``clawcodex_ext`` or ``src`` directly.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.4 and §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from extensions.capabilities.agent_definition_protocol import (
    AgentDefinitionProtocol,
)
from extensions.capabilities.permission_protocol import (
    PermissionContextProtocol,
)
from extensions.capabilities.skill_protocol import (
    SkillFrontmatterProtocol,
    SkillProtocol,
)
from extensions.capabilities.sop_provider_protocol import (
    SOPAssistantProviderProtocol,
)
from extensions.capabilities.tool_authoring_protocol import (
    ToolAuthoringProtocol,
)

__all__ = [
    "DEFAULTS",
    "SOPDefaults",
    "fill_defaults",
]


@dataclass
class SOPDefaults:
    """Dependency container for the SOP converter.

    Each field is populated once by :func:`fill_defaults` during module
    initialisation.  Consumers read from the module-level :data:`DEFAULTS`
    singleton.  All fields are ``Optional`` so the dataclass can be
    constructed empty; after ``fill_defaults`` every field except
    ``sop_provider`` is guaranteed non-``None``.
    """

    # --- Agent definition ---
    agent_definition_factory: Optional[Callable[..., AgentDefinitionProtocol]] = None
    """Factory: ``def agent_definition(**kwargs) -> AgentDefinitionProtocol``.

    The default implementation wraps
    ``clawcodex_ext.agent.agent_definitions.AgentDefinition``.
    """

    # --- Skills ---
    skill_factory: Optional[Callable[..., SkillProtocol]] = None
    """Factory: ``def skill(**kwargs) -> SkillProtocol``.

    The default implementation wraps
    ``clawcodex_ext.skills.model.Skill``.
    """

    frontmatter_parser: Optional[SkillFrontmatterProtocol] = None
    """Callable: ``parse_frontmatter(markdown: str) -> SkillFrontmatterResultProtocol``.

    The default implementation wraps
    ``clawcodex_ext.skills.frontmatter.parse_frontmatter``.
    """

    # --- Tool authoring ---
    tool_authoring: Optional[ToolAuthoringProtocol] = None
    """Aggregate tool-authoring surface (persistence / spec / validation / factory / registry).

    The default implementation wraps the five
    ``clawcodex_ext.agent.tool_authoring.*`` sub-modules.
    """

    # --- Permission ---
    permission_context_factory: Optional[Callable[..., PermissionContextProtocol]] = None
    """Factory: ``def permission_context(**kwargs) -> PermissionContextProtocol``.

    The default implementation wraps
    ``clawcodex_ext.permissions.types.ToolPermissionContext`` with
    property aliases (``is_bypass``, ``should_avoid_prompts``).
    """

    # --- Optional LLM assistant ---
    sop_provider: Optional[SOPAssistantProviderProtocol] = None
    """Optional LLM assistant for ``skill_grouper`` semantic grouping.

    When ``None`` the grouper falls back to rule-only grouping.
    """

    # --- Agent loader (exploration guard) ---
    agent_loader: Optional[Callable[[], list[AgentDefinitionProtocol]]] = None
    """Callable that returns all known agent definitions.

    Used by ``sop_exploration_guard`` to discover existing agents.
    The default implementation wraps
    ``clawcodex_ext.agent.load_agents_dir.get_agent_definitions_with_overrides``.
    """

    # --- Cache clearing (bundle skills) ---
    clear_sop_caches: Optional[Callable[[], None]] = None
    """Callable that clears runtime caches (commands + context) after loading
    bundle skills.

    The default implementation wraps
    ``clawcodex_ext.command_system.aggregator.clear_commands_cache`` and
    ``clawcodex_ext.context_system.prompt_assembly.clear_context_caches``.
    """


def fill_defaults(container: SOPDefaults) -> None:
    """Populate *container* with default adapter implementations.

    Idempotent — safe to call multiple times.  Each field is set only
    when it is still ``None``, so explicit overrides survive a re-run.
    """
    from .agent_definition_adapter import default_agent_definition_factory
    from .skill_adapter import default_skill_factory, default_frontmatter_parser
    from .tool_authoring_adapter import DefaultToolAuthoring
    from .permission_adapter import default_permission_context_factory

    if container.agent_definition_factory is None:
        container.agent_definition_factory = default_agent_definition_factory

    if container.skill_factory is None:
        container.skill_factory = default_skill_factory

    if container.frontmatter_parser is None:
        container.frontmatter_parser = default_frontmatter_parser

    if container.tool_authoring is None:
        container.tool_authoring = DefaultToolAuthoring()

    if container.permission_context_factory is None:
        container.permission_context_factory = default_permission_context_factory

    # sop_provider is intentionally left as None — it is optional and
    # must be explicitly set by the consumer that wants LLM-assisted
    # grouping.

    if container.agent_loader is None:
        from .agent_definition_adapter import default_agent_loader

        container.agent_loader = default_agent_loader

    if container.clear_sop_caches is None:
        from .skill_adapter import default_clear_sop_caches

        container.clear_sop_caches = default_clear_sop_caches


# Module-level singleton — populated by calling ``fill_defaults(DEFAULTS)``
# from ``extensions/sop_converter/__init__.py``.
DEFAULTS = SOPDefaults()
