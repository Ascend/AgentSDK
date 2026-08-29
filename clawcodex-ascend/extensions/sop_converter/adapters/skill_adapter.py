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

"""Default adapters for :class:`SkillProtocol` and :class:`SkillFrontmatterProtocol`.

Wraps ``clawcodex_ext.skills.model.Skill`` as a factory function and
``clawcodex_ext.skills.frontmatter.parse_frontmatter`` as a callable
so the SOP converter can work with skills without importing
``clawcodex_ext`` directly.

Field names are 1:1 between the upstream dataclass and the Protocol,
so no property aliasing is needed.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.4.
"""

from __future__ import annotations

import logging
from typing import Any

from extensions.capabilities.skill_protocol import (
    SkillFrontmatterProtocol,
    SkillFrontmatterResultProtocol,
    SkillProtocol,
)

logger = logging.getLogger(__name__)

__all__ = [
    "default_skill_factory",
    "default_frontmatter_parser",
    "default_clear_sop_caches",
]


def default_clear_sop_caches() -> None:
    """Clear runtime caches (commands, context, agent definitions) after loading
    bundle skills or agents.

    Wraps ``clawcodex_ext.command_system.aggregator.clear_commands_cache``,
    ``clawcodex_ext.context_system.prompt_assembly.clear_context_caches``,
    and ``clawcodex_ext.agent.load_agents_dir.clear_agent_definitions_cache``
    so bundle loading code can clear caches without importing ``clawcodex_ext``
    directly.  Each clear failure is logged as a warning so partial cache
    invalidation is diagnosable by the caller.
    """
    try:
        from clawcodex_ext.command_system.aggregator import clear_commands_cache

        clear_commands_cache()
    except Exception as exc:
        logger.warning("Failed to clear commands cache: %s", exc)
    try:
        from clawcodex_ext.context_system.prompt_assembly import clear_context_caches

        clear_context_caches()
    except Exception as exc:
        logger.warning("Failed to clear context caches: %s", exc)
    try:
        from clawcodex_ext.agent.load_agents_dir import clear_agent_definitions_cache

        clear_agent_definitions_cache()
    except Exception as exc:
        logger.warning("Failed to clear agent definitions cache: %s", exc)


def default_skill_factory(**kwargs: Any) -> SkillProtocol:
    """Construct a ``Skill``-compatible instance.

    Accepts the same keyword arguments as
    ``clawcodex_ext.skills.model.Skill``.

    All keyword arguments are forwarded verbatim; the upstream dataclass
    field names match the Protocol exactly.
    """
    from clawcodex_ext.skills.model import Skill

    return Skill(**kwargs)


class _FrontmatterWrapper:
    """Wraps ``parse_frontmatter`` as a ``SkillFrontmatterProtocol``.

    The upstream ``FrontmatterParseResult`` dataclass already exposes
    ``.frontmatter`` (dict) and ``.body`` (str), so it trivially
    satisfies ``SkillFrontmatterResultProtocol`` at runtime without
    an adapter.
    """

    def __call__(self, markdown: str) -> SkillFrontmatterResultProtocol:
        from clawcodex_ext.skills.frontmatter import parse_frontmatter

        return parse_frontmatter(markdown)  # type: ignore[return-value]


default_frontmatter_parser: SkillFrontmatterProtocol = _FrontmatterWrapper()
"""Default frontmatter parser singleton.

Usage::

    result = default_frontmatter_parser(markdown_text)
    frontmatter = result.frontmatter  # dict
    body = result.body               # str
"""
