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

# pylint: disable=cyclic-import

"""Model-visible skill selection for isolated agent contexts."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .catalog import get_skill_catalog


_AVAILABLE_SKILLS_SECTION = re.compile(
    r"(?:\n{2,})?^# Available Skills\s*\n.*?(?=\n{2,}^#\s|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)


def skill_tool_is_available(
    *,
    context: Any,
    tool_registry: Any,
    tools: Iterable[Any],
) -> bool:
    """Return whether this exact child request can expose ``Skill``."""

    skill_tool = tool_registry.get("Skill")
    if skill_tool is None:
        return False
    try:
        if not skill_tool.is_enabled():
            return False
    except Exception:
        return False

    permission_context = getattr(context, "permission_context", None)
    if permission_context is not None and permission_context.blocks(skill_tool.name):
        return False

    return any(str(getattr(tool, "name", "")).lower() == skill_tool.name.lower() for tool in tools)


def _skill_is_denied_by_permission(name: str, context: Any) -> bool:
    """Honor whole-tool, canonical-name, and trailing-star deny rules."""

    permission_context = getattr(context, "permission_context", None)
    if permission_context is None:
        return False
    try:
        from clawcodex_ext.permissions.rules import get_deny_rules  # pylint: disable=no-name-in-module

        rules = get_deny_rules(permission_context)
    except Exception:
        return True

    for rule in rules:
        value = rule.rule_value
        if value.tool_name != "Skill":
            continue
        pattern = value.rule_content
        if pattern is None:
            return True
        if pattern == name or (pattern.endswith("*") and name.startswith(pattern[:-1])):
            return True
    return False


def filter_model_visible_skills(skills: Iterable[Any], context: Any) -> list[Any]:
    """Return prompt skills enabled, model-invocable, and not explicitly denied."""

    visible: list[Any] = []
    for skill in skills:
        if getattr(skill, "type", "prompt") != "prompt":
            continue
        if getattr(skill, "disable_model_invocation", False):
            continue
        enabled = getattr(skill, "is_enabled", True)
        try:
            if not bool(enabled() if callable(enabled) else enabled):
                continue
        except Exception:
            continue
        name = str(getattr(skill, "name", skill))
        if _skill_is_denied_by_permission(name, context):
            continue
        visible.append(skill)
    return visible


def _strip_available_skills(system_prompt: str) -> str:
    return _AVAILABLE_SKILLS_SECTION.sub("", system_prompt).strip()


def refresh_agent_skill_listing(
    system_prompt: str,
    *,
    context: Any,
    tool_registry: Any,
    tools: Iterable[Any],
    provider: Any,
) -> str:
    """Replace inherited skill prose with the child's live catalog view."""

    prompt = _strip_available_skills(system_prompt)
    skills: list[Any] = []
    tools = tuple(tools)
    if skill_tool_is_available(
        context=context,
        tool_registry=tool_registry,
        tools=tools,
    ):
        snapshot = get_skill_catalog(context)
        skills = filter_model_visible_skills(snapshot.skills, context)

    if not skills:
        return prompt

    from clawcodex_ext.context_system.context_analyzer import get_context_window_for_model
    from clawcodex_ext.context_system.prompt_assembly import _build_skill_section

    section = _build_skill_section(
        skills,
        # This path deliberately rebuilds the child's *live* workspace-scoped
        # catalog.  The prompt-section cache is process-global and keyed only
        # as ``skills``; reusing it here can leak a previous workspace's skill
        # listing into a later fork.
        use_cache=False,
        context_window_tokens=get_context_window_for_model(str(getattr(provider, "model", "") or "")),
    )
    if section is None or not section.content:
        return prompt
    return f"{prompt}\n\n{section.content}" if prompt else section.content


__all__ = [
    "filter_model_visible_skills",
    "refresh_agent_skill_listing",
    "skill_tool_is_available",
]
