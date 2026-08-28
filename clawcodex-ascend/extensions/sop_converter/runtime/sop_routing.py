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

"""Runtime guards for SOP bundle agent delegation."""

from __future__ import annotations

import re
from typing import Any

from extensions.capabilities.agent_definition_protocol import AgentToolConstants

_BUILTIN_DELEGATION_TYPES = frozenset({"general-purpose", "Explore", "Plan"})

_SKILL_OR_TOOLSEARCH = re.compile(r"\b(?:Skill|ToolSearch)\s*\(", re.IGNORECASE)
_SELECT_TOOL = re.compile(r"\bselect:[a-z0-9-]+\b", re.IGNORECASE)
_AGENT_MENTION_IN_PROMPT = re.compile(r"@agent-([\w.@-]+)")


def _coerce_stage_id(stage_id: Any) -> int | None:
    """Normalise a YAML ``stage.id`` (int or digit string) to ``int``.

    ``None`` / non-numeric values degrade to ``None`` instead of being
    silently lost by an ``isinstance(int)`` guard that rejects strings.
    """
    if stage_id is None:
        return None
    try:
        return int(stage_id)
    except (TypeError, ValueError):
        return None


def looks_like_direct_sdk_execution(prompt: str) -> bool:
    """True when the prompt describes Skill/ToolSearch/SDK tool work."""
    text = (prompt or "").strip()
    if not text:
        return False
    if _SKILL_OR_TOOLSEARCH.search(text):
        return True
    if _SELECT_TOOL.search(text):
        return True
    lowered = text.lower()
    sdk_markers = (
        "team-memory-dir",
        "ensure-dir",
        "toolsearch",
        "memory-skill",
        "sdk 工具",
        "调用工具",
    )
    return any(marker in lowered for marker in sdk_markers)


def requested_agent_types_in_prompt(prompt: str) -> list[str]:
    """Return ``agent_type`` strings from ``@agent-<type>`` mentions in *prompt*."""
    if not prompt:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _AGENT_MENTION_IN_PROMPT.finditer(prompt):
        agent_type = match.group(1)
        if agent_type.startswith("agent-"):
            agent_type = agent_type[len("agent-") :]
        if agent_type not in seen:
            seen.add(agent_type)
            ordered.append(agent_type)
    return ordered


def list_domain_agent_types(agent_definitions: list[Any]) -> list[str]:
    """Return bundle domain agents (``*-agent``, excluding overview)."""
    names: list[str] = []
    for agent in agent_definitions:
        agent_type = getattr(agent, "agent_type", None) or getattr(agent, "name", "")
        if not isinstance(agent_type, str):
            continue
        if not agent_type.endswith("-agent"):
            continue
        if agent_type in ("clawcodex-overview",):
            continue
        names.append(agent_type)
    return sorted(set(names))


def check_bundle_agent_delegation(
    *,
    subagent_type: str | None,
    prompt: str,
    agent_definitions: list[Any],
) -> str | None:
    """Return an error message when overview delegates SDK work to the wrong agent."""
    try:
        from extensions.sop_converter.bundle_context import get_active_bundle
    except ImportError:
        return None

    if get_active_bundle() is None:
        return None

    resolved = subagent_type or "general-purpose"
    domain_agents = list_domain_agent_types(agent_definitions)
    if not domain_agents:
        return None

    if resolved in domain_agents:
        return None

    if resolved not in _BUILTIN_DELEGATION_TYPES:
        return None

    requested = requested_agent_types_in_prompt(prompt)
    for agent_type in requested:
        if agent_type in domain_agents:
            return (
                f'SOP bundle mode: user requested "@agent-{agent_type}" but delegation '
                f'used "{resolved}". Call Agent(subagent_type="{agent_type}", prompt="...") '
                f"instead."
            )

    if not looks_like_direct_sdk_execution(prompt):
        return None

    examples = ", ".join(f'Agent(subagent_type="{name}", prompt="...")' for name in domain_agents[:4])
    if len(domain_agents) > 4:
        examples += ", ..."
    return (
        f"SOP bundle mode: do not delegate SDK Skill/ToolSearch/tool execution to "
        f'"{resolved}". Use a domain agent instead ({", ".join(domain_agents)}). '
        f"Example: {examples}"
    )


def refresh_domain_agent_sop_prompts(agent_definitions: list[Any]) -> list[Any]:
    """Re-apply latest SOP domain-agent body when a bundle session is active."""
    try:
        from dataclasses import replace

        from extensions.sop_converter.bundle_context import get_active_bundle
        from extensions.sop_converter.sop_prompts import (
            agent_type_to_skill_name,
            domain_agent_sop_body,
            infer_stage_label_from_skill,
            pick_pipeline_execute_tool,
            stage_agent_sop_body,
        )
        from extensions.sop_converter.workflow_project import (
            is_prefixed_stage_agent,
            read_workflow_project_name,
            read_workflow_stage_for_agent,
        )
    except ImportError:
        return agent_definitions

    if get_active_bundle() is None:
        return agent_definitions

    bundle = get_active_bundle()
    sdk_source_dir = bundle.sdk_source_dir if bundle is not None else None
    project_name = read_workflow_project_name(bundle.bundle_path)

    refreshed: list[Any] = []
    for agent in agent_definitions:
        agent_type = getattr(agent, "agent_type", "")
        if not isinstance(agent_type, str) or not agent_type.endswith("-agent"):
            refreshed.append(agent)
            continue
        if agent_type == "clawcodex-overview":
            refreshed.append(agent)
            continue
        # Stage agents: prepend condensed SOP while keeping the hybrid body (contracts).
        if is_prefixed_stage_agent(agent_type, project_name):
            skill_name = agent_type_to_skill_name(agent_type, project_prefix=project_name)
            original_body = ""
            get_prompt = getattr(agent, "get_system_prompt", None)
            if callable(get_prompt):
                try:
                    original_body = get_prompt() or ""
                except TypeError:
                    original_body = get_prompt(**{}) or ""
            from extensions.sop_converter.workflow_mode.bridge.mcp_adapter import bridge_tool_name

            existing_tools = [t for t in (getattr(agent, "tools", None) or []) if isinstance(t, str)]
            stage_meta = read_workflow_stage_for_agent(bundle.bundle_path, agent_type) if bundle is not None else None
            stage_label = (
                str(stage_meta["name"])
                if isinstance(stage_meta, dict) and stage_meta.get("name")
                else infer_stage_label_from_skill(skill_name)
            )
            stage_id = stage_meta.get("id") if isinstance(stage_meta, dict) else None
            output_files = (
                [str(f) for f in stage_meta.get("output_files", []) if f] if isinstance(stage_meta, dict) else None
            )
            sop_prefix = stage_agent_sop_body(
                agent_type=agent_type,
                skill_name=skill_name,
                stage_label=stage_label,
                pipeline_tool=pick_pipeline_execute_tool(existing_tools),
                bridge_tool=bridge_tool_name(project_name or ""),
                sdk_source_dir=sdk_source_dir,
                stage_id=_coerce_stage_id(stage_id),
                output_files=output_files or None,
            )
            combined = f"{sop_prefix.strip()}\n\n{original_body.strip()}".strip()
            from extensions.sop_converter.workflow_mode.generator.tool_gen import (
                stage_agent_tool_names,
            )

            merged_tools = stage_agent_tool_names(existing_tools)
            refreshed.append(
                replace(
                    agent,
                    tools=merged_tools,
                    get_system_prompt=_make_system_prompt_fn(combined),
                )
            )
            continue
        skill_name = agent_type_to_skill_name(agent_type, project_prefix=project_name)
        body = domain_agent_sop_body(
            agent_type=agent_type,
            description=getattr(agent, "when_to_use", "") or "",
            skill_name=skill_name,
            sdk_source_dir=sdk_source_dir,
            bundle=bundle.bundle_path if bundle is not None else None,
        )
        tools = sorted(AgentToolConstants.POS_SOP_DOMAIN_AGENT_TOOLS)
        refreshed.append(
            replace(
                agent,
                tools=tools,
                get_system_prompt=_make_system_prompt_fn(body),
            )
        )
    return refreshed


def _make_system_prompt_fn(prompt_text: str):
    """Return a get_system_prompt callable with stable closure binding."""

    def _get_system_prompt(**_kwargs: Any) -> str:
        return prompt_text

    return _get_system_prompt
