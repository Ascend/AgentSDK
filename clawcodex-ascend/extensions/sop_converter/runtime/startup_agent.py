#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.

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

"""Build the main-loop AgentDefinition for ``--agent <bundle_dir>`` startup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extensions.capabilities.agent_definition_protocol import AgentToolConstants
from ..adapters import DEFAULTS


def build_bundle_overview_agent_definition(
    agent: dict[str, Any],
    *,
    bundle_dir: Path,
) -> Any:
    """Turn a parsed overview agent dict into the REPL main-loop definition.

    Uses :meth:`AgentToolConstants.registered_proxy_base_tools` so the
    session starts with the SOP Overview routing set (Skill, ToolSearch,
    Agent, Read, TodoWrite, StructuredOutput).
    """
    name = str(agent.get("name") or "clawcodex-overview")
    raw_skills = agent.get("skills")
    skills: list[str] | None = None
    if isinstance(raw_skills, list):
        skills = [s for s in raw_skills if isinstance(s, str)]

    return DEFAULTS.agent_definition_factory(
        agent_type=name,
        when_to_use=str(agent.get("description") or ""),
        tools=AgentToolConstants.registered_proxy_base_tools(),
        skills=skills,
        source="dynamic",
        base_dir=str(bundle_dir.resolve()),
        model=agent.get("model") if isinstance(agent.get("model"), str) else None,
        get_system_prompt=lambda **_kwargs: "",
    )
