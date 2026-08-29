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

"""Load per-stage agent markdown from a SOP convert bundle."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)


def register_bundle_agents(bundle_path: Path) -> list[str]:
    """Parse ``.claude/agents/*.md`` in *bundle_path* into AgentRegistry."""
    bundle_path = bundle_path.resolve()
    agents_dir = bundle_path / ".claude" / "agents"
    if not agents_dir.is_dir():
        return []

    from clawcodex_ext.agent.parse_agent_markdown import parse_agent_from_markdown
    from clawcodex_ext.agent.registry import AgentRegistry, SOURCE_EXTENSIONS
    from ..adapters import DEFAULTS

    registered: list[str] = []
    for md_path in sorted(agents_dir.glob("*.md")):
        if not md_path.is_file():
            continue
        try:
            parsed = DEFAULTS.frontmatter_parser(md_path.read_text(encoding="utf-8"))
            agent = parse_agent_from_markdown(
                file_path=str(md_path),
                frontmatter=parsed.frontmatter,
                body=parsed.body,
                source="project",
                base_dir=str(bundle_path),
            )
        except Exception as exc:
            logger.warning("Skip bundle agent %s: %s", md_path.name, exc)
            continue
        if agent is None:
            continue
        agent = replace(agent, source=SOURCE_EXTENSIONS, base_dir=str(bundle_path))
        AgentRegistry.register_definition(agent)
        registered.append(agent.agent_type)
        logger.info("Registered bundle stage agent: %s", agent.agent_type)

    if registered:
        try:
            DEFAULTS.clear_sop_caches()
        except Exception as exc:  # nosec
            logger.warning("Failed to clear SOP caches after bundle agent registration: %s", exc)

    return sorted(set(registered))
