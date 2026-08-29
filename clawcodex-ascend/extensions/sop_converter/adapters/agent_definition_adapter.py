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

"""Default adapter for :class:`AgentDefinitionProtocol`.

Wraps ``clawcodex_ext.agent.agent_definitions.AgentDefinition`` as a
factory function so the SOP converter can construct agent definitions
without importing ``clawcodex_ext`` directly.

Field names are already 1:1 between the upstream dataclass and the
Protocol, so no property aliasing is needed — the factory is a simple
``**kwargs`` passthrough.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.4.
"""

from __future__ import annotations

from typing import Any

from extensions.capabilities.agent_definition_protocol import (
    AgentDefinitionProtocol,
)

__all__ = [
    "default_agent_definition_factory",
    "default_agent_loader",
]


def default_agent_definition_factory(**kwargs: Any) -> AgentDefinitionProtocol:
    """Construct an ``AgentDefinition``-compatible instance.

    Accepts the same keyword arguments as
    ``clawcodex_ext.agent.agent_definitions.AgentDefinition``.

    All keyword arguments are forwarded verbatim; no field aliasing is
    needed because the upstream dataclass field names match the Protocol
    exactly.
    """
    from clawcodex_ext.agent.agent_definitions import AgentDefinition

    return AgentDefinition(**kwargs)


def default_agent_loader() -> list[AgentDefinitionProtocol]:
    """Return all known agent definitions.

    Wraps
    ``clawcodex_ext.agent.load_agents_dir.get_agent_definitions_with_overrides``
    using the current working directory as the root.
    """
    import os

    from clawcodex_ext.agent.load_agents_dir import (
        get_agent_definitions_with_overrides,
    )

    return list(get_agent_definitions_with_overrides(os.getcwd()))
