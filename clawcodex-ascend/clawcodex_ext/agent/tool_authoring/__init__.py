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

"""Tool authoring — enables agents to dynamically create tools.

This module provides the infrastructure for the ``CreateAgentTool`` meta-tool:
agent-authored tools are defined by an ``AgentToolSpec``, validated for security,
built into ``Tool`` objects, and registered at runtime and on disk.

Architecture::

    AgentToolSpec        # dataclass describing the tool
         │
         ▼
    validate_spec()      # security checks (validators.py)
         │
         ▼
    build_tool_from_spec()  # factory.py → Tool
         │
         ▼
    registry_ext.add_tool()  # runtime registry
    persistence.save_spec()  # disk persistence

Call handlers (call_handlers/) dispatch to the actual implementation:
    - ``execute_bash`` — subprocess with timeout, command whitelist
    - ``execute_http`` — urllib with method/url whitelist
    - ``execute_python`` — registered function dispatch
"""

from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.agent.tool_authoring.validators import (
    ValidationError,
    validate_spec,
    register_python_function,
    list_python_functions,
)
from clawcodex_ext.agent.tool_authoring.factory import (
    build_tool_from_spec,
    create_and_validate,
)
from clawcodex_ext.agent.tool_authoring.registry_ext import (
    add_tool,
    get_tool,
    list_tools,
    remove_tool,
    clear,
)
from clawcodex_ext.agent.tool_authoring.persistence import (
    save_spec,
    load_spec,
    delete_spec,
    list_persisted_specs,
    clear_persisted,
    TOOL_DIR,
)

__all__ = [
    "AgentToolSpec",
    "ValidationError",
    "validate_spec",
    "build_tool_from_spec",
    "create_and_validate",
    "add_tool",
    "get_tool",
    "list_tools",
    "remove_tool",
    "clear",
    "save_spec",
    "load_spec",
    "delete_spec",
    "list_persisted_specs",
    "clear_persisted",
    "register_python_function",
    "list_python_functions",
    "TOOL_DIR",
]
