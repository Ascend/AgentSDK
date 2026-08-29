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

"""Downstream agent extensions — registry, policy primitives, and bundled agents.

Importing this module triggers the eager registration of every bundled
agent in :mod:`clawcodex_ext.agent._bundled_agents` via their
``@AgentRegistry.register`` decorators, so any consumer that simply
imports ``clawcodex_ext.agent`` gets the full decoupled agent set.

The original background-runner / background-state exports from the
prior version of this package are preserved at the bottom of the
``__all__`` list for backwards compatibility.
"""

from __future__ import annotations

# Policy primitives (imported first so bundled agents can reference them).
from clawcodex_ext.agent.policy import (  # noqa: F401
    IDENTITY_CLAWCODEX_AGENT,
    IDENTITY_CODE_REVIEWER,
    IDENTITY_DOCS_WRITER,
    IDENTITY_READ_ONLY_EXPLORER,
    IDENTITY_SOFTWARE_ARCHITECT,
    IDENTITY_TEST_RUNNER,
    IDENTITY_WEB_RESEARCHER,
    NORM_CODE_AUTHOR,
    NORM_DIFF_FOCUSED,
    NORM_GIT_OPERATOR,
    NORM_READ_ONLY,
    NORM_WEB_RESEARCHER,
    TOOL_SET_AUTHOR,
    TOOL_SET_READ_ONLY,
    TOOL_SET_TESTING,
    TOOL_SET_WEB_ONLY,
    build_agent_prompt,
)

# Registry API.
from clawcodex_ext.agent.registry import (  # noqa: F401
    AgentRegistry,
    SOURCE_CLAWCODEX_EXT,
    SOURCE_EXTENSIONS,
    register,
)


# Markdown discovery stays lazy so importing ``clawcodex_ext.agent`` from
# inside ``src.agent`` does not pull ``src.agent.parse_agent_markdown`` while
# ``src.agent.agent_definitions`` is still initialising.
def discover_clawcodex_ext_agents(*args, **kwargs):
    from clawcodex_ext.agent.markdown_discovery import (
        discover_clawcodex_ext_agents as _impl,
    )

    return _impl(*args, **kwargs)


def discover_extension_agents(*args, **kwargs):
    from clawcodex_ext.agent.markdown_discovery import (
        discover_extension_agents as _impl,
    )

    return _impl(*args, **kwargs)


def ensure_bundled_agents_registered() -> None:
    from importlib import import_module, reload

    modules = (
        "clawcodex_ext.agent._bundled_agents.code_reviewer",
        "clawcodex_ext.agent._bundled_agents.docs_writer",
        "clawcodex_ext.agent._bundled_agents.test_runner",
    )
    for module_name in modules:
        module = import_module(module_name)
        reload(module)


# Downstream background runner / state (preserved from prior version).
from clawcodex_ext.agent.background_runner import (  # noqa: E402, F401
    launch_background_runner,
    get_background_runner_status,
)
from clawcodex_ext.agent.background_state import (  # noqa: E402, F401
    background_signal,
    is_backgrounded,
    set_backgrounded,
    signal_background,
    reset_background,
)


__all__ = [
    # Registry
    "AgentRegistry",
    "SOURCE_CLAWCODEX_EXT",
    "SOURCE_EXTENSIONS",
    "register",
    # Policy — identities
    "IDENTITY_CLAWCODEX_AGENT",
    "IDENTITY_CODE_REVIEWER",
    "IDENTITY_DOCS_WRITER",
    "IDENTITY_READ_ONLY_EXPLORER",
    "IDENTITY_SOFTWARE_ARCHITECT",
    "IDENTITY_TEST_RUNNER",
    "IDENTITY_WEB_RESEARCHER",
    # Policy — norms
    "NORM_CODE_AUTHOR",
    "NORM_DIFF_FOCUSED",
    "NORM_GIT_OPERATOR",
    "NORM_READ_ONLY",
    "NORM_WEB_RESEARCHER",
    # Policy — tool sets
    "TOOL_SET_AUTHOR",
    "TOOL_SET_READ_ONLY",
    "TOOL_SET_TESTING",
    "TOOL_SET_WEB_ONLY",
    "build_agent_prompt",
    # Discovery
    "discover_clawcodex_ext_agents",
    "discover_extension_agents",
    "ensure_bundled_agents_registered",
    # Background runner / state (preserved)
    "launch_background_runner",
    "get_background_runner_status",
    "background_signal",
    "is_backgrounded",
    "set_backgrounded",
    "signal_background",
    "reset_background",
]
