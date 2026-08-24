# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""
Agent Tool Configuration

Defines how tools are selected for an agent based on mode and bundles.
Used by ToolRegistryExt.get_tools_for_config() to filter tools.

Modes:
    - bare:  No tools (zero tool agent for pure reasoning)
    - default: Default bundle set (bash, edit, read, search)
    - all: All available tools
"""

from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .bundles import MODE_BUNDLES, ALL_BUNDLE_NAMES


class ToolMode(str, Enum):
    """Tool loading mode for an agent.

    ``str`` mixin so members compare equal to the plain-string mode
    values used by ``AgentToolConfigMode`` / ``AgentToolConfig.mode``
    (e.g. ``ToolMode.ALL == "all"`` is True).
    """

    BARE = "bare"
    DEFAULT = "default"
    ALL = "all"


# Type alias for clarity
AgentToolConfigMode = Literal["bare", "default", "all"]


@dataclass
class AgentToolConfig:
    """
    Configuration for which tools an agent can access.

    Attributes:
        mode: Tool loading mode (bare/default/all)
        bundles: Explicit list of bundle names to load (overrides mode default)
        exclude: Tool names to exclude from the selected set
    """

    mode: AgentToolConfigMode = "default"
    bundles: list[str] | None = None
    exclude: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate bundle names."""
        if self.bundles is not None:
            unknown = set(self.bundles) - set(ALL_BUNDLE_NAMES)
            if unknown:
                raise ValueError(f"unknown bundles: {sorted(unknown)}")

    def get_effective_bundle_names(self) -> list[str]:
        """Get the bundle names this config will use."""
        if self.bundles is not None:
            return self.bundles
        if self.mode == "bare":
            return []
        if self.mode == "all":
            return ALL_BUNDLE_NAMES
        return MODE_BUNDLES.get("default", [])


def load_tool_config(
    mode: AgentToolConfigMode | None = None,
    bundles: list[str] | None = None,
    exclude: list[str] | None = None,
) -> AgentToolConfig:
    """
    Factory function to create AgentToolConfig with validation.

    Args:
        mode: Tool mode (bare/default/all), defaults to "default"
        bundles: Explicit bundle list, overrides mode's default bundles
        exclude: Tool names to exclude

    Returns:
        Validated AgentToolConfig instance
    """
    return AgentToolConfig(
        mode=mode or "default",
        bundles=bundles,
        exclude=exclude or [],
    )
