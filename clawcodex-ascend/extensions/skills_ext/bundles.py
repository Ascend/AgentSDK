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
Skill Bundle Definitions

Defines skill loading modes for agents:
- bare: No skills (pure reasoning agent)
- default: Default skill bundle
- clawcodex: All clawcodex native built-in skills
- all: All available skills

Mirrors the TOOL_BUNDLES pattern from tool_system_ext.
"""

from __future__ import annotations


# Bundle definitions: bundle_name -> list of skill names
SKILL_BUNDLES: dict[str, list[str]] = {
    "default": [
        # Default skills available to all agents
        "git:commit",
        "git:push",
        "review-pr",
        "simplify",
        "debug",
    ],
    "clawcodex": [
        # All clawcodex native built-in skills
        "simplify",
        "debug",
        "loop",
        "verify-content",
        "pr-review",
        "code-review",
        "feature-dev",
        "dream",
        "keybindings-help",
        "update-config",
        "cron-list",
        "cron-delete",
        "stuck",
        "ask",
        "convert-sop-to-agent",
    ],
}

# Mode to bundle names mapping
MODE_BUNDLES: dict[str, list[str]] = {
    "bare": [],
    "default": ["default"],
    "clawcodex": ["clawcodex"],
    "all": list(SKILL_BUNDLES.keys()),
}

# All available bundle names
ALL_BUNDLE_NAMES: list[str] = list(SKILL_BUNDLES.keys())


def get_bundle_skills(bundle_name: str) -> list[str]:
    """Get skill names for a bundle, returns empty list if bundle not found."""
    return list(SKILL_BUNDLES.get(bundle_name, []))


def get_all_bundle_skills() -> list[str]:
    """Get all skill names across all bundles (deduped)."""
    seen: set[str] = set()
    result: list[str] = []
    for skills in SKILL_BUNDLES.values():
        for s in skills:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result
