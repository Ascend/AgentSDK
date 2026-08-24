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
Skills Extension Layer (skills_ext)

Extends upstream skills loader with clawcodex-specific functionality.
Follows the same pattern as tool_system_ext for consistency.

Architecture:
    src/skills/           - Layer 1: Upstream original code (read-only)
    src/skills_ext/       - Layer 2: clawcodex extension layer (new)

Extension components:
    - SkillRegistryExt: Wrapper for upstream loader with bundle support
    - bundles.py: Skill bundle definitions
    - agent_config.py: Per-agent skill configuration
    - paths.py: clawcodex-specific path resolution
    - hooks.py: Skill lifecycle callbacks
    - cache.py: Extension layer caching
    - bundled/: clawcodex-native bundled skills (SOP converter, etc.)
"""

from .registry_ext import SkillRegistryExt, SkillRegistrationCallback

__all__ = [
    "SkillRegistryExt",
    "SkillRegistrationCallback",
    "init_skill_catalog_extensions",
]


def init_skill_catalog_extensions() -> bool:
    """Ensure extension-owned prompt skills exist in the canonical catalog."""
    from clawcodex_ext.skills.bundled_skills import get_registered_bundled_skills

    if any(skill.name == "convert-sop-to-agent" for skill in get_registered_bundled_skills()):
        return True

    from .bundled import register_convert_sop_skill

    try:
        return register_convert_sop_skill()
    except Exception:
        return False


def init_skills_ext() -> None:
    """Initialize skills_ext bundled skills.

    Registers clawcodex-native skills that are not part of upstream.
    Each skill is registered independently — one failure does not
    block others.
    """
    from .bundled import register_convert_sop_skill, register_dream_skill

    import logging

    logger = logging.getLogger(__name__)
    try:
        register_convert_sop_skill()
    except Exception as exc:
        logger.warning("failed to register convert-sop-to-agent: %s", exc)

    try:
        register_dream_skill()
    except Exception as exc:
        logger.warning("failed to register dream skill: %s", exc)
