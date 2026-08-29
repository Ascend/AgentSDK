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

"""Feature-gate helpers for upstream-compatible goals."""

from __future__ import annotations

from clawcodex_ext.feature_gate import FeatureFlag, get_registry

GOALS_FEATURE = "goals"


def ensure_goals_feature_registered() -> None:
    """Register the goals gate if a fresh registry does not have it yet."""
    registry = get_registry()
    if registry.get_flag(GOALS_FEATURE) is None:
        registry.register(
            FeatureFlag(
                name=GOALS_FEATURE,
                default=True,
                description="Enable upstream-compatible /goal mode",
            )
        )


def goal_enabled() -> bool:
    """Return whether the upstream-compatible goal surface is enabled."""
    ensure_goals_feature_registered()
    return get_registry().is_enabled(GOALS_FEATURE)


__all__ = ["GOALS_FEATURE", "ensure_goals_feature_registered", "goal_enabled"]
