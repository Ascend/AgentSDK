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

"""Built-in model-group templates."""

from __future__ import annotations

from .config import GroupConfig, SlotConfig

PRESETS: dict[str, GroupConfig] = {
    "quick-compare": GroupConfig(
        "parallel",
        (
            SlotConfig("sonnet", "anthropic", "claude-sonnet-4-6"),
            SlotConfig("gpt4o", "openai", "gpt-4o"),
            SlotConfig("deepseek", "deepseek", "deepseek-v4-flash"),
        ),
        aggregator="passthrough",
    ),
    "high-reliability": GroupConfig(
        "voting",
        (
            SlotConfig("sonnet", "anthropic", "claude-sonnet-4-6", weight=2),
            SlotConfig("gpt4o", "openai", "gpt-4o"),
            SlotConfig("deepseek", "deepseek", "deepseek-v4-flash"),
        ),
        aggregator="majority",
        min_votes=2,
    ),
    "budget-safe": GroupConfig(
        "fallback",
        (
            SlotConfig("primary", "deepseek", "deepseek-v4-flash"),
            SlotConfig("fallback1", "openai", "gpt-4o"),
            SlotConfig("fallback2", "anthropic", "claude-sonnet-4-6"),
        ),
    ),
}


def get_preset(name: str) -> GroupConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown preset '{name}'") from exc
