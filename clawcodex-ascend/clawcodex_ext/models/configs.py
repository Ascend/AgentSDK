#!/usr/bin/env python3
# coding=utf-8

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

"""Extended model configurations — models not in the upstream base.

These configs are registered into ``MODEL_CONFIGS`` automatically when
``clawcodex_ext.models`` is imported (via the side-effect import in
``clawcodex_ext/models/__init__.py``).

To add a new model:

1. Define its ``ModelConfig`` here.
2. Add it to ``EXTRA_MODEL_CONFIGS``.
3. The import in ``clawcodex_ext/models/__init__.py`` will register it.
"""

from __future__ import annotations

from src.models.configs import ModelConfig

# ---------------------------------------------------------------------------
# Extra model configurations — keyed by model ID
# ---------------------------------------------------------------------------

EXTRA_MODEL_CONFIGS: dict[str, ModelConfig] = {
    # DeepSeek V4 series
    "deepseek-v4-flash": ModelConfig(
        model_id="deepseek-v4-flash",
        display_name="DeepSeek V4 Flash",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_thinking=False,
        supports_tools=True,
        supports_vision=False,
        cost_input_per_mtok=0.30,
        cost_output_per_mtok=0.60,
        cost_cache_create_per_mtok=0.30,
        cost_cache_read_per_mtok=0.075,
    ),
    "deepseek-v4-pro": ModelConfig(
        model_id="deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_thinking=True,
        supports_tools=True,
        supports_vision=False,
        cost_input_per_mtok=2.0,
        cost_output_per_mtok=8.0,
        cost_cache_create_per_mtok=2.0,
        cost_cache_read_per_mtok=0.50,
    ),
    # Gemini 2.5 Flash
    "gemini-2.5-flash": ModelConfig(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_thinking=False,
        supports_tools=True,
        supports_vision=True,
        cost_input_per_mtok=0.15,
        cost_output_per_mtok=0.60,
        cost_cache_create_per_mtok=0.10,
        cost_cache_read_per_mtok=0.025,
    ),
}


__all__ = [
    "EXTRA_MODEL_CONFIGS",
]
