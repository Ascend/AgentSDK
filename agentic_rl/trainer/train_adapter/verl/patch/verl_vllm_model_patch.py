# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.

VLLM model patch for unsupported model types.

This module provides mock modules for VLLM model types that are not available in the current environment.
The patch is applied lazily (only when explicitly called) to avoid global side effects during module import.
-------------------------------------------------------------------------
"""

import types
import sys

# Guard to ensure patch is applied only once
_PATCH_APPLIED = False


def apply_vllm_model_patch():
    """
    Apply patch for VLLM model types that are not available in the current environment.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from vllm.model_executor import models

    # Define the mock modules to create
    mock_modules = [
        "deepseek_v2",
        "deepseek_mtp",
        "mixtral",
        "qwen2_moe",
        "qwen3_moe",
        "kimi_vl",
    ]
    # Create mock modules
    for module_name in mock_modules:
        full_module_path = f"vllm.model_executor.models.{module_name}"
        mock_module = types.ModuleType(full_module_path)
        setattr(models, module_name, mock_module)
        sys.modules[full_module_path] = mock_module

    # Define mock classes for deepseek_v2
    models.deepseek_v2.DeepseekV2ForCausalLM = object
    models.deepseek_v2.DeepseekV3ForCausalLM = object
    models.deepseek_v2.yarn_get_mscale = object
    models.deepseek_v2.DeepseekV2Attention = object
    models.deepseek_v2.DeepseekV2DecoderLayer = object
    models.deepseek_v2.DeepseekV2MLAAttention = object

    # Define mock classes for deepseek_mtp
    models.deepseek_mtp.DeepSeekMTP = object
    models.deepseek_mtp.DeepSeekMultiTokenPredictor = object
    models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer = object
    models.deepseek_mtp.SharedHead = object

    # Define mock classes for other models
    models.mixtral.MixtralForCausalLM = object
    models.qwen2_moe.Qwen2MoeForCausalLM = object
    models.qwen2_moe.FusedMoE = object
    models.qwen3_moe.Qwen3MoeForCausalLM = object
    models.kimi_vl.KimiVLForConditionalGeneration = object

    _PATCH_APPLIED = True
