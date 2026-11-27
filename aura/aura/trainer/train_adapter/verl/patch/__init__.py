#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# patch不生效, 先注释掉, 通过启动脚本sed修改
# import verl.workers.engine.utils as utils
# from .engine_utils_patch import postprocess_batch_func_patch
# utils.postprocess_batch_func = postprocess_batch_func_patch

# patch不生效, 先注释掉, 通过启动脚本sed修改
# import verl.workers.utils.padding as padding
# from .padding_patch import no_padding_2_padding_patch
# padding.no_padding_2_padding = no_padding_2_padding_patch

# from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
# from .fsdp_transformer_impl_patch import prepare_model_outputs_patch
# FSDPEngineWithLMHead.prepare_model_outputs = prepare_model_outputs_patch
#
# import verl.utils.torch_functional as torch_functional
# from .torch_functional_patch import logprobs_from_logits_torch_npu_patch
# torch_functional.logprobs_from_logits_torch_npu = logprobs_from_logits_torch_npu_patch
