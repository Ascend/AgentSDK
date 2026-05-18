#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

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
