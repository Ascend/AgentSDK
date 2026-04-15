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

# [do not delete !!!]
from vllm_ascend.patch import platform
from vllm_ascend.patch import worker

# patch_utils should be the first import, because it will be used by other
# patch files.
from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_9_1 import patch_worker_v1
from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_9_1 import patch_camem
from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_9_1 import patch_attention
from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_9_1 import patch_attention_v1