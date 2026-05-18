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

# patch_utils should be the first import, because it will be used by other
# patch files.
from . import patch_worker_v1
from . import patch_camem
from . import patch_schedule_config
from . import patch_model_runner_v1
from . import patch_qwen3_moe
from . import patch_scheduler
from . import patch_attention_mask
from . import patch_attention_v1
from . import patch_vllm_qwen3_moe
from . import patch_serving_completion
from . import patch_acl_graph
from . import patch_base
from . import patch_llmdatadist_c_mgr_connector
from . import patch_multiproc_executor
from . import patch_abstract
from . import patch_sampler
from . import patch_vllm_sampler
