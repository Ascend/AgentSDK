#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import third_party.rl.mindspeed_rl
import torch
import torch_npu
from mindspeed.op_builder import GMMOpBuilder, GMMV2OpBuilder
from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder
from mindspeed.ops.npu_groupmatmul_add import groupmatmul_add_op_builder
from mindspeed.ops.npu_matmul_add import matmul_add_op_builder
from mindspeed.ops.npu_moe_token_permute import moe_token_permute_op_builder
from mindspeed.ops.npu_moe_token_unpermute import moe_token_unpermute_op_builder
from mindspeed.ops.npu_ring_attention_update import op_builder as ring_op_builder
from mindspeed.ops.npu_rotary_position_embedding import rope_op_builder

if __name__ == '__main__':
    moe_token_permute_op_builder.load()
    moe_token_unpermute_op_builder.load()
    rope_op_builder.load()
    GMMOpBuilder().load()
    # GMMV2OpBuilder().load()
    groupmatmul_add_op_builder.load()
    matmul_add_op_builder.load()
    ring_op_builder.load()
    FusedAdamWV2OpBuilder().load()