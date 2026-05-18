#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd.2023-2025. All rights reserved.

# 初始化mindspeed_rl目录

from mindspeed.op_builder import GMMOpBuilder
from mindspeed.ops.npu_groupmatmul_add import groupmatmul_add_op_builder
from mindspeed.ops.npu_matmul_add import matmul_add_op_builder
from mindspeed.ops.npu_moe_token_permute import moe_token_permute_op_builder
from mindspeed.ops.npu_moe_token_unpermute import moe_token_unpermute_op_builder
from mindspeed.ops.npu_rotary_position_embedding import rope_op_builder
from mindspeed.ops.npu_ring_attention_update import op_builder as ring_op_builder
from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder


if __name__ == "__main__":
    moe_token_permute_op_builder.load()
    moe_token_unpermute_op_builder.load()
    rope_op_builder.load()
    GMMOpBuilder().load()
    # GMMV2OpBuilder().load()
    groupmatmul_add_op_builder.load()
    matmul_add_op_builder.load()
    ring_op_builder.load()
    FusedAdamWV2OpBuilder().load()
