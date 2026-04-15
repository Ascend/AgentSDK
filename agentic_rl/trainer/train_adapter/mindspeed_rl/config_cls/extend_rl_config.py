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


from mindspeed_rl import RLConfig


class ExtendedRLConfig(RLConfig):

    def __init__(self, config_dict):
        self.validate_freq = 10
        self.validate_num_samples = 100
        self.test_before_train = False
        self.test_only = False
        self.validate_n_samples = 1
        self.simplify_think_content = False
        self.mock_rollout = False
        self.mock_prompt_mean = 500
        self.mock_prompt_gap = 200
        self.mock_response_mean = 1000
        self.mock_response_gap = 400
        self.mock_eos_token_id = 151643
        self.ref_max_packing_token_size = None

        super().__init__(config_dict)
        if self.ref_max_packing_token_size is None:
            self.ref_max_packing_token_size = self.max_packing_token_size
