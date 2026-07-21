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
        defaults = {
            "validate_freq": 10,
            "validate_num_samples": 100,
            "test_before_train": False,
            "test_only": False,
            "validate_n_samples": 1,
            "simplify_think_content": False,
            "use_stepwise_advantage": False,
            "stepwise_advantage_mode": "immediate_reward_centered_scaled",
            "stepwise_advantage_beta": 1.6,
            "mock_rollout": False,
            "mock_prompt_mean": 500,
            "mock_prompt_gap": 200,
            "mock_response_mean": 1000,
            "mock_response_gap": 400,
            "mock_eos_token_id": 151643,
            "ref_max_packing_token_size": None,
            "use_tensorboard": False,
            "tensorboard_dir": "",
        }
        for key, value in defaults.items():
            setattr(self, key, value)

        super().__init__(config_dict)
        for key, value in defaults.items():
            setattr(self, key, config_dict.get(key, value))
        if self.ref_max_packing_token_size is None:
            self.ref_max_packing_token_size = self.max_packing_token_size
