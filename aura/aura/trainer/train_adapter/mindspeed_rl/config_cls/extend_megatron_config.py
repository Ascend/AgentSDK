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


from typing import Dict

from mindspeed_rl import MegatronConfig


class ExtendMegatronConfig(MegatronConfig):

    def __init__(self, training_config: Dict, model_config: Dict):
        defaults = {
            "test_data_path": None,
            "train_data_path": None,
            "swap_optimizer_times": 4,
            "fix_router": False,
        }
        for key, value in defaults.items():
            setattr(self, key, value)
        super().__init__(training_config, model_config)
        for key, value in defaults.items():
            setattr(self, key, training_config.get(key, value))
