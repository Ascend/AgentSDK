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

from rllm.rewards.reward_types import RewardOutput, RewardConfig
from agents.search_r1_agent.reward.search_r1_reward import SearchR1ResRewardFn


def search_r1_res_reward_fn(action: str, task_info=None) -> RewardOutput:
    reward_config = RewardConfig()
    reward_fn = SearchR1ResRewardFn(reward_config)
    return reward_fn(action, task_info)
