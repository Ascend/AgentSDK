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


from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.validate_config import validate_rl_args

from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls.agentic_env import AgenticEnvConfig


def validate_agent_rl_args(
        actor_config: MegatronConfig,
        ref_config: MegatronConfig,
        reward_config: MegatronConfig,
        rl_config: RLConfig,
        generate_config: GenerateConfig,
        agentic_config: AgenticEnvConfig,
):
    validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config)

    # if agentic_config.tool_url and agentic_config.agent_name != "dtn":
    #     raise ValueError(f"agentic_config.tool_url not none and agentic_config.agent_name should be dtn.")


