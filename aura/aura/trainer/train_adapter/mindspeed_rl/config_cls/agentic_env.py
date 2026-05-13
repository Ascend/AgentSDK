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


import datasets

import third_party.rl
from mindspeed_rl.config_cls.base_config import BaseConfig


class AgenticEnvConfig(BaseConfig):

    def __init__(self, config_dict):
        self.namespace = "agentic_raygroup"
        self.rollout_output_path = "./outputs"
        self.agent_name = "netopt" # TODO NET optimization needs to be adapted
        self.tool_url = None
        self.mcp_server_url = None
        self.max_steps = 5
        self.max_tool_length = 4096
        self.use_sse = False
        self.mcp_server_command: str | None = None
        self.mcp_server_args: list[str] | None = None
        self.mcp_server_env: dict[str, str] | None = None
        self.tool_timeout = 2000
        self.trajectory_timeout = 7200

        self.update(config_dict)
