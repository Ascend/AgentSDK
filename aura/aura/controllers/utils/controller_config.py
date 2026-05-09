#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

          http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os


class ControllerConfig:
    def __init__(self):
        self.controller_base_port = int(os.getenv("CONTROLLER_BASE_PORT", "4001"))
        self.rollout_server_addr = os.getenv("ROLLOUT_NODE", "0.0.0.0") + f":{self.controller_base_port}"
        self.train_server_addr = os.getenv("TRAIN_NODE", "0.0.0.0") + f":{self.controller_base_port + 1}"
