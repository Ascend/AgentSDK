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
class WebWalkerConfig:
    def __init__(self):
        # Tree / beam stepwise (reward_fn TOOLS_RETURN, reward_mode=step)
        self.explore_reward_pos = 1.0

        # Chain trajectory-level (env_utils._compute_webwalker_chain_reward)
        self.chain_success_reward = 1.0   # reached source_url (answer page)
        self.chain_failure_reward = 0.0


_DEFAULT_CONFIG: WebWalkerConfig | None = None


def get_webwalker_reward_config() -> WebWalkerConfig:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = WebWalkerConfig()
    return _DEFAULT_CONFIG
