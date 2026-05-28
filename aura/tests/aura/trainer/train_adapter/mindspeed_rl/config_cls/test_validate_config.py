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

import pytest
from unittest.mock import MagicMock


class TestValidateConfig:

    def test_validate_agent_rl_args(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.validate_config import validate_agent_rl_args

        actor_config = MagicMock()
        ref_config = MagicMock()
        reward_config = MagicMock()
        rl_config = MagicMock()
        generate_config = MagicMock()
        agentic_config = MagicMock()

        result = validate_agent_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config, agentic_config)
        assert result is None

    def test_validate_agent_rl_args_no_api_key(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.validate_config import validate_agent_rl_args

        actor_config = MagicMock()
        ref_config = MagicMock()
        reward_config = MagicMock()
        rl_config = MagicMock()
        generate_config = MagicMock()
        agentic_config = MagicMock()

        result = validate_agent_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config, agentic_config)
        assert result is None
