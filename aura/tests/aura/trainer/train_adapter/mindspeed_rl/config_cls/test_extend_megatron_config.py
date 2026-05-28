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
from unittest.mock import MagicMock, patch


class TestExtendMegatronConfig:

    def test_default_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_megatron_config import ExtendMegatronConfig
        config = ExtendMegatronConfig({}, {})
        assert config.test_data_path is None
        assert config.train_data_path is None
        assert config.swap_optimizer_times == 4
        assert config.fix_router is False

    def test_custom_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_megatron_config import ExtendMegatronConfig
        config = ExtendMegatronConfig({"test_data_path": "/path/to/test"}, {})
        assert config.test_data_path == "/path/to/test"
