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


class TestExtendedGenerateConfig:

    def test_default_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
        config = ExtendedGenerateConfig({})
        assert config.base_url == ""
        assert config.api_key == "empty"
        assert config.train_backend == "mindspeed_rl"
        assert config.enable_sleep_mode == False
        assert config.load_format == "megatron"
        assert config.agent_engine == "rllm"
        assert config.infer_backend == "vllm"

    def test_validate_sampling(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
        config = ExtendedGenerateConfig({})
        assert config.validate_sampling == {"max_tokens": 8192, "top_p": 0.5, "top_k": 50, "min_p": 0.01, "temperature": 0.2}

    def test_hybrid_params(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
        config = ExtendedGenerateConfig({})
        assert config.hybrid_batch_num == 1
        assert config.enable_version_control == False
        assert config.use_on_policy == False

    def test_prefill_params(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
        config = ExtendedGenerateConfig({})
        assert config.prefill_enforce_eager is None
        assert config.prefill_max_num_seqs is None
        assert config.prefill_max_num_batched_tokens is None

    def test_custom_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_generate import ExtendedGenerateConfig
        config = ExtendedGenerateConfig({"base_url": "http://test.com", "api_key": "test_key"})
        assert config.base_url == "http://test.com"
        assert config.api_key == "test_key"
