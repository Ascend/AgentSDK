#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestMSRLConfigParser:
    @pytest.fixture
    def config(self):
        config = {
            "tokenizer_name_or_path": "/path/to/tokenizer",
            "model_name": "llama",
            "agent_name": "my_agent",
            "agent_engine_wrapper_path": "/path/to/wrapper",
            "train_backend": "mindspeed_rl",
            "mindspeed_rl": {
                "data_path": "/path/to/data",
                "load_params_path": "/path/to/load",
                "save_params_path": "/path/to/save",
            },
        }
        return config

    @pytest.fixture
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.config_cls": MagicMock(),
            "mindspeed_rl.config_cls.validate_config": MagicMock(),
            "mindspeed_rl.config_cls.generate_config": MagicMock(),
            "mindspeed_rl.config_cls.megatron_config": MagicMock(),
            "mindspeed_rl.config_cls.rl_config": MagicMock(),
        }):
            yield

    @pytest.fixture
    def patch_target(self, patch_modules):
        with (patch("mindspeed_rl.config_cls.generate_config.GenerateConfig", MagicMock()),
              patch("mindspeed_rl.config_cls.megatron_config.MegatronConfig", MagicMock()),
              patch("mindspeed_rl.config_cls.rl_config.RLConfig", MagicMock()),
              patch("mindspeed_rl.config_cls.validate_config.validate_rl_args", MagicMock())):
            yield

    def test_missing_required_key(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        del config["tokenizer_name_or_path"]
        with pytest.raises(ValueError) as context:
            parser = MSRLConfigParser(config)
            parser.process_config()

    def test_invalid_type(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["mindspeed_rl"]["train_iters"] = "not_an_int"
        with (patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid", return_value=True),
              patch("os.path.exists", return_value=True)):
            with pytest.raises(ValueError) as context:
                parser = MSRLConfigParser(config)
                parser.process_config()

    def test_invalid_path(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        with patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid",
                   side_effect=RuntimeError("Invalid path")):
            with pytest.raises(RuntimeError) as context:
                parser = MSRLConfigParser(config)
                parser.process_config()

        assert "Invalid path" == str(context.value)

    def test_negative_value(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["mindspeed_rl"]["train_iters"] = -1
        with (patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid", return_value=True),
              patch("os.path.exists", return_value=True)):
            with pytest.raises(ValueError) as context:
                parser = MSRLConfigParser(config)
                parser.process_config()

    def test_invalid_string_value(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["agent_name"] = "invalid@name"
        with (patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid", return_value=True),
              patch("os.path.exists", return_value=True)):
            with pytest.raises(ValueError) as context:
                parser = MSRLConfigParser(config)
                parser.process_config()

    def test_data_path_ends_with_data(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["mindspeed_rl"]["data_path"] = "/valid/path/to/data"
        with (patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid", return_value=True),
              patch("os.path.exists", return_value=True)):
            parser = MSRLConfigParser(config)
            parser._validate_config()

    def test_data_path_parent_directory_invalid(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["mindspeed_rl"]["data_path"] = "/valid/path/to/data"
        with patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid",
                   side_effect=RuntimeError("Invalid directory")):
            with pytest.raises(RuntimeError) as context:
                parser = MSRLConfigParser(config)
                parser._validate_config()

        assert "Invalid directory" == str(context.value)

    def test_check_data_path_is_valid_raises_error(self, config, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
        config = config.copy()
        config["mindspeed_rl"]["data_path"] = "/valid/path/to/data"
        with patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid",
                   side_effect=RuntimeError("Invalid data path")):
            with pytest.raises(RuntimeError) as context:
                parser = MSRLConfigParser(config)
                parser._validate_config()

        assert "Invalid data path" == str(context.value)
