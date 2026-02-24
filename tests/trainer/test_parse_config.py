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

import unittest
from unittest.mock import patch
from agentic_rl.trainer.train_adapter.parse_config import ConfigParser
from agentic_rl.trainer.train_adapter.schema import GlobalConfig
 
 
class TestConfigParser(unittest.TestCase):
    def setUp(self):
        self.valid_verl_config = {
            "tokenizer_name_or_path": "/path/to/tokenizer",
            "model_name": "llama",
            "agent_name": "my_agent",
            "agent_engine_wrapper_path": "/path/to/wrapper",
            "train_backend": "verl",
            "verl": {
                "train_files": "/path/to/train",
                "val_files": "/path/to/val",
                "total_epochs": 2,
            },
        }
        self.valid_msrl_config = {
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
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_initialization(self, mock_check):
        parser = ConfigParser(self.valid_verl_config)
        self.assertEqual(parser.raw_config, self.valid_verl_config)
        self.assertIsNone(parser.global_config)
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_success(self, mock_check):
        parser = ConfigParser(self.valid_verl_config)
        global_config = parser._validate_config()
        self.assertIsInstance(global_config, GlobalConfig)
        self.assertEqual(global_config.train_backend, "verl")
        self.assertEqual(global_config.verl.total_epochs, 2)
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_missing_verl_config(self, mock_check):
        # Missing 'verl' key for 'verl' backend
        config = self.valid_verl_config.copy()
        del config["verl"]
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("verl config section is required", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_field_type(self, mock_check):
        config = self.valid_verl_config.copy()
        config["gpu_memory_utilization"] = 1.5  # Should be <= 1.0
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_process_config(self, mock_check):
        parser = ConfigParser(self.valid_verl_config)
        processed = parser.process_config()
        self.assertIsInstance(processed, dict)
        self.assertEqual(processed["train_backend"], "verl")
        self.assertEqual(processed["verl"]["train_files"], "/path/to/train")
 
    @patch("os.path.exists")
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_mindspeed_rl_success(self, mock_check, mock_exists):
        mock_exists.return_value = True
        parser = ConfigParser(self.valid_msrl_config)
        global_config = parser._validate_config()
        self.assertIsInstance(global_config, GlobalConfig)
        self.assertEqual(global_config.train_backend, "mindspeed_rl")
        self.assertIsNotNone(global_config.mindspeed_rl)
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_missing_mindspeed_rl_config(self, mock_check):
        # Missing 'mindspeed_rl' key for 'mindspeed_rl' backend
        config = {
            "tokenizer_name_or_path": "/path/to/tokenizer",
            "model_name": "llama",
            "agent_name": "my_agent",
            "agent_engine_wrapper_path": "/path/to/wrapper",
            "train_backend": "mindspeed_rl",
        }
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("mindspeed_rl config section is required", str(context.exception))
 
    def test_validate_config_missing_required_field(self):
        # Missing 'tokenizer_name_or_path' which is required
        config = {
            "model_name": "llama",
            "agent_name": "my_agent",
            "agent_engine_wrapper_path": "/path/to/wrapper",
            "train_backend": "verl",
        }
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_extra_field_forbidden(self, mock_check):
        # Extra field 'unknown_field' should be rejected due to extra="forbid"
        config = self.valid_verl_config.copy()
        config["unknown_field"] = "some_value"
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_top_p(self, mock_check):
        # top_p must be between 0 and 1
        config = self.valid_verl_config.copy()
        config["top_p"] = 1.5
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_min_p(self, mock_check):
        # min_p must be between 0 and 1
        config = self.valid_verl_config.copy()
        config["min_p"] = -0.1
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_lr_warmup_fraction(self, mock_check):
        # lr_warmup_fraction must be between 0 and 1
        config = self.valid_verl_config.copy()
        config["lr_warmup_fraction"] = 1.5
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_num_gpus_per_node(self, mock_check):
        # num_gpus_per_node must be positive
        config = self.valid_verl_config.copy()
        config["num_gpus_per_node"] = 0
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_config_invalid_rollout_n(self, mock_check):
        # rollout_n must be positive
        config = self.valid_verl_config.copy()
        config["rollout_n"] = -1
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_raw_config_is_deep_copied(self, mock_check):
        # Verify that modifying the original config doesn't affect the parser
        original_config = {
            "tokenizer_name_or_path": "/path/to/tokenizer",
            "model_name": "llama",
            "agent_name": "my_agent",
            "agent_engine_wrapper_path": "/path/to/wrapper",
            "train_backend": "verl",
            "verl": {
                "train_files": "/path/to/train",
                "val_files": "/path/to/val",
                "total_epochs": 2,
            },
        }
        parser = ConfigParser(original_config)
        # Modify the original config after creating parser
        original_config["model_name"] = "modified_model"
        original_config["verl"]["total_epochs"] = 100
        # Verify parser's raw_config is unchanged
        self.assertEqual(parser.raw_config["model_name"], "llama")
        self.assertEqual(parser.raw_config["verl"]["total_epochs"], 2)
 
    @patch("agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid")
    def test_validate_verl_config_invalid_total_epochs(self, mock_check):
        # total_epochs in VerlConfig must be positive
        config = self.valid_verl_config.copy()
        config["verl"] = self.valid_verl_config["verl"].copy()
        config["verl"]["total_epochs"] = 0
        parser = ConfigParser(config)
        with self.assertRaises(ValueError) as context:
            parser._validate_config()
        self.assertIn("validation failed", str(context.exception))
 
 
if __name__ == "__main__":
    unittest.main()