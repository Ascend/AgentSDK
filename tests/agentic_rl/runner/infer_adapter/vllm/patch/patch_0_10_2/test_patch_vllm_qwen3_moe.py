#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib.util

test_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
sys.path.append(project_root)


class TestPatchVllmQwen3Moe(unittest.TestCase):
    """Test patch_vllm_qwen3_moe.py module"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.Tensor = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_model_executor = MagicMock()
        cls.mock_models = MagicMock()

        class MockParameter:
            def __init__(self, name):
                self.name = name

        class MockQwen3MoeModel:
            def __init__(self):
                self.params = {
                    "qkv_proj": MockParameter("qkv_proj"),
                    "gate_up_proj": MockParameter("gate_up_proj"),
                    "mlp.experts.0.gate_up_proj": MockParameter("mlp.experts.0.gate_up_proj"),
                    "regular_param": MockParameter("regular_param"),
                    "attn.kv_scale": MockParameter("attn.kv_scale")
                }

            def named_parameters(self):
                return self.params.items()

            def get_expert_mapping(self):
                return [
                    ("mlp.experts.0.gate_up_proj", "mlp.experts.0.gate_proj", 0, 0),
                    ("mlp.experts.0.gate_up_proj", "mlp.experts.0.up_proj", 0, 1)
                ]

        cls.MockQwen3MoeModel = MockQwen3MoeModel

        cls.mock_qwen3_moe = MagicMock()
        cls.mock_qwen3_moe.Qwen3MoeModel = MockQwen3MoeModel

        cls.mock_utils = MagicMock()
        cls.mock_utils.is_pp_missing_parameter.return_value = False

        cls.mock_model_loader = MagicMock()

        cls.mock_weight_utils = MagicMock()
        cls.mock_weight_utils.default_weight_loader = MagicMock()
        cls.mock_weight_utils.maybe_remap_kv_scale_name.return_value = None

        cls.mock_logger = MagicMock()
        cls.mock_logger.warning_once = MagicMock()

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch.Tensor': cls.mock_torch.Tensor,
            'torch.nn': MagicMock(),
            'vllm': cls.mock_vllm,
            'vllm.model_executor': cls.mock_model_executor,
            'vllm.model_executor.models': cls.mock_models,
            'vllm.model_executor.models.qwen3_moe': cls.mock_qwen3_moe,
            'vllm.model_executor.models.utils': cls.mock_utils,
            'vllm.model_executor.model_loader': cls.mock_model_loader,
            'vllm.model_executor.model_loader.weight_utils': cls.mock_weight_utils,
            'vllm.logger': cls.mock_logger,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_vllm_qwen3_moe.py')
        spec = importlib.util.spec_from_file_location('patch_vllm_qwen3_moe', module_path)
        cls.patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.patch_module)

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.mock_utils.is_pp_missing_parameter.reset_mock()
        self.mock_utils.is_pp_missing_parameter.return_value = False
        self.mock_weight_utils.default_weight_loader.reset_mock()
        self.mock_weight_utils.maybe_remap_kv_scale_name.reset_mock()
        self.mock_weight_utils.maybe_remap_kv_scale_name.return_value = None
        self.mock_logger.warning_once.reset_mock()

        self.mock_tensor = MagicMock()
        self.model_instance = self.MockQwen3MoeModel()

    def test_load_weights_patch_stacked_params(self):
        pass

    def test_load_weights_patch_expert_params(self):
        mock_weight_loader = MagicMock(return_value=True)
        self.model_instance.params["mlp.experts.0.gate_up_proj"].weight_loader = mock_weight_loader

        weights = [
            ("mlp.experts.0.gate_proj", self.mock_tensor),
            ("mlp.experts.0.up_proj", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(set(result), {"mlp.experts.0.gate_up_proj"})
        self.assertEqual(mock_weight_loader.call_count, 2)

    def test_load_weights_patch_regular_params(self):
        pass

    def test_load_weights_patch_ignore_suffixes(self):
        weights = [
            ("param.bias", self.mock_tensor),
            ("param_k_scale", self.mock_tensor),
            ("param.v_scale", self.mock_tensor),
            ("param.weight_scale", self.mock_tensor),
            ("param.input_scale", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(len(result), 0)

    def test_load_weights_patch_missing_param(self):
        weights = [
            ("non_existent_param", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(len(result), 0)

    def test_load_weights_patch_pp_missing_param(self):
        self.mock_utils.is_pp_missing_parameter.return_value = True

        weights = [
            ("regular_param", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(len(result), 0)

    def test_load_weights_patch_kv_scale(self):
        weights = [
            ("model.layers.0.kv_scale", self.mock_tensor)
        ]

        remapped_name = "model.layers.0.attn.kv_scale"
        self.model_instance.params[remapped_name] = MagicMock()

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(set(result), {remapped_name})

    def test_load_weights_patch_custom_weight_loader(self):
        mock_weight_loader = MagicMock()
        self.model_instance.params["qkv_proj"].weight_loader = mock_weight_loader

        weights = [
            ("q_proj", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(set(result), {"qkv_proj"})
        mock_weight_loader.assert_called_once()

    def test_load_weights_patch_expert_weight_not_mapped(self):
        mock_weight_loader = MagicMock(return_value=False)
        self.model_instance.params["mlp.experts.0.gate_up_proj"].weight_loader = mock_weight_loader

        weights = [
            ("mlp.experts.0.gate_proj", self.mock_tensor)
        ]

        result = self.patch_module.load_weights_patch(self.model_instance, weights)

        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
