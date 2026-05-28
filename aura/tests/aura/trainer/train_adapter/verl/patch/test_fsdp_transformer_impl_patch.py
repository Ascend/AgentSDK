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
import torch
from unittest.mock import MagicMock, patch
from tensordict import TensorDict
import sys
from types import ModuleType


class TestFSDPTransformerImplPatch:

    def setup_method(self):
        """Setup mock modules for verl dependencies."""
        self.verl_mock = ModuleType('verl')
        self.verl_utils_mock = ModuleType('verl.utils')
        self.verl_dataset_mock = ModuleType('verl.utils.dataset')
        self.verl_dataset_utils_mock = ModuleType('verl.utils.dataset.dataset_utils')
        self.verl_torch_func_mock = ModuleType('verl.utils.torch_functional')
        self.verl_ulysses_mock = ModuleType('verl.utils.ulysses')

        class MockDatasetPadMode:
            NO_PADDING = 0
            FIXED = 1

        self.MockDatasetPadMode = MockDatasetPadMode
        self.verl_dataset_utils_mock.DatasetPadMode = MockDatasetPadMode
        self.verl_utils_mock.tensordict_utils = MagicMock()
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(100))
        self.verl_torch_func_mock.entropy_from_logits = MagicMock(return_value=torch.randn(2, 10))
        self.verl_ulysses_mock.gather_outputs_and_unpad = MagicMock(return_value=torch.randn(50))

        self.verl_dataset_mock.dataset_utils = self.verl_dataset_utils_mock
        self.verl_utils_mock.dataset = self.verl_dataset_mock
        self.verl_utils_mock.torch_functional = self.verl_torch_func_mock
        self.verl_utils_mock.ulysses = self.verl_ulysses_mock
        self.verl_mock.utils = self.verl_utils_mock

        self.modules_to_clean = []
        for name, module in [
            ('verl', self.verl_mock),
            ('verl.utils', self.verl_utils_mock),
            ('verl.utils.dataset', self.verl_dataset_mock),
            ('verl.utils.dataset.dataset_utils', self.verl_dataset_utils_mock),
            ('verl.utils.torch_functional', self.verl_torch_func_mock),
            ('verl.utils.ulysses', self.verl_ulysses_mock),
        ]:
            if name in sys.modules:
                del sys.modules[name]
            sys.modules[name] = module
            self.modules_to_clean.append(name)

    def teardown_method(self):
        """Clean up mocked modules."""
        for name in self.modules_to_clean:
            if name in sys.modules:
                del sys.modules[name]
        for name in list(sys.modules.keys()):
            if 'aura.trainer.train_adapter.verl' in name:
                del sys.modules[name]

    def _setup_params(self, use_remove_padding, use_fused_kernels, calculate_entropy):
        """Setup parameters for tests."""
        def get_non_tensor_data(data, key, default=None):
            params = {
                'use_remove_padding': use_remove_padding,
                'pad_mode': self.MockDatasetPadMode.NO_PADDING,
                'use_fused_kernels': use_fused_kernels,
                'calculate_entropy': calculate_entropy,
                'max_response_length': 10,
            }
            return params.get(key, default)

        self.verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=get_non_tensor_data)

    def test_remove_padding_fused_no_entropy(self):
        """use_remove_padding=True, use_fused_kernels=True, calculate_entropy=False"""
        self._setup_params(True, True, False)

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(1, 50)
        mock_output.entropy = torch.randn(1, 50)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_remove_padding_fused_entropy(self):
        """use_remove_padding=True, use_fused_kernels=True, calculate_entropy=True"""
        self._setup_params(True, True, True)

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(1, 50)
        mock_output.entropy = torch.randn(1, 50)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_remove_padding_no_fused_no_entropy(self):
        """use_remove_padding=True, use_fused_kernels=False, calculate_entropy=False"""
        self._setup_params(True, False, False)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 50, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_remove_padding_no_fused_entropy(self):
        """use_remove_padding=True, use_fused_kernels=False, calculate_entropy=True"""
        self._setup_params(True, False, True)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False
        mock_self.engine_config = MagicMock()
        mock_self.engine_config.entropy_checkpointing = False
        mock_self.compute_entropy_from_logits = MagicMock(return_value=torch.randn(50))

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 50, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_remove_padding_no_fused_entropy_checkpointing(self):
        """use_remove_padding=True, use_fused_kernels=False, calculate_entropy=True, entropy_checkpointing=True"""
        self._setup_params(True, False, True)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False
        mock_self.engine_config = MagicMock()
        mock_self.engine_config.entropy_checkpointing = True
        mock_self.compute_entropy_from_logits = MagicMock(return_value=torch.randn(50))

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 50, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        def mock_checkpoint(fn, *args, **kwargs):
            return fn(*args)

        with patch('torch.utils.checkpoint.checkpoint', side_effect=mock_checkpoint):
            result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_remove_padding_sp_repeat_factor(self):
        """use_remove_padding=True, use_fused_kernels=False, sp repeat factor applied"""
        self._setup_params(True, False, False)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(100))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 100, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (100,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 50))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 50, 100]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_remove_padding_ulysses_sp_fused_no_entropy(self):
        """use_remove_padding=True, use_fused_kernels=True, use_ulysses_sp=True"""
        self._setup_params(True, True, False)
        self.verl_ulysses_mock.gather_outputs_and_unpad = MagicMock(return_value=torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = True

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(1, 100)
        mock_output.entropy = torch.randn(1, 100)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (100,)),
            "temperature_rmpad": torch.ones(100),
            "pad_size": torch.tensor([50, 50]),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_remove_padding_ulysses_sp_fused_entropy(self):
        """use_remove_padding=True, use_fused_kernels=True, use_ulysses_sp=True, calculate_entropy=True"""
        self._setup_params(True, True, True)
        self.verl_ulysses_mock.gather_outputs_and_unpad = MagicMock(side_effect=lambda x, **kwargs: torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = True

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(1, 100)
        mock_output.entropy = torch.randn(1, 100)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (100,)),
            "temperature_rmpad": torch.ones(100),
            "pad_size": torch.tensor([50, 50]),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_remove_padding_ulysses_sp_no_fused_no_entropy(self):
        """use_remove_padding=True, use_fused_kernels=False, use_ulysses_sp=True"""
        self._setup_params(True, False, False)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(50))
        self.verl_ulysses_mock.gather_outputs_and_unpad = MagicMock(return_value=torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 100, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (100,)),
            "temperature_rmpad": torch.ones(100),
            "pad_size": torch.tensor([50, 50]),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_remove_padding_ulysses_sp_no_fused_entropy(self):
        """use_remove_padding=True, use_fused_kernels=False, use_ulysses_sp=True, calculate_entropy=True"""
        self._setup_params(True, False, True)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(50))
        self.verl_ulysses_mock.gather_outputs_and_unpad = MagicMock(side_effect=lambda x, **kwargs: torch.randn(50))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = True
        mock_self.engine_config = MagicMock()
        mock_self.engine_config.entropy_checkpointing = False
        mock_self.compute_entropy_from_logits = MagicMock(return_value=torch.randn(100))

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 100, 50257)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (100,)),
            "temperature_rmpad": torch.ones(100),
            "pad_size": torch.tensor([50, 50]),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_no_remove_padding_fused_no_entropy(self):
        """use_remove_padding=False, use_fused_kernels=True, calculate_entropy=False"""
        self._setup_params(False, True, False)

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(2, 20)
        mock_output.entropy = torch.randn(2, 20)

        output_args = {
            "temperature": torch.ones(2),
        }

        micro_batch = TensorDict({
            "input_ids": torch.randint(0, 50257, (2, 20)),
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_no_remove_padding_fused_entropy(self):
        """use_remove_padding=False, use_fused_kernels=True, calculate_entropy=True"""
        self._setup_params(False, True, True)

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(2, 20)
        mock_output.entropy = torch.randn(2, 20)

        output_args = {
            "temperature": torch.ones(2),
        }

        micro_batch = TensorDict({
            "input_ids": torch.randint(0, 50257, (2, 20)),
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_no_remove_padding_no_fused_no_entropy(self):
        """use_remove_padding=False, use_fused_kernels=False, calculate_entropy=False"""
        self._setup_params(False, False, False)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(20))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 20, 50257)

        output_args = {
            "temperature": torch.ones(2),
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (20,)),
        }

        input_ids = torch.randint(0, 50257, (2, 10))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 10, 20]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result

    def test_no_remove_padding_no_fused_entropy(self):
        """use_remove_padding=False, use_fused_kernels=False, calculate_entropy=True"""
        self._setup_params(False, False, True)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(20))
        self.verl_torch_func_mock.entropy_from_logits = MagicMock(return_value=torch.randn(2, 20))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False
        mock_self.engine_config = MagicMock()
        mock_self.engine_config.entropy_checkpointing = False

        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 20, 50257)

        output_args = {
            "temperature": torch.ones(2),
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (20,)),
        }

        input_ids = torch.randint(0, 50257, (2, 10))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 10, 20]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_no_remove_padding_no_fused_entropy_checkpointing(self):
        """use_remove_padding=False, use_fused_kernels=False, calculate_entropy=True, checkpointing=True"""
        self._setup_params(False, False, True)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(20))
        self.verl_torch_func_mock.entropy_from_logits = MagicMock(return_value=torch.randn(2, 20))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False
        mock_self.engine_config = MagicMock()
        mock_self.engine_config.entropy_checkpointing = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 20, 50257)

        output_args = {
            "temperature": torch.ones(2),
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (20,)),
        }

        input_ids = torch.randint(0, 50257, (2, 10))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 10, 20]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        def mock_checkpoint(fn, *args, **kwargs):
            return fn(*args)

        with patch('torch.utils.checkpoint.checkpoint', side_effect=mock_checkpoint):
            result = prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

        assert "log_probs" in result
        assert "entropy" in result

    def test_remove_padding_pad_mode_fixed_raises(self):
        """Test NotImplementedError when pad_mode=FIXED with use_remove_padding=True"""
        def get_non_tensor_data(data, key, default=None):
            params = {
                'use_remove_padding': True,
                'pad_mode': self.MockDatasetPadMode.FIXED,
                'use_fused_kernels': True,
                'calculate_entropy': False,
                'max_response_length': 10,
            }
            return params.get(key, default)

        self.verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=get_non_tensor_data)

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.log_probs = torch.randn(1, 50)
        mock_output.entropy = torch.randn(1, 50)

        output_args = {
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (50,)),
            "temperature_rmpad": torch.ones(50),
        }

        input_ids = torch.randint(0, 50257, (2, 25))
        input_ids.offsets = MagicMock(return_value=torch.tensor([0, 25, 50]))

        micro_batch = TensorDict({
            "input_ids": input_ids,
        }, batch_size=2)

        with pytest.raises(NotImplementedError, match="pad_mode 1 not implemented"):
            prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)

    def test_no_remove_padding_pad_mode_fixed_raises(self):
        """Test NotImplementedError when pad_mode=FIXED with use_remove_padding=False and use_fused_kernels=False"""
        def get_non_tensor_data(data, key, default=None):
            params = {
                'use_remove_padding': False,
                'pad_mode': self.MockDatasetPadMode.FIXED,
                'use_fused_kernels': False,
                'calculate_entropy': False,
                'max_response_length': 10,
            }
            return params.get(key, default)

        self.verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=get_non_tensor_data)
        self.verl_torch_func_mock.logprobs_from_logits = MagicMock(return_value=torch.randn(20))

        from aura.trainer.train_adapter.verl.patch.fsdp_transformer_impl_patch import prepare_model_outputs_patch

        mock_self = MagicMock()
        mock_self.use_ulysses_sp = False

        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 20, 50257)

        output_args = {
            "temperature": torch.ones(2),
            "input_ids_rmpad_rolled": torch.randint(0, 50257, (20,)),
        }

        micro_batch = TensorDict({
            "input_ids": torch.randint(0, 50257, (2, 20)),
        }, batch_size=2)

        with pytest.raises(NotImplementedError, match="pad_mode 1 not implemented"):
            prepare_model_outputs_patch(mock_self, mock_output, output_args, micro_batch)
