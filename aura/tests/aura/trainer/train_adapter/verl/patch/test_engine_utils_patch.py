import pytest
import torch
from unittest.mock import MagicMock, patch
from tensordict import TensorDict
import sys
from types import ModuleType


class TestEngineUtilsPatch:

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Setup mock modules for verl dependencies."""
        verl_mock = ModuleType('verl')
        verl_utils_mock = ModuleType('verl.utils')
        verl_dataset_mock = ModuleType('verl.utils.dataset')
        verl_dataset_utils_mock = ModuleType('verl.utils.dataset.dataset_utils')
        verl_py_func_mock = ModuleType('verl.utils.py_functional')
        verl_seqlen_mock = ModuleType('verl.utils.seqlen_balancing')

        class MockDatasetPadMode:
            NO_PADDING = 0
            FIXED = 1

        verl_dataset_utils_mock.DatasetPadMode = MockDatasetPadMode
        verl_utils_mock.tensordict_utils = MagicMock()
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(return_value=False)
        verl_py_func_mock.append_to_dict = MagicMock(side_effect=lambda agg, new: agg.update(new))
        verl_seqlen_mock.restore_dynamic_batch = MagicMock(side_effect=lambda x, y: x)

        verl_dataset_mock.dataset_utils = verl_dataset_utils_mock
        verl_utils_mock.dataset = verl_dataset_mock
        verl_utils_mock.py_functional = verl_py_func_mock
        verl_utils_mock.seqlen_balancing = verl_seqlen_mock
        verl_mock.utils = verl_utils_mock

        original_modules = {}
        for name, module in [
            ('verl', verl_mock),
            ('verl.utils', verl_utils_mock),
            ('verl.utils.dataset', verl_dataset_mock),
            ('verl.utils.dataset.dataset_utils', verl_dataset_utils_mock),
            ('verl.utils.py_functional', verl_py_func_mock),
            ('verl.utils.seqlen_balancing', verl_seqlen_mock),
        ]:
            if name in sys.modules:
                original_modules[name] = sys.modules[name]
            sys.modules[name] = module

        yield verl_utils_mock, MockDatasetPadMode

        for name, module in original_modules.items():
            sys.modules[name] = module

    def test_postprocess_batch_func_patch_empty_output(self, setup_mock):
        """Test postprocess_batch_func_patch with empty output list."""
        verl_utils_mock, MockDatasetPadMode = setup_mock
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=lambda data, key, default=None: {
            'use_dynamic_bsz': False,
            'pad_mode': MockDatasetPadMode.NO_PADDING,
        }.get(key, default))

        from aura.trainer.train_adapter.verl.patch.engine_utils_patch import postprocess_batch_func_patch

        output_lst = []
        indices = torch.tensor([0])

        data = TensorDict({
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }, batch_size=1)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '1'}):
            result = postprocess_batch_func_patch(output_lst, indices, data)

            assert "model_output" in result
            assert "loss" in result
            assert "metrics" in result
            assert len(result["model_output"]) == 0
            assert len(result["loss"]) == 0
            assert len(result["metrics"]) == 0

    def test_postprocess_batch_func_patch_with_loss_and_metrics(self, setup_mock):
        """Test postprocess_batch_func_patch with loss and metrics."""
        verl_utils_mock, MockDatasetPadMode = setup_mock
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=lambda data, key, default=None: {
            'use_dynamic_bsz': False,
            'pad_mode': MockDatasetPadMode.NO_PADDING,
        }.get(key, default))

        from aura.trainer.train_adapter.verl.patch.engine_utils_patch import postprocess_batch_func_patch

        output_lst = [
            {"loss": torch.tensor(0.5), "metrics": {"acc": 0.8}},
            {"loss": torch.tensor(0.6), "metrics": {"acc": 0.9}},
        ]
        indices = torch.tensor([0, 1])

        data = TensorDict({
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }, batch_size=2)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '1'}):
            result = postprocess_batch_func_patch(output_lst, indices, data)

            assert "model_output" in result
            assert len(result["loss"]) == 2
            assert "acc" in result["metrics"]



    def test_postprocess_batch_func_patch_with_model_output_offsets(self, setup_mock):
        """Test postprocess_batch_func_patch with model_output using offsets."""
        verl_utils_mock, MockDatasetPadMode = setup_mock
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=lambda data, key, default=None: {
            'use_dynamic_bsz': False,
            'pad_mode': MockDatasetPadMode.NO_PADDING,
        }.get(key, default))

        from aura.trainer.train_adapter.verl.patch.engine_utils_patch import postprocess_batch_func_patch

        mock_nested_tensor = MagicMock()
        mock_nested_tensor.values.return_value = torch.randn(6, 64)
        mock_nested_tensor._lengths = None
        mock_nested_tensor.offsets.return_value = torch.tensor([0, 3, 6])

        output_lst = [
            {"model_output": {"logits": mock_nested_tensor}},
        ]
        indices = torch.tensor([0, 1])

        data = TensorDict({
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }, batch_size=2)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '1'}):
            result = postprocess_batch_func_patch(output_lst, indices, data)

            assert "model_output" in result
            assert "logits" in result["model_output"]

    def test_postprocess_batch_func_patch_use_dynamic_bsz(self, setup_mock):
        """Test postprocess_batch_func_patch with use_dynamic_bsz=True."""
        verl_utils_mock, MockDatasetPadMode = setup_mock
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=lambda data, key, default=None: {
            'use_dynamic_bsz': True,
            'pad_mode': MockDatasetPadMode.NO_PADDING,
        }.get(key, default))

        from aura.trainer.train_adapter.verl.patch.engine_utils_patch import postprocess_batch_func_patch

        mock_nested_tensor = MagicMock()
        mock_nested_tensor.values.return_value = torch.randn(6, 64)
        mock_nested_tensor._lengths = torch.tensor([3, 3])

        output_lst = [
            {"model_output": {"logits": mock_nested_tensor}},
        ]
        indices = torch.tensor([1, 0])

        data = TensorDict({
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }, batch_size=2)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '1'}):
            result = postprocess_batch_func_patch(output_lst, indices, data)

            assert "model_output" in result
            assert "logits" in result["model_output"]

    def test_postprocess_batch_func_patch_sp_size(self, setup_mock):
        """Test postprocess_batch_func_patch with sequence parallel size > 1."""
        verl_utils_mock, MockDatasetPadMode = setup_mock
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(side_effect=lambda data, key, default=None: {
            'use_dynamic_bsz': False,
            'pad_mode': MockDatasetPadMode.NO_PADDING,
        }.get(key, default))

        from aura.trainer.train_adapter.verl.patch.engine_utils_patch import postprocess_batch_func_patch

        mock_nested_tensor = MagicMock()
        mock_nested_tensor.values.return_value = torch.randn(7, 64)
        mock_nested_tensor._lengths = torch.tensor([3, 5])

        output_lst = [
            {"model_output": {"logits": mock_nested_tensor}},
        ]
        indices = torch.tensor([0, 1])

        data = TensorDict({
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }, batch_size=2)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '2'}):
            result = postprocess_batch_func_patch(output_lst, indices, data)

            assert "model_output" in result
