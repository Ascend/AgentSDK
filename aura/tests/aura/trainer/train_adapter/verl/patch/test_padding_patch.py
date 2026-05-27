import sys
import pytest
import torch
from unittest.mock import MagicMock, patch
from types import ModuleType
from tensordict import TensorDict


class TestPaddingPatch:

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Setup mock modules for verl dependencies."""
        modules_to_remove = []
        original_modules = {}

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('verl') or mod_name.startswith('aura.trainer.train_adapter.verl.patch'):
                original_modules[mod_name] = sys.modules[mod_name]
                modules_to_remove.append(mod_name)

        for mod_name in modules_to_remove:
            del sys.modules[mod_name]

        verl_mock = ModuleType('verl')
        verl_utils_mock = ModuleType('verl.utils')
        verl_utils_mock.tensordict_utils = MagicMock()
        verl_utils_mock.tensordict_utils.get_non_tensor_data = MagicMock(return_value=-1)
        verl_mock.utils = verl_utils_mock

        sys.modules['verl'] = verl_mock
        sys.modules['verl.utils'] = verl_utils_mock

        yield verl_utils_mock

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('verl') or mod_name.startswith('aura.trainer.train_adapter.verl.patch'):
                del sys.modules[mod_name]

        for name, module in original_modules.items():
            sys.modules[name] = module

    def test_no_padding_2_padding_patch_basic(self, setup_mock):
        """Test no_padding_2_padding_patch with basic case."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        result = no_padding_2_padding_patch(tensor, data)
        assert result.shape == (batch_size, response_len)

    def test_no_padding_2_padding_patch_sp_size(self, setup_mock):
        """Test no_padding_2_padding_patch with sequence parallel size > 1."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens * 2)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '2'}):
            result = no_padding_2_padding_patch(tensor, data)
            assert result.shape == (batch_size, response_len)

    def test_no_padding_2_padding_patch_sp_size_not_matching(self, setup_mock):
        """Test no_padding_2_padding_patch with sp_size not matching ratio."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens * 3)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        with patch.dict('os.environ', {'ULYSSES_SEQUENCE_PARALLEL_SIZE': '2'}):
            with pytest.raises(ValueError, match="sequence_offsets\\[-1\\]"):
                no_padding_2_padding_patch(tensor, data)

    def test_no_padding_2_padding_patch_tensor_is_nested(self, setup_mock):
        """Test no_padding_2_padding_patch with tensor.is_nested attribute check."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        setup_mock.tensordict_utils.get_non_tensor_data.return_value = -1

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        result = no_padding_2_padding_patch(tensor, data)
        assert result.shape == (batch_size, response_len)

    def test_no_padding_2_padding_patch_offset_mismatch(self, setup_mock):
        """Test no_padding_2_padding_patch raises error when offsets don't match values."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens + 5)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        with pytest.raises(ValueError, match="sequence_offsets\\[-1\\]"):
            no_padding_2_padding_patch(tensor, data)

    def test_no_padding_2_padding_patch_response_len_variation(self, setup_mock):
        """Test no_padding_2_padding_patch with different response lengths."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        setup_mock.tensordict_utils.get_non_tensor_data.return_value = -1

        batch_size = 2
        prompt_len = 4
        response_len = 3

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        result = no_padding_2_padding_patch(tensor, data)
        assert result.shape == (batch_size, response_len)

    def test_no_padding_2_padding_patch_log_info(self, setup_mock):
        """Test no_padding_2_padding_patch logger info call."""
        from aura.trainer.train_adapter.verl.patch.padding_patch import no_padding_2_padding_patch

        setup_mock.tensordict_utils.get_non_tensor_data.return_value = -1

        batch_size = 2
        prompt_len = 3
        response_len = 2

        prompts = torch.randint(0, 1000, (batch_size, prompt_len))
        responses = torch.randint(0, 1000, (batch_size, response_len))
        attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64)

        total_tokens = batch_size * (prompt_len + response_len)
        values = torch.randn(total_tokens)
        tensor = values

        data = TensorDict({
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        }, batch_size=batch_size)

        with patch("aura.trainer.train_adapter.verl.patch.padding_patch.logger") as mock_logger:
            result = no_padding_2_padding_patch(tensor, data)
            assert result.shape == (batch_size, response_len)
            mock_logger.info.assert_called()
