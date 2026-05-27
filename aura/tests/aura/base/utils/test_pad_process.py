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

import sys
import types
from unittest.mock import MagicMock, patch, ANY
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for pad_process
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build isolated fake modules and return the module under test."""

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")

    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.utils = types.ModuleType("torch.nn.utils")
    fake_torch.nn.utils.rnn = types.ModuleType("torch.nn.utils.rnn")
    fake_torch.nn.utils.rnn.pad_sequence = MagicMock()
    fake_torch.nn.functional = MagicMock()
    fake_torch.nn.functional.pad = MagicMock()

    fake_torch.tensor = MagicMock()
    fake_torch.stack = MagicMock()
    fake_torch.full = MagicMock()
    fake_torch.clamp = MagicMock()
    fake_torch.nonzero = MagicMock()
    fake_torch.Tensor = MagicMock
    fake_torch.int32 = "int32"

    # ---- fake tensordict ----
    fake_tensordict = types.ModuleType("tensordict")
    fake_tensordict.TensorDict = MagicMock()
    fake_tensordict.TensorDict.from_dict = MagicMock()

    # ---- fake copy ----
    fake_copy = types.ModuleType("copy")
    fake_copy.deepcopy = MagicMock(side_effect=lambda x: x)

    # ---- aura packages to locate real file ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_utils = types.ModuleType("aura.base.utils")
    fake_aura_base_utils.__path__ = [os.path.join(base, "base/utils")]

    fakes = {
        "torch": fake_torch,
        "torch.nn": fake_torch.nn,
        "torch.nn.utils": fake_torch.nn.utils,
        "torch.nn.utils.rnn": fake_torch.nn.utils.rnn,
        "torch.nn.functional": fake_torch.nn.functional,
        "tensordict": fake_tensordict,
        "copy": fake_copy,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.utils": fake_aura_base_utils,
    }

    target = "aura.base.utils.pad_process"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.base.utils.pad_process as mod
        yield {
            "mod": mod,
            "fake_torch": fake_torch,
            "fake_tensordict": fake_tensordict,
            "fake_copy": fake_copy,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_tensor(shape, device="cpu"):
    """Create a mock tensor with given shape (accepts int or list)."""
    if isinstance(shape, int):
        shape = [shape]
    t = MagicMock()
    t.shape = list(shape)
    t.dtype = "float32"
    t.device = device
    t.item.return_value = 0
    t.__len__.return_value = shape[0] if shape else 0
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPaddingDictToTensorDict:
    def test_basic_padding(self, fake_env):
        mod = fake_env["mod"]
        data = {
            "a": [make_tensor(3), make_tensor(5)],
            "b": [make_tensor(2), make_tensor(4)],
        }
        result = mod.padding_dict_to_tensor_dict(data)
        fake_env["fake_torch"].stack.assert_called()
        fake_env["fake_torch"].nn.functional.pad.assert_called()
        fake_env["fake_tensordict"].TensorDict.from_dict.assert_called_once()

    def test_empty_input(self, fake_env):
        mod = fake_env["mod"]
        data = {}
        mod.padding_dict_to_tensor_dict(data)


class TestRemovePaddingTensorDictToDict:
    def test_remove_padding(self, fake_env):
        mod = fake_env["mod"]
        data_dict = MagicMock()
        data_dict.keys.return_value = ["a", "b", "original_length"]
        len_tensor = MagicMock()
        len_tensor.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=3))
        data_dict.__getitem__.side_effect = lambda key: {
            "a": MagicMock(), "b": MagicMock(), "original_length": len_tensor
        }[key]
        data_dict.items.return_value = [("a", MagicMock()), ("b", MagicMock()), ("original_length", len_tensor)]

        result = mod.remove_padding_tensor_dict_to_dict(data_dict)
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result
        assert "original_length" not in result

    def test_none_input(self, fake_env):
        mod = fake_env["mod"]
        assert mod.remove_padding_tensor_dict_to_dict(None) == {}

    def test_missing_original_length(self, fake_env):
        mod = fake_env["mod"]
        data_dict = MagicMock()
        data_dict.keys.return_value = ["a"]
        data_dict.__getitem__.side_effect = lambda key: MagicMock()
        result = mod.remove_padding_tensor_dict_to_dict(data_dict)
        assert result is data_dict


class TestPadMultiple:
    def test_pad_multiple(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        data_list = [make_tensor(3), make_tensor(5)]
        padded_mock = MagicMock()
        padded_mock.size.return_value = 6
        fake_torch.nn.utils.rnn.pad_sequence.return_value = padded_mock
        result = mod.pad_multiple(data_list, 0, multiple=4)
        fake_torch.nn.utils.rnn.pad_sequence.assert_called_once()
        fake_torch.nn.functional.pad.assert_called_once()

    def test_pad_multiple_already_multiple(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        padded_mock = MagicMock()
        padded_mock.size.return_value = 8
        fake_torch.nn.utils.rnn.pad_sequence.return_value = padded_mock
        result = mod.pad_multiple([make_tensor(3), make_tensor(5)], 0, multiple=4)
        fake_torch.nn.functional.pad.assert_called()


class TestTruncateMiddleAndPad:
    def test_truncate_middle(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        responses = MagicMock()
        responses.shape = [2, 10]
        input_tensor = MagicMock()
        input_tensor.shape = [2, 15, 5]
        truncate_lengths = MagicMock()
        truncate_lengths.shape = [2, 2]
        truncate_lengths.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=idx))
        fake_torch.clamp.return_value = truncate_lengths
        fake_torch.full.return_value = MagicMock()
        result = mod.truncate_middle_and_pad(responses, input_tensor, truncate_lengths)
        assert result is not None


class TestTruncatePromptAndPad:
    def test_truncate_prompt(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        responses = MagicMock()
        responses.shape = [2, 10]
        input_tensor = MagicMock()
        input_tensor.shape = [2, 15]
        truncate_lengths = MagicMock()
        truncate_lengths.shape = [2, 2]
        truncate_lengths.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=idx))
        fake_torch.clamp.return_value = truncate_lengths
        fake_torch.full.return_value = MagicMock()
        result = mod.truncate_prompt_and_pad(responses, input_tensor, truncate_lengths)
        assert result is not None


class TestTruncateRows:
    def test_truncate_rows_left_pad_false(self, fake_env):
        mod = fake_env["mod"]
        tensor = MagicMock()
        tensor.shape = [2, 10]
        index_tensor = MagicMock()
        index_tensor.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=3))
        tensor.__getitem__.side_effect = lambda idx: MagicMock(cpu=lambda: MagicMock())
        result = mod.truncate_rows(tensor, index_tensor, left_pad=False)
        assert len(result) == 2

    def test_truncate_rows_left_pad_true(self, fake_env):
        mod = fake_env["mod"]
        tensor = MagicMock()
        tensor.shape = [2, 10]
        index_tensor = MagicMock()
        index_tensor.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=4))
        tensor.__getitem__.side_effect = lambda idx: MagicMock(cpu=lambda: MagicMock())
        result = mod.truncate_rows(tensor, index_tensor, left_pad=True)
        assert len(result) == 2

    def test_truncate_rows_empty_row(self, fake_env):
        mod = fake_env["mod"]
        tensor = MagicMock()
        tensor.shape = [1, 5]
        index_tensor = MagicMock()
        index_tensor.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=0))
        tensor.__getitem__.side_effect = lambda idx: MagicMock(item=MagicMock(return_value=-1), cpu=lambda: MagicMock())
        result = mod.truncate_rows(tensor, index_tensor)
        assert len(result) == 1


class TestPutPromptsExperience:
    def test_put_prompts_basic(self, fake_env):
        mod = fake_env["mod"]
        batch = {
            "prompts": [[1,2,3], [4,5]],
        }
        with patch.object(mod, "padding_dict_to_tensor_dict", return_value=MagicMock()):
            result, indexes = mod.put_prompts_experience(
                batch, n_samples_per_prompt=2, dataset_additional_keys=[], indexes=None
            )
        assert result is not None
        assert len(indexes) == 4

    def test_put_prompts_with_add_keys(self, fake_env):
        mod = fake_env["mod"]
        batch = {
            "prompts": [[1,2]],
            "extra": ["a"],
        }
        with patch.object(mod, "padding_dict_to_tensor_dict", return_value=MagicMock()):
            result, indexes = mod.put_prompts_experience(
                batch, n_samples_per_prompt=1, dataset_additional_keys=["extra"], indexes=None
            )
        assert result is not None

    def test_put_prompts_add_another_batch(self, fake_env):
        mod = fake_env["mod"]
        batch = {
            "prompts": [[1], [2]],
        }
        with patch.object(mod, "padding_dict_to_tensor_dict", return_value=MagicMock()):
            result, indexes = mod.put_prompts_experience(
                batch, n_samples_per_prompt=1, dataset_additional_keys=[], add_another_batch=True
            )
        assert indexes == [2, 3]
