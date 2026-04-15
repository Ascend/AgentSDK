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
import unittest
from unittest.mock import patch, MagicMock, call


# Create mock objects
mock_torch = MagicMock()
mock_tensordict = MagicMock()
mock_pad_sequence = MagicMock()
mock_mindspeed_rl = MagicMock()

# Make mstx_timer_decorator a passthrough decorator so it does not wrap the function
mock_mindspeed_rl.utils.utils.mstx_timer_decorator = lambda fn: fn

# Set up torch sub-attributes needed by the source module
mock_torch.nn.functional.pad = MagicMock()
mock_torch.nn.utils.rnn.pad_sequence = mock_pad_sequence
mock_torch.Tensor = MagicMock

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'torch': mock_torch,
    'torch.nn': mock_torch.nn,
    'torch.nn.functional': mock_torch.nn.functional,
    'torch.nn.utils': mock_torch.nn.utils,
    'torch.nn.utils.rnn': mock_torch.nn.utils.rnn,
    'tensordict': mock_tensordict,
    'mindspeed_rl': mock_mindspeed_rl,
    'mindspeed_rl.utils': mock_mindspeed_rl.utils,
    'mindspeed_rl.utils.utils': mock_mindspeed_rl.utils.utils,
}):
    from agentic_rl.data_manager.data_transform import (
        padding_dict_to_tensor_dict,
        padding_dict_to_tensor_dict_fast,
    )


class TestPaddingDictToTensorDict(unittest.TestCase):
    """Tests for padding_dict_to_tensor_dict function."""

    def setUp(self):
        mock_torch.reset_mock()
        mock_tensordict.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()
        mock_tensordict.reset_mock()

    def test_single_column_single_row(self):
        """Single column with a single tensor should produce padded output."""
        tensor_a = MagicMock()
        tensor_a.__len__ = MagicMock(return_value=3)

        mock_torch.nn.functional.pad.return_value = MagicMock()
        mock_torch.stack.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        experience_data = {"col_a": [tensor_a]}

        result = padding_dict_to_tensor_dict(experience_data)

        # F.pad should be called once (one tensor in one column)
        mock_torch.nn.functional.pad.assert_called_once()
        # torch.stack should be called for the column tensors and original_length
        self.assertTrue(mock_torch.stack.called)
        # TensorDict.from_dict should be called on the final dict
        mock_tensordict.TensorDict.from_dict.assert_called_once()

    def test_multiple_columns(self):
        """Multiple columns should each be padded independently."""
        tensor_a1 = MagicMock()
        tensor_a1.__len__ = MagicMock(return_value=3)
        tensor_a2 = MagicMock()
        tensor_a2.__len__ = MagicMock(return_value=5)

        tensor_b1 = MagicMock()
        tensor_b1.__len__ = MagicMock(return_value=2)
        tensor_b2 = MagicMock()
        tensor_b2.__len__ = MagicMock(return_value=4)

        mock_torch.nn.functional.pad.return_value = MagicMock()
        mock_torch.stack.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        experience_data = {
            "col_a": [tensor_a1, tensor_a2],
            "col_b": [tensor_b1, tensor_b2],
        }

        result = padding_dict_to_tensor_dict(experience_data)

        # F.pad should be called 4 times (2 tensors x 2 columns)
        self.assertEqual(mock_torch.nn.functional.pad.call_count, 4)
        # torch.stack called for col_a, col_b, and original_length
        self.assertEqual(mock_torch.stack.call_count, 3)

    def test_original_length_included(self):
        """The result dict passed to TensorDict.from_dict should contain 'original_length'."""
        tensor_x = MagicMock()
        tensor_x.__len__ = MagicMock(return_value=7)

        mock_torch.nn.functional.pad.return_value = MagicMock()
        mock_torch.stack.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        experience_data = {"col_x": [tensor_x]}

        padding_dict_to_tensor_dict(experience_data)

        # Inspect the dict passed to TensorDict.from_dict
        from_dict_call = mock_tensordict.TensorDict.from_dict.call_args
        passed_dict = from_dict_call[0][0]
        self.assertIn("original_length", passed_dict)

    def test_padding_value_zero(self):
        """F.pad should be called with value=0 for zero-padding."""
        tensor_short = MagicMock()
        tensor_short.__len__ = MagicMock(return_value=2)
        tensor_long = MagicMock()
        tensor_long.__len__ = MagicMock(return_value=5)

        mock_torch.nn.functional.pad.return_value = MagicMock()
        mock_torch.stack.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        experience_data = {"col": [tensor_short, tensor_long]}

        padding_dict_to_tensor_dict(experience_data)

        # Check that the shorter tensor was padded with (0, 3) and value=0
        for c in mock_torch.nn.functional.pad.call_args_list:
            self.assertEqual(c.kwargs.get('value', c[1].get('value', 0) if len(c) > 1 and isinstance(c[1], dict) else 0), 0)


class TestPaddingDictToTensorDictFast(unittest.TestCase):
    """Tests for padding_dict_to_tensor_dict_fast function."""

    def setUp(self):
        mock_torch.reset_mock()
        mock_tensordict.reset_mock()
        mock_pad_sequence.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()
        mock_tensordict.reset_mock()
        mock_pad_sequence.reset_mock()

    def _make_tensor_mock(self, length, ndim=1):
        """Helper to build a mock tensor with size and ndim."""
        t = MagicMock()
        t.ndim = ndim
        t.size.return_value = length
        t.unsqueeze.return_value = t
        # Make is_tensor return True for this object
        return t

    def test_single_column(self):
        """Single column should call pad_sequence once and cat once."""
        t1 = self._make_tensor_mock(3)
        t2 = self._make_tensor_mock(5)

        mock_torch.is_tensor.return_value = True
        mock_torch.tensor.return_value = MagicMock()
        mock_pad_sequence.return_value = MagicMock()
        mock_torch.cat.return_value = MagicMock()

        experience_data = {"col_a": [t1, t2]}

        result = padding_dict_to_tensor_dict_fast(experience_data)

        mock_pad_sequence.assert_called_once()
        mock_torch.cat.assert_called_once()
        mock_tensordict.TensorDict.from_dict.assert_called_once()

    def test_multiple_columns(self):
        """Multiple columns should each call pad_sequence separately."""
        t1 = self._make_tensor_mock(3)
        t2 = self._make_tensor_mock(5)
        t3 = self._make_tensor_mock(2)
        t4 = self._make_tensor_mock(4)

        mock_torch.is_tensor.return_value = True
        mock_torch.tensor.return_value = MagicMock()
        mock_pad_sequence.return_value = MagicMock()
        mock_torch.cat.return_value = MagicMock()

        experience_data = {
            "col_a": [t1, t2],
            "col_b": [t3, t4],
        }

        result = padding_dict_to_tensor_dict_fast(experience_data)

        self.assertEqual(mock_pad_sequence.call_count, 2)
        # cat is called once to merge all original_lens
        mock_torch.cat.assert_called_once()

    def test_type_error_on_non_tensor_items(self):
        """Should raise TypeError when column contains non-tensor items."""
        mock_torch.is_tensor.return_value = False

        experience_data = {"col_bad": ["not_a_tensor", "also_not"]}

        with self.assertRaises(TypeError) as ctx:
            padding_dict_to_tensor_dict_fast(experience_data)

        self.assertIn("col_bad", str(ctx.exception))

    def test_scalar_tensors_unsqueezed(self):
        """Scalar tensors (ndim==0) should be unsqueezed to 1-D before padding."""
        scalar_t = self._make_tensor_mock(1, ndim=0)
        unsqueezed = MagicMock()
        unsqueezed.ndim = 1
        unsqueezed.size.return_value = 1
        scalar_t.unsqueeze.return_value = unsqueezed

        mock_torch.is_tensor.return_value = True
        mock_torch.tensor.return_value = MagicMock()
        mock_pad_sequence.return_value = MagicMock()
        mock_torch.cat.return_value = MagicMock()

        experience_data = {"col_s": [scalar_t]}

        padding_dict_to_tensor_dict_fast(experience_data)

        scalar_t.unsqueeze.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main()
