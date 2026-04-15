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
from unittest.mock import patch, MagicMock


# Create mock objects
mock_torch = MagicMock()

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'torch': mock_torch,
}):
    from agentic_rl.controllers.utils.msg_handler import (
        is_seq_like,
        _find_key,
        _len0,
        deserialize_and_split,
    )


class TestIsSeqLike(unittest.TestCase):
    """Tests for is_seq_like function."""

    def setUp(self):
        mock_torch.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()

    def test_list_matching_length(self):
        """A list with length equal to B should return True."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like([1, 2, 3], 3)
        self.assertTrue(result)

    def test_list_non_matching_length(self):
        """A list with length not equal to B should return False."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like([1, 2, 3], 5)
        self.assertFalse(result)

    def test_tuple_matching_length(self):
        """A tuple with length equal to B should return True."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like((10, 20), 2)
        self.assertTrue(result)

    def test_tuple_non_matching_length(self):
        """A tuple with length not equal to B should return False."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like((10, 20), 4)
        self.assertFalse(result)

    def test_tensor_with_matching_size(self):
        """A tensor with dim > 0 and size(0) == B should return True."""
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 2
        mock_tensor.size.return_value = 4
        mock_torch.is_tensor.return_value = True
        result = is_seq_like(mock_tensor, 4)
        self.assertTrue(result)
        mock_tensor.dim.assert_called_once()
        mock_tensor.size.assert_called_once_with(0)

    def test_tensor_with_non_matching_size(self):
        """A tensor with dim > 0 but size(0) != B should return False."""
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 2
        mock_tensor.size.return_value = 3
        mock_torch.is_tensor.return_value = True
        result = is_seq_like(mock_tensor, 5)
        self.assertFalse(result)

    def test_tensor_zero_dim(self):
        """A scalar tensor (dim == 0) should return False."""
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 0
        mock_torch.is_tensor.return_value = True
        result = is_seq_like(mock_tensor, 1)
        self.assertFalse(result)

    def test_non_seq_non_tensor(self):
        """A non-list, non-tuple, non-tensor value should return False."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like(42, 1)
        self.assertFalse(result)

    def test_empty_list_with_zero_b(self):
        """An empty list should return True when B is 0."""
        mock_torch.is_tensor.return_value = False
        result = is_seq_like([], 0)
        self.assertTrue(result)


class TestFindKey(unittest.TestCase):
    """Tests for _find_key function."""

    def test_find_first_matching_key(self):
        """Should return the first candidate key found in the dict."""
        d = {"alpha": 1, "beta": 2, "gamma": 3}
        result = _find_key(d, ["beta", "gamma"])
        self.assertEqual(result, "beta")

    def test_no_matching_key(self):
        """Should return None when no candidate key exists in the dict."""
        d = {"alpha": 1, "beta": 2}
        result = _find_key(d, ["delta", "epsilon"])
        self.assertIsNone(result)

    def test_empty_candidates(self):
        """Should return None when the candidates list is empty."""
        d = {"alpha": 1}
        result = _find_key(d, [])
        self.assertIsNone(result)


class TestLen0(unittest.TestCase):
    """Tests for _len0 function."""

    def setUp(self):
        mock_torch.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()

    def test_tensor_input(self):
        """For a tensor, _len0 should call .size(0) and return its value."""
        mock_tensor = MagicMock()
        mock_tensor.size.return_value = 7
        mock_torch.is_tensor.return_value = True
        result = _len0(mock_tensor)
        self.assertEqual(result, 7)
        mock_tensor.size.assert_called_once_with(0)

    def test_list_input(self):
        """For a list, _len0 should return len()."""
        mock_torch.is_tensor.return_value = False
        result = _len0([10, 20, 30, 40])
        self.assertEqual(result, 4)

    def test_tuple_input(self):
        """For a tuple, _len0 should return len()."""
        mock_torch.is_tensor.return_value = False
        result = _len0((1, 2))
        self.assertEqual(result, 2)


class TestDeserializeAndSplit(unittest.TestCase):
    """Tests for deserialize_and_split function."""

    def setUp(self):
        mock_torch.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()

    def test_deserialize_calls_torch_load(self):
        """Verify torch.load is called with the correct arguments."""
        expected_output = {"key": "value"}
        mock_torch.load.return_value = expected_output

        result = deserialize_and_split("/tmp/test_file.pt")

        mock_torch.load.assert_called_once_with("/tmp/test_file.pt", map_location="cpu")
        self.assertEqual(result, expected_output)

    def test_deserialize_returns_loaded_data(self):
        """Verify the function returns exactly what torch.load returns."""
        mock_data = [MagicMock(), MagicMock()]
        mock_torch.load.return_value = mock_data

        result = deserialize_and_split("some_path.bin")
        self.assertIs(result, mock_data)


if __name__ == '__main__':
    unittest.main()
