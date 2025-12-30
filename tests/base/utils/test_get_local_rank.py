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
from unittest.mock import patch, MagicMock

from agentic_rl.base.utils.get_local_rank import get_local_rank


class TestGetLocalRank(unittest.TestCase):
    """Test suite for get_local_rank function."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original environment state
        self.original_env = {}

    def tearDown(self):
        """Clean up after tests."""
        pass

    # =========================================================================
    # Parameter Validation Tests
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_valid_logger_name_none(self, mock_ray_context, mock_loggers):
        """Test get_local_rank with logger_name=None (default behavior)."""
        # Mock Ray to raise RuntimeError (no Ray context)
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        # Create mock logger instance
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank(logger_name=None)
        
        self.assertEqual(result, 0)
        # Verify logger was created with __name__ since logger_name=None
        mock_loggers.assert_called_once()

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_valid_logger_name_string(self, mock_ray_context, mock_loggers):
        """Test get_local_rank with valid string logger_name."""
        # Mock Ray to raise RuntimeError (no Ray context)
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        # Create mock logger instance
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank(logger_name="custom_logger")
        
        self.assertEqual(result, 0)
        # Verify logger was created with custom name
        mock_loggers.assert_called_once_with("custom_logger")

    def test_invalid_logger_name_int(self):
        """Test get_local_rank raises ValueError for integer logger_name."""
        with self.assertRaises(ValueError) as context:
            get_local_rank(logger_name=123)
        
        self.assertIn("logger_name must be a str", str(context.exception))
        self.assertIn("int", str(context.exception))

    def test_invalid_logger_name_list(self):
        """Test get_local_rank raises ValueError for list logger_name."""
        with self.assertRaises(ValueError) as context:
            get_local_rank(logger_name=['logger'])
        
        self.assertIn("logger_name must be a str", str(context.exception))

    def test_invalid_logger_name_dict(self):
        """Test get_local_rank raises ValueError for dict logger_name."""
        with self.assertRaises(ValueError) as context:
            get_local_rank(logger_name={'name': 'logger'})
        
        self.assertIn("logger_name must be a str", str(context.exception))

    # =========================================================================
    # LOCAL_RANK Environment Variable Tests
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': '0'})
    def test_local_rank_env_valid_zero(self, mock_loggers):
        """Test get_local_rank with LOCAL_RANK=0."""
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 0)
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from LOCAL_RANK env var: 0"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': '1'})
    def test_local_rank_env_valid_one(self, mock_loggers):
        """Test get_local_rank with LOCAL_RANK=1."""
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 1)
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from LOCAL_RANK env var: 1"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': '5'})
    def test_local_rank_env_valid_five(self, mock_loggers):
        """Test get_local_rank with LOCAL_RANK=5."""
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 5)
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from LOCAL_RANK env var: 5"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': 'abc'})
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_local_rank_env_invalid_string(self, mock_ray_context, mock_loggers):
        """Test get_local_rank with invalid LOCAL_RANK (non-numeric string)."""
        # Mock Ray to raise RuntimeError (no Ray context)
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify warning was logged for invalid LOCAL_RANK
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list]
        self.assertTrue(len(warning_calls) > 0)
        warning_msg = str(warning_calls[0])
        self.assertIn("Invalid LOCAL_RANK value", warning_msg)
        self.assertIn("abc", warning_msg)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': 'invalid'})
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_local_rank_env_invalid_non_numeric(self, mock_ray_context, mock_loggers):
        """Test get_local_rank with invalid LOCAL_RANK (non-numeric)."""
        # Mock Ray to raise RuntimeError (no Ray context)
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify warning was logged
        self.assertTrue(mock_logger_instance.warning.called)

    # =========================================================================
    # Ray Runtime Context Tests
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_npu_present(self, mock_ray_context, mock_loggers):
        """Test get_local_rank with NPU accelerator IDs from Ray."""
        # Mock Ray runtime context with NPU accelerator
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": [2]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 2)
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from Ray runtime context: 2"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_npu_multiple_ids(self, mock_ray_context, mock_loggers):
        """Test get_local_rank uses first NPU ID when multiple present."""
        # Mock Ray runtime context with multiple NPU accelerators
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": [3, 4, 5]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should use the first NPU ID
        self.assertEqual(result, 3)
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from Ray runtime context: 3"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_npu_empty_list(self, mock_ray_context, mock_loggers):
        """Test get_local_rank falls back when NPU list is empty."""
        # Mock Ray runtime context with empty NPU list
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": []}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify debug log for no NPU IDs
        mock_logger_instance.warning.assert_called_once_with(
            "Ray runtime context available but no NPU accelerator IDs found"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_no_npu_key(self, mock_ray_context, mock_loggers):
        """Test get_local_rank falls back when NPU key not present."""
        # Mock Ray runtime context without NPU key
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"GPU": [0, 1]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify debug log for no NPU IDs
        mock_logger_instance.warning.assert_called_once_with(
            "Ray runtime context available but no NPU accelerator IDs found"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_runtime_error(self, mock_ray_context, mock_loggers):
        """Test get_local_rank handles RuntimeError from Ray."""
        # Mock Ray to raise RuntimeError
        mock_ray_context.side_effect = RuntimeError("Ray not initialized")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify debug log for Ray error
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list]
        self.assertTrue(len(warning_calls) > 0)
        warning_msg = str(warning_calls[0])
        self.assertIn("Could not get local rank from Ray runtime context", warning_msg)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_key_error(self, mock_ray_context, mock_loggers):
        """Test get_local_rank handles KeyError from Ray."""
        # Mock Ray to raise KeyError
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.side_effect = KeyError("NPU")
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify debug log for Ray error
        debug_calls = [call for call in mock_logger_instance.warning.call_args_list]
        self.assertTrue(len(debug_calls) > 0)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_index_error(self, mock_ray_context, mock_loggers):
        """Test get_local_rank handles IndexError from Ray."""
        # Mock Ray context that causes IndexError when accessing first element
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": []}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_value_error(self, mock_ray_context, mock_loggers):
        """Test get_local_rank handles ValueError from Ray."""
        # Mock Ray context that causes ValueError when converting to int
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": ["not_a_number"]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify warning log for Ray error
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list]
        self.assertTrue(len(warning_calls) > 0)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_context_unexpected_exception(self, mock_ray_context, mock_loggers):
        """Test get_local_rank handles unexpected exceptions from Ray."""
        # Mock Ray to raise unexpected exception
        mock_ray_context.side_effect = Exception("Unexpected error")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        # Should fall back to default 0
        self.assertEqual(result, 0)
        
        # Verify warning was logged for unexpected error
        mock_logger_instance.warning.assert_called_once()
        warning_msg = str(mock_logger_instance.warning.call_args)
        self.assertIn("Unexpected error accessing Ray runtime context", warning_msg)

    # =========================================================================
    # Fallback Behavior Test
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_fallback_to_default_zero(self, mock_ray_context, mock_loggers):
        """Test get_local_rank defaults to 0 when all methods fail."""
        # Mock Ray to raise RuntimeError (no Ray context)
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 0)
        
        # Verify info log for default fallback
        info_calls = [call for call in mock_logger_instance.info.call_args_list]
        self.assertTrue(len(info_calls) > 0)
        final_info = str(info_calls[-1])
        self.assertIn("Unable to determine local rank", final_info)
        self.assertIn("Defaulting to 0", final_info)

    # =========================================================================
    # Environment Variable Side Effects Tests
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_ray_sets_local_rank_env_var(self, mock_ray_context, mock_loggers):
        """Test that Ray path sets os.environ['LOCAL_RANK']."""
        # Start with no LOCAL_RANK in environment
        import os
        if 'LOCAL_RANK' in os.environ:
            del os.environ['LOCAL_RANK']
        
        # Mock Ray runtime context with NPU accelerator
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": [7]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 7)
        
        # Verify LOCAL_RANK was set in environment
        self.assertEqual(os.environ.get('LOCAL_RANK'), '7')
        
        # Clean up
        if 'LOCAL_RANK' in os.environ:
            del os.environ['LOCAL_RANK']

    # =========================================================================
    # Logging Verification Tests
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': '3'})
    def test_logging_info_local_rank_path(self, mock_loggers):
        """Test info logging for LOCAL_RANK path."""
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 3)
        
        # Verify info log message
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from LOCAL_RANK env var: 3"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_logging_info_ray_path(self, mock_ray_context, mock_loggers):
        """Test info logging for Ray path."""
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {"NPU": [4]}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 4)
        
        # Verify info log message
        mock_logger_instance.info.assert_called_once_with(
            "Local rank determined from Ray runtime context: 4"
        )

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': 'not_a_number'})
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_logging_warning_invalid_local_rank(self, mock_ray_context, mock_loggers):
        """Test warning logging for invalid LOCAL_RANK."""
        mock_ray_context.side_effect = RuntimeError("No runtime context")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 0)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_logging_debug_ray_failure(self, mock_ray_context, mock_loggers):
        """Test debug logging for Ray failures."""
        mock_ray_context.side_effect = RuntimeError("Ray not available")
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 0)
        
        # Verify debug log for Ray error
        debug_calls = [call for call in mock_logger_instance.warning.call_args_list]
        self.assertTrue(len(debug_calls) > 0)
        debug_msg = str(debug_calls[0])
        self.assertIn("Could not get local rank from Ray runtime context", debug_msg)

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {}, clear=True)
    @patch('agentic_rl.base.utils.get_local_rank.ray.get_runtime_context')
    def test_logging_debug_no_npu_ids(self, mock_ray_context, mock_loggers):
        """Test debug logging when Ray has no NPU accelerator IDs."""
        mock_context = MagicMock()
        mock_context.get_accelerator_ids.return_value = {}
        mock_ray_context.return_value = mock_context
        
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 0)
        
        # Verify debug log message
        mock_logger_instance.warning.assert_called_once_with(
            "Ray runtime context available but no NPU accelerator IDs found"
        )

    # =========================================================================
    # Lazy Logger Initialization Test
    # =========================================================================

    @patch('agentic_rl.base.utils.get_local_rank.Loggers')
    @patch.dict('os.environ', {'LOCAL_RANK': '2'})
    def test_lazy_logger_initialization(self, mock_loggers):
        """Test that logger is only created when needed for logging."""
        mock_logger_instance = MagicMock()
        mock_loggers.return_value = mock_logger_instance
        
        result = get_local_rank()
        
        self.assertEqual(result, 2)
        
        # Verify logger was created (needed for info logging)
        mock_loggers.assert_called_once()


if __name__ == "__main__":
    unittest.main()
