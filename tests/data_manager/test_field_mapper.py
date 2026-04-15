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

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'torch': mock_torch,
}):
    from agentic_rl.data_manager.field_mapper import FieldMapper


class TestConvertBatch(unittest.TestCase):
    """Tests for FieldMapper.convert_batch classmethod."""

    def setUp(self):
        mock_torch.reset_mock()

        # Build two mock prompt tensors and two mock response tensors
        self.prompt_0 = MagicMock()
        self.prompt_0.size.return_value = 3
        self.prompt_0.device = "cpu"

        self.prompt_1 = MagicMock()
        self.prompt_1.size.return_value = 2

        self.response_0 = MagicMock()
        self.response_0.size.return_value = 5
        self.response_0.device = "cpu"

        self.response_1 = MagicMock()
        self.response_1.size.return_value = 4

        # Common raw_data with required fields
        self.raw_data = {
            'prompts': [self.prompt_0, self.prompt_1],
            'responses': [self.response_0, self.response_1],
        }

        # mock_torch.full and mock_torch.zeros return indexable MagicMock objects
        mock_torch.full.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()
        mock_torch.long = "long"
        mock_torch.float32 = "float32"

    def tearDown(self):
        mock_torch.reset_mock()

    def test_convert_batch_basic(self):
        """Verify convert_batch builds the batch dict with correct keys."""
        batch = FieldMapper.convert_batch(self.raw_data, pad_token_id=0)

        # Batch should have the standard keys
        expected_keys = {
            "prompts", "responses", "input_ids", "attention_mask",
            "response_mask", "position_ids", "rm_scores",
            "token_level_rewards", "rollout_log_probs",
        }
        self.assertTrue(expected_keys.issubset(set(batch.keys())))

        # torch.full should be called for prompts, responses, input_ids
        self.assertTrue(mock_torch.full.called)
        # torch.zeros should be called for attention_mask, response_mask, etc.
        self.assertTrue(mock_torch.zeros.called)

    def test_convert_batch_with_prompt_ids(self):
        """When prompt_ids is present in raw_data, batch should include _prompt_id."""
        self.raw_data['prompt_ids'] = [100, 200]
        batch = FieldMapper.convert_batch(self.raw_data, pad_token_id=0)
        self.assertIn('_prompt_id', batch)
        self.assertEqual(batch['_prompt_id'], [100, 200])

    def test_convert_batch_without_prompt_ids(self):
        """When prompt_ids is absent, batch should not include _prompt_id."""
        batch = FieldMapper.convert_batch(self.raw_data, pad_token_id=0)
        self.assertNotIn('_prompt_id', batch)

    def test_convert_batch_custom_pad_token(self):
        """Verify custom pad_token_id is passed to torch.full."""
        FieldMapper.convert_batch(self.raw_data, pad_token_id=99)
        # Check that torch.full was called with pad_token_id=99 at least once
        found_custom_pad = False
        for c in mock_torch.full.call_args_list:
            # Positional: (shape, pad_value, ...) or keyword
            if len(c.args) >= 2 and c.args[1] == 99:
                found_custom_pad = True
                break
        self.assertTrue(found_custom_pad, "torch.full should be called with pad_token_id=99")


class TestProcessSingleSample(unittest.TestCase):
    """Tests for FieldMapper._process_single_sample classmethod."""

    def setUp(self):
        mock_torch.reset_mock()

        # Build a batch dict with indexable MagicMock tensors
        self.batch = {
            "prompts": MagicMock(),
            "responses": MagicMock(),
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "response_mask": MagicMock(),
            "position_ids": MagicMock(),
            "rm_scores": MagicMock(),
            "token_level_rewards": MagicMock(),
            "rollout_log_probs": MagicMock(),
        }

        # Prompt tensor
        self.prompt = MagicMock()
        self.prompt.squeeze.return_value = self.prompt
        self.prompt.size.return_value = 3

        # Response tensor
        self.response = MagicMock()
        self.response.squeeze.return_value = self.response
        self.response.size.return_value = 5

        # Cumsum mock for position_ids
        mock_torch.cumsum.return_value = MagicMock()

    def tearDown(self):
        mock_torch.reset_mock()

    def test_process_single_sample_basic(self):
        """Basic call with no optional fields should not raise."""
        raw_data = {
            'prompts': [self.prompt],
            'responses': [self.response],
        }
        # Should not raise
        FieldMapper._process_single_sample(
            batch=self.batch, idx=0, raw_data=raw_data,
            prompts_list=raw_data['prompts'],
            responses_list=raw_data['responses'],
            prompt_length=4, response_length=6,
        )
        self.prompt.squeeze.assert_called_once()
        self.response.squeeze.assert_called_once()

    def test_process_single_sample_with_response_mask(self):
        """When response_mask is present, it should be applied to batch."""
        rm_tensor = MagicMock()
        rm_tensor.squeeze.return_value = rm_tensor
        rm_tensor.size.return_value = 5
        raw_data = {
            'prompts': [self.prompt],
            'responses': [self.response],
            'response_mask': [rm_tensor],
        }
        FieldMapper._process_single_sample(
            batch=self.batch, idx=0, raw_data=raw_data,
            prompts_list=raw_data['prompts'],
            responses_list=raw_data['responses'],
            prompt_length=4, response_length=6,
        )
        rm_tensor.squeeze.assert_called_once()

    def test_process_single_sample_with_rm_scores(self):
        """When rm_scores is present, it should be written to batch."""
        score_tensor = MagicMock()
        score_tensor.flatten.return_value = score_tensor
        score_tensor.size.return_value = 5
        raw_data = {
            'prompts': [self.prompt],
            'responses': [self.response],
            'rm_scores': [score_tensor],
        }
        FieldMapper._process_single_sample(
            batch=self.batch, idx=0, raw_data=raw_data,
            prompts_list=raw_data['prompts'],
            responses_list=raw_data['responses'],
            prompt_length=4, response_length=6,
        )
        score_tensor.flatten.assert_called_once()

    def test_process_single_sample_with_rollout_log_probs(self):
        """When rollout_log_probs is present, it should be written to batch."""
        logprob_tensor = MagicMock()
        logprob_tensor.flatten.return_value = logprob_tensor
        logprob_tensor.size.return_value = 5
        raw_data = {
            'prompts': [self.prompt],
            'responses': [self.response],
            'rollout_log_probs': [logprob_tensor],
        }
        FieldMapper._process_single_sample(
            batch=self.batch, idx=0, raw_data=raw_data,
            prompts_list=raw_data['prompts'],
            responses_list=raw_data['responses'],
            prompt_length=4, response_length=6,
        )
        logprob_tensor.flatten.assert_called_once()


class TestConvertDataprotoToMsrl(unittest.TestCase):
    """Tests for FieldMapper.convert_dataproto_to_msrl classmethod."""

    def setUp(self):
        mock_torch.reset_mock()

    def tearDown(self):
        mock_torch.reset_mock()

    def test_basic_conversion(self):
        """Verify batch and non_tensor_batch are merged into the result."""
        mock_data_proto = MagicMock(spec=[])
        mock_input_ids = MagicMock()
        mock_attention_mask = MagicMock()
        mock_data_proto.batch = {"input_ids": mock_input_ids, "attention_mask": mock_attention_mask}
        mock_data_proto.non_tensor_batch = {"uid": [1, 2, 3]}

        result = FieldMapper.convert_dataproto_to_msrl(mock_data_proto)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("uid", result)
        self.assertEqual(result["uid"], [1, 2, 3])

    def test_empty_non_tensor_batch(self):
        """Verify conversion works when non_tensor_batch is empty."""
        mock_data_proto = MagicMock(spec=[])
        mock_responses = MagicMock()
        mock_data_proto.batch = {"responses": mock_responses}
        mock_data_proto.non_tensor_batch = {}

        result = FieldMapper.convert_dataproto_to_msrl(mock_data_proto)

        self.assertIn("responses", result)
        self.assertEqual(len(result), 1)

    def test_none_batch(self):
        """When batch is None, only non_tensor_batch keys should appear."""
        mock_data_proto = MagicMock(spec=[])
        mock_data_proto.batch = None
        mock_data_proto.non_tensor_batch = {"meta": "info"}

        result = FieldMapper.convert_dataproto_to_msrl(mock_data_proto)

        self.assertIn("meta", result)
        self.assertEqual(result["meta"], "info")
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
