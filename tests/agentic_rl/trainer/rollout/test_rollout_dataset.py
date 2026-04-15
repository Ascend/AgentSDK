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
import unittest
from unittest.mock import MagicMock
import torch

# Import test objects
from agentic_rl.trainer.rollout.rollout_dataset import optimized_preprocess_input, optimized_put_prompt_experience


class TestRolloutDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        pass
    
    def tearDown(self):
        """Clean up test environment"""
        pass
    def test_optimized_preprocess_input(self):
        """Test the normal flow of optimized_preprocess_input function"""
        # Prepare test data
        batch = {
            "mini_batch_id": [0, 0, 1, 1, 1],
            "input_ids": [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8, 9]),
                torch.tensor([10, 11, 12]),
                torch.tensor([13, 14, 15])
            ],
            "prompt_id": [0, 1, 2, 3, 4]
        }

        # Call the function
        mini_batches, prompt_ids = optimized_preprocess_input(batch)

        # Verify results
        self.assertEqual(len(mini_batches), 2)
        self.assertEqual(len(prompt_ids), 2)
        self.assertEqual(len(mini_batches[0]), 2)
        self.assertEqual(len(mini_batches[1]), 3)
        self.assertEqual(len(prompt_ids[0]), 2)
        self.assertEqual(len(prompt_ids[1]), 3)
        self.assertTrue(torch.equal(mini_batches[0][0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(mini_batches[0][1], torch.tensor([4, 5, 6])))
        self.assertTrue(torch.equal(mini_batches[1][0], torch.tensor([7, 8, 9])))
        self.assertTrue(torch.equal(mini_batches[1][1], torch.tensor([10, 11, 12])))
        self.assertTrue(torch.equal(mini_batches[1][2], torch.tensor([13, 14, 15])))
        self.assertEqual(prompt_ids[0], [0, 1])
        self.assertEqual(prompt_ids[1], [2, 3, 4])

    def test_optimized_preprocess_input_empty_batch(self):
        """Test optimized_preprocess_input function with empty batch"""
        # Prepare test data
        batch = {
            "mini_batch_id": [],
            "input_ids": [],
            "prompt_id": []
        }

        # Call the function
        mini_batches, prompt_ids = optimized_preprocess_input(batch)

        # Verify results
        self.assertEqual(len(mini_batches), 0)
        self.assertEqual(len(prompt_ids), 0)

    def test_optimized_preprocess_input_single_batch(self):
        """Test optimized_preprocess_input function with single batch"""
        # Prepare test data
        batch = {
            "mini_batch_id": [0],
            "input_ids": [torch.tensor([1, 2, 3])],
            "prompt_id": [0]
        }

        # Call the function
        mini_batches, prompt_ids = optimized_preprocess_input(batch)

        # Verify results
        self.assertEqual(len(mini_batches), 1)
        self.assertEqual(len(prompt_ids), 1)
        self.assertEqual(len(mini_batches[0]), 1)
        self.assertEqual(len(prompt_ids[0]), 1)
        self.assertTrue(torch.equal(mini_batches[0][0], torch.tensor([1, 2, 3])))
        self.assertEqual(prompt_ids[0], [0])

    def test_optimized_put_prompt_experience(self):
        """Test the normal flow of optimized_put_prompt_experience function"""
        # Prepare test data
        mini_batch = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6])
        ]
        prompts_ids = [[0], [1]]

        # Call the function
        data_dict, indexes = optimized_put_prompt_experience(mini_batch, prompts_ids)

        # Verify results
        self.assertEqual(len(data_dict), 2)
        self.assertIn("prompt_length", data_dict)
        self.assertIn("prompts", data_dict)
        self.assertEqual(len(data_dict["prompt_length"]), 2)
        self.assertEqual(len(data_dict["prompts"]), 2)
        self.assertTrue(torch.equal(data_dict["prompt_length"][0], torch.tensor([3])))
        self.assertTrue(torch.equal(data_dict["prompt_length"][1], torch.tensor([3])))
        self.assertTrue(torch.equal(data_dict["prompts"][0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(data_dict["prompts"][1], torch.tensor([4, 5, 6])))
        self.assertEqual(indexes, [0, 1])

    def test_optimized_put_prompt_experience_with_dict_to_tensor_dict(self):
        """Test optimized_put_prompt_experience function with dict_to_tensor_dict"""
        # Prepare test data
        mini_batch = [torch.tensor([1, 2, 3])]
        prompts_ids = [[0]]
        mock_dict_to_tensor_dict = MagicMock(return_value={"processed": "data"})

        # Call the function
        data_dict, indexes = optimized_put_prompt_experience(mini_batch, prompts_ids, mock_dict_to_tensor_dict)

        # Verify results
        mock_dict_to_tensor_dict.assert_called_once()
        self.assertEqual(data_dict, {"processed": "data"})
        self.assertEqual(indexes, [0])

    def test_optimized_put_prompt_experience_empty_input(self):
        """Test optimized_put_prompt_experience function with empty input"""
        # Prepare test data
        mini_batch = []
        prompts_ids = []

        # Call the function
        data_dict, indexes = optimized_put_prompt_experience(mini_batch, prompts_ids)

        # Verify results
        self.assertEqual(len(data_dict), 2)
        self.assertIn("prompt_length", data_dict)
        self.assertIn("prompts", data_dict)
        self.assertEqual(len(data_dict["prompt_length"]), 0)
        self.assertEqual(len(data_dict["prompts"]), 0)
        self.assertEqual(indexes, [])


if __name__ == '__main__':
    unittest.main()
