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

import torch
from tensordict import TensorDict
import unittest

from agentic_rl.data_manager.data_transform import padding_dict_to_tensor_dict


class TestPaddingDictToTensorDict(unittest.TestCase):
    def test_empty_experience_data(self):
        with self.assertRaises(ValueError):
            padding_dict_to_tensor_dict({})

    def test_non_dict_experience_data(self):
        with self.assertRaises(TypeError):
            padding_dict_to_tensor_dict("not a dict")

    def test_non_tensor_list_experience_data(self):
        with self.assertRaises(TypeError):
            padding_dict_to_tensor_dict({"experience": ["not a tensor"]})

    def test_scalar_tensor_experience_data(self):
        with self.assertRaises(ValueError):
            padding_dict_to_tensor_dict({"experience": torch.tensor(1)})

    def test_non_str_key_experience_data(self):
        with self.assertRaises(TypeError):
            padding_dict_to_tensor_dict({1: torch.tensor([1, 2])})

    def test_valid_experience_data(self):
        experience_data = {"experience": [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]}
        result = padding_dict_to_tensor_dict(experience_data)
        self.assertIsInstance(result, TensorDict)
        self.assertEqual(result["experience"].shape, (2, 3))
        self.assertEqual(result["original_length"].tolist(), [2, 3])

if __name__ == "__main__":
    unittest.main()