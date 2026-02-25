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

from agentic_rl.data_manager.data_manager import DataManager


class TestDataManagerInit(unittest.TestCase):
    def setUp(self):
        self.data_manager = DataManager()
        self.data_manager.data_manager_instance = MagicMock()

    def test_init_with_valid_train_backend(self):
        train_backend = "mindspeed_rl"
        with patch('agentic_rl.data_manager.data_manager.data_manager_class',
                   return_value=lambda: self.data_manager.data_manager_instance):
            dm = DataManager(train_backend)

        self.assertEqual(dm.train_backend, train_backend)
        self.assertEqual(dm.data_manager_instance, self.data_manager.data_manager_instance)

    def test_init_with_empty_train_backend(self):
        train_backend = ""
        with self.assertRaises(ValueError) as context:
            DataManager(train_backend)

        self.assertEqual(str(context.exception), "train_backend cannot be empty")

    def test_init_with_non_string_train_backend(self):
        train_backend = 123
        with self.assertRaises(TypeError) as context:
            DataManager(train_backend)

        self.assertEqual(str(context.exception), "train_backend must be a string")

    def test_sync_init_data_manager_with_none(self):
        with self.assertRaises(ValueError) as context:
            self.data_manager.sync_init_data_manager(None)

        self.assertIn('data_manager is None', str(context.exception))

    def test_sync_init_data_manager_success(self):
        test_data_manager = MagicMock()
        self.data_manager.sync_init_data_manager(test_data_manager)

        self.data_manager.data_manager_instance.sync_init_data_manager.assert_called_once_with(test_data_manager)

    def test_all_consumed_with_valid_stage(self):
        stage = "valid_stage"
        self.data_manager.data_manager_instance.all_consumed.return_value = True

        result = self.data_manager.all_consumed(stage)

        self.assertTrue(result)
        self.data_manager.data_manager_instance.all_consumed.assert_called_once_with(stage)

    def test_all_consumed_with_none_stage(self):
        stage = None

        with self.assertRaises(ValueError) as context:
            self.data_manager.all_consumed(stage)

        self.assertEqual(str(context.exception), "experience_consumer_stage is None or empty")

    def test_all_consumed_with_empty_stage(self):
        stage = ""

        with self.assertRaises(ValueError) as context:
            self.data_manager.all_consumed(stage)

        self.assertEqual(str(context.exception), "experience_consumer_stage is None or empty")

    def test_get_data_experience_consumer_stage_none(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data(None, ['column1', 'column2'], 10, True)

    def test_get_data_experience_consumer_stage_empty(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("", ['column1', 'column2'], 10, True)

    def test_get_data_experience_columns_not_list(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("stage1", "column1", 10, True)

    def test_get_data_experience_columns_empty(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("stage1", [], 10, True)

    def test_get_data_experience_count_not_int(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("stage1", ['column1', 'column2'], "10", True)

    def test_get_data_experience_count_zero(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("stage1", ['column1', 'column2'], 0, True)

    def test_get_data_experience_count_negative(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_data("stage1", ['column1', 'column2'], -10, True)

    def test_get_data_get_n_samples_not_bool(self):
        with self.assertRaises(TypeError):
            self.data_manager.get_data("stage1", ['column1', 'column2'], 10, "True")

    def test_get_data_success(self):
        self.data_manager.get_data("stage1", ['column1', 'column2'], 10, True)
        self.data_manager.data_manager_instance.get_data.assert_called_once_with("stage1", ['column1', 'column2'], 10,
                                                                                 True)

    def test_put_data_with_valid_output_and_index(self):
        output = "valid_output"
        index = "valid_index"
        self.data_manager.put_data(output, index)
        self.data_manager.data_manager_instance.put_data.assert_called_once_with(output, index)

    def test_put_data_with_none_output(self):
        output = None
        index = "valid_index"
        with self.assertRaises(ValueError) as context:
            self.data_manager.put_data(output, index)
        self.assertEqual(str(context.exception), "output cannot be None")

    def test_put_data_with_none_index(self):
        output = "valid_output"
        index = None
        with self.assertRaises(ValueError) as context:
            self.data_manager.put_data(output, index)
        self.assertEqual(str(context.exception), "index cannot be None")

    def test_put_data_with_none_output_and_index(self):
        output = None
        index = None
        with self.assertRaises(ValueError) as context:
            self.data_manager.put_data(output, index)
        self.assertEqual(str(context.exception), "output cannot be None")

    def test_update_metrics_with_valid_input(self):
        self.data_manager.update_metrics("test_key", [1, 2, 3], True)
        self.data_manager.data_manager_instance.update_metrics.assert_called_once_with("test_key", [1, 2, 3], True)

    def test_update_metrics_with_none_key(self):
        with self.assertRaises(ValueError):
            self.data_manager.update_metrics(None, [1, 2, 3], True)

    def test_update_metrics_with_empty_key(self):
        with self.assertRaises(ValueError):
            self.data_manager.update_metrics("", [1, 2, 3], True)

    def test_update_metrics_with_non_string_key(self):
        with self.assertRaises(ValueError):
            self.data_manager.update_metrics(123, [1, 2, 3], True)

    def test_update_metrics_with_non_number_value(self):
        with self.assertRaises(TypeError):
            self.data_manager.update_metrics("test_key", ["1", "2", "3"], True)

    def test_update_metrics_with_non_boolean_cumulate(self):
        with self.assertRaises(TypeError):
            self.data_manager.update_metrics("test_key", [1, 2, 3], "True")

    def test_reset_experience_len_with_positive_integer(self):
        experience_len = 10
        self.data_manager.reset_experience_len(experience_len)

    def test_reset_experience_len_with_zero(self):
        with self.assertRaises(ValueError):
            self.data_manager.reset_experience_len(0)

    def test_reset_experience_len_with_negative_integer(self):
        with self.assertRaises(ValueError):
            self.data_manager.reset_experience_len(-10)

    def test_reset_experience_len_with_non_integer(self):
        with self.assertRaises(ValueError):
            self.data_manager.reset_experience_len(3.14)

    def test_reset_experience_len_with_string(self):
        with self.assertRaises(ValueError):
            self.data_manager.reset_experience_len("10")

    def test_reset_experience_len_with_none(self):
        with self.assertRaises(ValueError):
            self.data_manager.reset_experience_len(None)


if __name__ == '__main__':
    unittest.main()