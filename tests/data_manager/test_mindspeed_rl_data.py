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

import ray
import torch
import unittest
from unittest.mock import patch, MagicMock

from agentic_rl.data_manager.mindspeed_rl_data import MindSpeedRLDataManager


class TestMindSpeedRLDataManager(unittest.TestCase):

    def setUp(self):
        self.mind_speed_rl_data_manager = MindSpeedRLDataManager()
        self.mind_speed_rl_data_manager.data_manager = MagicMock()

    def test_sync_init_data_manager_with_none(self):
        with self.assertRaises(ValueError):
            self.mind_speed_rl_data_manager.sync_init_data_manager(None)

    def test_sync_init_data_manager_without_required_methods(self):
        data_manager = MagicMock()
        delattr(data_manager, 'update_metrics')
        with self.assertRaises(AttributeError):
            self.mind_speed_rl_data_manager.sync_init_data_manager(data_manager)

    def test_sync_init_data_manager_with_required_methods(self):
        data_manager = MagicMock()
        data_manager.all_consumed = MagicMock()
        data_manager.get_experience = MagicMock()
        data_manager.put_experience = MagicMock()
        data_manager.update_metrics = MagicMock()
        self.mind_speed_rl_data_manager.sync_init_data_manager(data_manager)
        self.assertEqual(self.mind_speed_rl_data_manager.data_manager, data_manager)

    @patch('ray.get')
    def test_all_consumed_valid_input(self, mock_ray_get):
        mock_ray_get.return_value = True
        result = self.mind_speed_rl_data_manager.all_consumed('test_stage')
        self.assertEqual(result, 0)
        mock_ray_get.assert_called_once_with(
            self.mind_speed_rl_data_manager.data_manager.all_consumed.remote('test_stage'))

    @patch('ray.get')
    def test_all_consumed_empty_string(self, mock_ray_get):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.all_consumed('')

    @patch('ray.get')
    def test_all_consumed_non_string(self, mock_ray_get):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.all_consumed(123)

    @patch('ray.get')
    def test_all_consumed_ray_error(self, mock_ray_get):
        mock_ray_get.side_effect = ray.exceptions.RayError('Test Ray error')
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.all_consumed('test_stage')

    @patch('ray.get')
    def test_all_consumed_other_error(self, mock_ray_get):
        mock_ray_get.side_effect = Exception('Test other error')
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.all_consumed('test_stage')

    @patch('ray.get')
    def test_get_data_success(self, mock_ray_get):
        mock_ray_get.return_value = ({'data': 'value'}, 'index')
        result = self.mind_speed_rl_data_manager.get_data('stage', ['column'], 1)
        self.assertEqual(result, ({'data': 'value'}, 'index'))

    @patch('ray.get')
    def test_get_data_no_data(self, mock_ray_get):
        mock_ray_get.return_value = ({}, [])
        result = self.mind_speed_rl_data_manager.get_data('stage', ['column'], 1)
        self.assertEqual(result, ({}, []))

    def test_get_data_invalid_experience_consumer_stage(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('', ['column'], 1)

    def test_get_data_invalid_experience_columns(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('stage', [], 1)

    def test_get_data_invalid_experience_count(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('stage', ['column'], 0)

    def test_get_data_invalid_get_n_samples(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('stage', ['column'], 1, 'invalid')

    @patch('ray.get', side_effect=ray.exceptions.RayError('Ray error'))
    def test_get_data_ray_error(self, mock_ray_get):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('stage', ['column'], 1)

    @patch('ray.get', side_effect=Exception('Exception'))
    def test_get_data_exception(self, mock_ray_get):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.get_data('stage', ['column'], 1)

    def test_put_data_with_non_dict_output(self):
        with self.assertRaises(TypeError):
            self.mind_speed_rl_data_manager.put_data("not a dict", [1, 2, 3])

    def test_put_data_with_empty_output(self):
        with patch('logging.Logger.warning') as mock_warning:
            self.mind_speed_rl_data_manager.put_data({}, [1, 2, 3])
            assert mock_warning.call_count == 1
            assert mock_warning.call_args[0] == ("output is empty",)

    def test_put_data_with_invalid_index(self):
        with self.assertRaises(ValueError):
            self.mind_speed_rl_data_manager.put_data({"key": torch.tensor([1, 2, 3])}, [])

    @patch('agentic_rl.data_manager.mindspeed_rl_data.padding_dict_to_tensor_dict')
    def test_put_data_with_other_error(self, mock_padding):
        mock_padding.side_effect = Exception("Other error")
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.put_data({"key": torch.tensor([1, 2, 3])}, [1, 2, 3])

    @patch('agentic_rl.data_manager.mindspeed_rl_data.padding_dict_to_tensor_dict')
    def test_put_data_success(self, mock_padding):
        mock_padding.return_value = {"key": torch.tensor([1, 2, 3])}
        sample_data = {"key": torch.tensor([1, 2, 3])}
        self.mind_speed_rl_data_manager.put_data(sample_data, [1, 2, 3])
        mock_padding.assert_called_once_with(sample_data)
        self.mind_speed_rl_data_manager.data_manager.put_experience.remote.assert_called_once()

    @patch('ray.get')
    @patch('ray.exceptions.RayError')
    def test_update_metrics_success(self, mock_ray_error, mock_ray_get):
        mock_ray_get.return_value = None
        self.mind_speed_rl_data_manager.update_metrics('test_key', [1, 2, 3], True)
        mock_ray_get.assert_called_once()

    def test_update_metrics_empty_key(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics('', [1, 2, 3], True)

    def test_update_metrics_non_string_key(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics(123, [1, 2, 3], True)

    def test_update_metrics_non_number_value(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics('test_key', ['a', 'b', 'c'], True)

    def test_update_metrics_non_boolean_cumulate(self):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics('test_key', [1, 2, 3], 'True')

    @patch('ray.get', side_effect=ray.exceptions.RayError('Ray error'))
    def test_update_metrics_ray_error(self, mock_ray_error):
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics('test_key', [1, 2, 3], True)

    @patch('ray.get')
    def test_update_metrics_other_error(self, mock_ray_get):
        mock_ray_get.side_effect = RuntimeError('Test Other Error')
        with self.assertRaises(RuntimeError):
            self.mind_speed_rl_data_manager.update_metrics('test_key', [1, 2, 3], True)

    def test_reset_experience_len_positive_integer(self):
        self.mind_speed_rl_data_manager.reset_experience_len(10)
        self.mind_speed_rl_data_manager.data_manager.reset_experience_len.remote.assert_called_once_with(10)

    def test_reset_experience_len_non_integer(self):
        with self.assertRaises(ValueError):
            self.mind_speed_rl_data_manager.reset_experience_len(3.14)

    def test_reset_experience_len_zero(self):
        with self.assertRaises(ValueError):
            self.mind_speed_rl_data_manager.reset_experience_len(0)

    def test_reset_experience_len_negative(self):
        with self.assertRaises(ValueError):
            self.mind_speed_rl_data_manager.reset_experience_len(-5)


if __name__ == '__main__':
    unittest.main()