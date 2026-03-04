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
import sys
import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch


class TestComputeAdvantage:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.utils": MagicMock(),
            "mindspeed_rl.utils.pad_process": MagicMock(),
            "mindspeed_rl.trainer": MagicMock(),
            "mindspeed_rl.trainer.utils": MagicMock(),
            "mindspeed_rl.trainer.utils.transfer_dock": MagicMock(),
            "mindspeed_rl.trainer.utils.compute_utils": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_utils": MagicMock()
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import AdvantageComputationConfig
        config = AdvantageComputationConfig
        config.gamma = 0.99
        config.lam = 0.95
        config.n_sample_per_prompt = 2
        config.use_stepwise_advantage = True
        config.use_kl_in_reward = False
        config.adv_estimator = "gae"
        config.tokenizer = MagicMock()
        config.global_batch_size = 10
        config.guarantee_order = True
        config.tokenizer.pad = 0
        config.tokenizer.eod = 1

        with patch('mindspeed_rl.utils.utils.generate_mask') as mock_generate_mask, \
                patch("mindspeed_rl.utils.utils.get_current_dp_range_indexes") as mock_get_current_dp_range_indexes, \
                patch("mindspeed_rl.utils.utils.extract_from_dict") as mock_extract_from_dict, \
                patch("mindspeed_rl.utils.pad_process.truncate_rows") as mock_truncate_rows, \
                patch("mindspeed_rl.utils.pad_process.remove_padding_tensor_dict_to_dict") as mock_remove_padding_tensor_dict_to_dict, \
                patch("mindspeed_rl.utils.pad_process.padding_dict_to_tensor_dict") as mock_padding_dict_to_tensor_dict, \
                patch("mindspeed_rl.trainer.utils.transfer_dock.pad_experience") as mock_pad_experience, \
                patch("mindspeed_rl.trainer.utils.compute_utils.compute_gae_advantage_return") as mock_compute_gae_advantage_return, \
                patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_utils.compute_group_norm_advantage_return_patch') as mock_compute_group_norm_advantage_return_patch:

            patches = {"config": config,
                       "mock_generate_mask": mock_generate_mask,
                       "mock_get_current_dp_range_indexes": mock_get_current_dp_range_indexes,
                       "mock_extract_from_dict": mock_extract_from_dict,
                       "mock_truncate_rows": mock_truncate_rows,
                       "mock_remove_padding_tensor_dict_to_dict": mock_remove_padding_tensor_dict_to_dict,
                       "mock_padding_dict_to_tensor_dict": mock_padding_dict_to_tensor_dict,
                       "mock_pad_experience": mock_pad_experience,
                       "mock_compute_gae_advantage_return": mock_compute_gae_advantage_return,
                       "mock_compute_group_norm_advantage_return_patch": mock_compute_group_norm_advantage_return_patch}
            yield patches

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.time')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_advantage_utils')
    def test_compute_advantage_blocking_true(self, mock_utils, mock_time, mock_ray_get, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import compute_advantage
        # Arrange
        mock_time = MagicMock()
        mock_time.time = MagicMock()
        mock_time.time.side_effect = [100.0, 200.0]
        mock_ray_get.return_value = None  # Mock ray.get
        mock_utils.options.return_value.remote.return_value = MagicMock()

        # Create an instance of the class containing compute_advantage
        instance = MagicMock()
        instance.micro_batch_size = 10
        instance.gamma = 0.99
        instance.lam = 0.95
        instance.adv_estimator = 'gae'
        instance.tokenizer = MagicMock()
        instance.global_batch_size = 100
        instance.n_samples_per_prompt = 5
        instance.actor_worker.rl_config.n_samples_per_prompt = 5
        instance.use_stepwise_advantage = False
        instance.num_cpus_for_local_task = 1
        instance.transfer_dock = MagicMock()

        instance.compute_advantage = compute_advantage

        # Act
        compute_advantage(instance, blocking=True, guarantee_order=False)

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.time')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_advantage_utils')
    def test_compute_advantage_blocking_false(self, mock_utils, mock_time, mock_ray_get, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import compute_advantage
        # Arrange
        mock_time = MagicMock()
        mock_time.time = MagicMock()
        mock_time.time.side_effect = [100.0, 200.0]
        mock_ray_get.return_value = None  # Mock ray.get
        mock_utils.options.return_value.remote.return_value = MagicMock()

        # Create an instance of the class containing compute_advantage
        instance = MagicMock()
        instance.micro_batch_size = 10
        instance.gamma = 0.99
        instance.lam = 0.95
        instance.adv_estimator = 'gae'
        instance.tokenizer = MagicMock()
        instance.global_batch_size = 100
        instance.n_samples_per_prompt = 5
        instance.actor_worker.rl_config.n_samples_per_prompt = 5
        instance.use_stepwise_advantage = False
        instance.num_cpus_for_local_task = 1
        instance.transfer_dock = MagicMock()
        instance.compute_advantage = compute_advantage      # 必须手动传入instance

        # Act
        compute_advantage(instance, blocking=False, guarantee_order=False)

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.get_current_dp_range_indexes')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.remove_padding_tensor_dict_to_dict')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage._process_single_batch')
    def test_compute_advantage_utils_gae_with_kl(self, mocke_process_single_batch, mock_remove_padding, mock_get_indexes, mock_ray_get, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import AdvantageComputationConfig, compute_advantage_utils
        # Mock the AdvantageComputationConfig
        config = AdvantageComputationConfig
        config.adv_estimator = "gae"
        config.use_kl_in_reward = True
        config.tokenizer = MagicMock()
        config.tokenizer.pad = 1
        config.tokenizer.eod = 2
        config.global_batch_size = 10
        config.guarantee_order = True

        # Mock the data queue object
        td = MagicMock()
        td.get_experience_len.return_value = 100
        td.all_consumed.return_value = False
        td.get_experience.return_value = (
        {"values": [1, 2], "responses": [3, 4], "token_level_rewards": [5, 6], "response_length": [7, 8]}, [0, 1])
        td.get_experience_len.return_value = 100

        # Mock the get_current_dp_range_indexes function

        compute_advantage_utils._function(td, config)

        # Assertions
        mock_get_indexes.assert_called_once()
        mock_ray_get.assert_called()

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.get_current_dp_range_indexes')
    def test_compute_advantage_utils_gae_without_kl(self, mock_get_indexes, mock_ray_get, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import AdvantageComputationConfig, compute_advantage_utils
        mock_get_indexes.return_value = [0, 1]
        # Mock the AdvantageComputationConfig
        config = AdvantageComputationConfig
        config.adv_estimator = "gae"
        config.use_kl_in_reward = False
        config.tokenizer = MagicMock()
        config.tokenizer.pad = 1
        config.tokenizer.eod = 2
        config.global_batch_size = 10
        config.guarantee_order = True

        # Mock the data queue object
        td = MagicMock()
        td.get_experience_len.remote.return_value = 100
        td.all_consumed.remote.return_value = False
        td.get_experience.remote.return_value = ({"values": [1, 2], "responses": [3, 4], "rm_scores": [5, 6], "response_length": [7, 8]}, [0, 1])
        td.get_experience_len.remote.return_value = 100

        # Mock the get_current_dp_range_indexes function

        # Call the function
        compute_advantage_utils._function(td, config)

        # Assertions
        mock_get_indexes.assert_called_once()
        mock_ray_get.assert_called()

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.get_current_dp_range_indexes')
    def test_compute_advantage_utils_other_estimator(self, mock_get_indexes, mock_ray_get, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import AdvantageComputationConfig, compute_advantage_utils
        mock_get_indexes.return_value = [0, 1]
        # Mock the AdvantageComputationConfig
        config = AdvantageComputationConfig
        config.adv_estimator = "other_estimator"
        config.tokenizer = MagicMock()
        config.tokenizer.pad = 1
        config.tokenizer.eod = 2
        config.global_batch_size = 10
        config.guarantee_order = True

        # Mock the data queue object
        td = MagicMock()
        td.get_experience_len.remote.return_value = 100
        td.all_consumed.remote.return_value = False
        td.get_experience.remote.return_value = ({"responses": [3, 4], "rm_scores": [5, 6], "response_length": [7, 8]}, [0, 1])
        td.get_experience_len.remote.return_value = 100

        # Mock the get_current_dp_range_indexes function

        # Call the function
        compute_advantage_utils._function(td, config)

        # Assertions
        mock_get_indexes.assert_called_once()
        mock_ray_get.assert_called()

    @patch('ray.get')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.get_current_dp_range_indexes')
    def test_compute_advantage_utils_no_guarantee_order(self, mock_get_indexes, mock_ray_get):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import AdvantageComputationConfig, \
            compute_advantage_utils
        # Mock the AdvantageComputationConfig
        config = AdvantageComputationConfig
        config.adv_estimator = "other_estimator"
        config.tokenizer = MagicMock()
        config.tokenizer.pad = 1
        config.tokenizer.eod = 2
        config.global_batch_size = 10
        config.guarantee_order = False

        # Mock the data queue object
        td = MagicMock()
        td.get_experience_len.remote.return_value = 100
        td.all_consumed.remote.return_value = False
        td.get_experience.remote.return_value = ({"responses": [3, 4], "rm_scores": [5, 6], "response_length": [7, 8]}, [0, 1])
        td.get_experience_len.remote.return_value = 100

        # Mock the get_current_dp_range_indexes function
        # Call the function
        compute_advantage_utils._function(td, config)

        # Assertions
        mock_get_indexes.assert_not_called()
        mock_ray_get.assert_called()

    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.pad_experience')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.generate_mask')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_gae_advantage_return')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_group_norm_advantage_return_patch')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.truncate_rows')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.padding_dict_to_tensor_dict')
    def test_process_single_batch_gae_estimator(self, mock_padding_dict_to_tensor_dict, mock_truncate_rows,
                                                mock_compute_group_norm_advantage_return_patch,
                                                mock_compute_gae_advantage_return, mock_generate_mask,
                                                mock_pad_experience, patch_target):
        # Mock input data
        mock_td = MagicMock()
        mock_batch_data = {
            "responses": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "response_length": torch.tensor([3, 2]),
            "values": torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.0, 0.0]]),
            "rm_scores": torch.tensor([0.8, 0.9]),
            "token_level_rewards": torch.tensor([[0.0, 0.0, 0.8, 0.0], [0.0, 0.9, 0.0, 0.0]])
        }
        mock_index = [0, 1]
        mock_pad_token_id = 0

        # Mock return values of functions
        mock_pad_experience.return_value = mock_batch_data
        mock_generate_mask.return_value = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        mock_compute_gae_advantage_return.return_value = (torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.0, 0.0]]),

                                                          torch.tensor([[0.9, 0.8, 0.7, 0.0], [0.6, 0.5, 0.0, 0.0]]))

        def mock_truncate_rows_side_effect(x, y):
            return x[:, :y.max().item()]

        mock_truncate_rows.side_effect = mock_truncate_rows_side_effect
        mock_padding_dict_to_tensor_dict.return_value = {"advantages": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                                        "returns": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])}

        # Set config to use GAE estimator
        config = patch_target["config"]
        config.adv_estimator = "gae"

        # Call the function under test
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import _process_single_batch
        _process_single_batch(mock_td, mock_batch_data, mock_index, mock_pad_token_id, config)

        # Verify function calls
        mock_pad_experience.assert_called_once_with(mock_batch_data, mock_pad_token_id)
        mock_generate_mask.assert_called_once_with(mock_batch_data["responses"], mock_batch_data["response_length"])
        mock_compute_gae_advantage_return.assert_called_once()

    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.pad_experience')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.generate_mask')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_gae_advantage_return')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.compute_group_norm_advantage_return_patch')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.truncate_rows')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.padding_dict_to_tensor_dict')
    def test_process_single_batch_group_norm_estimator(self, mock_padding_dict_to_tensor_dict, mock_truncate_rows,
                                                       mock_compute_group_norm_advantage_return_patch,
                                                       mock_compute_gae_advantage_return, mock_generate_mask,
                                                       mock_pad_experience, patch_target):
        # Mock input data
        mock_td = MagicMock()
        mock_batch_data = {
            "responses": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "response_length": torch.tensor([3, 2]),
            "values": torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.0, 0.0]]),
            "rm_scores": torch.tensor([0.8, 0.9]),
            "token_level_rewards": torch.tensor([[0.0, 0.0, 0.8, 0.0], [0.0, 0.9, 0.0, 0.0]])
        }
        mock_index = [0, 1]
        mock_pad_token_id = 0

        # Mock return values of functions
        mock_pad_experience.return_value = mock_batch_data
        mock_generate_mask.return_value = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        mock_compute_group_norm_advantage_return_patch.return_value = (torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.0, 0.0]]),
                                                                       torch.tensor([[0.9, 0.8, 0.7, 0.0], [0.6, 0.5, 0.0, 0.0]]))

        def mock_truncate_rows_side_effect(x, y):
            return x[:, :y.max().item()]

        mock_truncate_rows.side_effect = mock_truncate_rows_side_effect
        mock_padding_dict_to_tensor_dict.return_value = {"advantages": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                                        "returns": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])}

        # Set config to use GroupNorm estimator
        config = patch_target["config"]
        config.adv_estimator = "group_norm"

        # Call the function under test
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import _process_single_batch
        _process_single_batch(mock_td, mock_batch_data, mock_index, mock_pad_token_id, config)

        # Verify function calls
        mock_pad_experience.assert_called_once_with(mock_batch_data, mock_pad_token_id)
        mock_generate_mask.assert_called_once_with(mock_batch_data["responses"], mock_batch_data["response_length"])
        mock_compute_group_norm_advantage_return_patch.assert_called_once_with(
            token_level_rewards=mock_batch_data["rm_scores"],
            eos_mask=mock_generate_mask.return_value,
            response_length=mock_batch_data["response_length"],
            n_sample_per_prompt=config.n_sample_per_prompt,
            use_stepwise_advantage=config.use_stepwise_advantage
        )

    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.pad_experience')
    @patch('agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage.generate_mask')
    def test_process_single_batch_unsupported_estimator(self, mock_generate_mask,
                                                        mock_pad_experience, patch_target):
        # Set config to an unsupported estimator
        config = patch_target["config"]
        config.adv_estimator = "unsupported"

        # Mock input data
        mock_td = MagicMock()
        mock_batch_data = {
            "responses": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "response_length": torch.tensor([3, 2]),
            "values": torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.0, 0.0]]),
            "rm_scores": torch.tensor([0.8, 0.9]),
            "token_level_rewards": torch.tensor([[0.0, 0.0, 0.8, 0.0], [0.0, 0.9, 0.0, 0.0]])
        }
        mock_index = [0, 1]
        mock_pad_token_id = 0

        # Verify that a NotImplementedError is raised
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_advantage import _process_single_batch
        with pytest.raises(NotImplementedError):
            _process_single_batch(mock_td, mock_batch_data, mock_index, mock_pad_token_id, config)


if __name__ == '__main__':
    unittest.main()
