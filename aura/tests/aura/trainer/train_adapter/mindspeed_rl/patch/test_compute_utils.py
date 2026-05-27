# -*- coding: utf-8 -*-
import torch
import pytest
from unittest.mock import MagicMock, patch


class TestComputeUtils:

    def test_compute_advantage(self):
        mock_self = MagicMock()
        mock_self.micro_batch_size = 8
        mock_self.num_cpus_for_local_task = 1
        mock_self.gamma = 0.99
        mock_self.lam = 0.95
        mock_self.adv_estimator = "gae"
        mock_self.tokenizer = MagicMock()
        mock_self.tokenizer.pad = 0
        mock_self.global_batch_size = 32
        mock_self.n_samples_per_prompt = 4
        mock_self.transfer_dock = MagicMock()
        mock_self.actor_worker = MagicMock()
        mock_self.actor_worker.rl_config = MagicMock()
        mock_self.actor_worker.rl_config.n_samples_per_prompt = 4
        mock_self.actor_worker.rl_config.use_stepwise_advantage = False

        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils.compute_advantage_utils') as mock_utils:
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils.ray.get', return_value=None):
                from aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils import compute_advantage

                compute_advantage(mock_self, blocking=True)

                mock_utils.options.assert_called_once()

    def test_compute_advantage_utils_gae(self):
        mock_td = MagicMock()
        mock_td.get_experience_len.remote.return_value = 8
        mock_td.all_consumed.remote.return_value = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad = 0

        _ray_get_call_count = [0]
        def _mock_ray_get(val):
            _ray_get_call_count[0] += 1
            if _ray_get_call_count[0] == 1:
                return 8
            elif _ray_get_call_count[0] == 2:
                return True
            else:
                return ({}, 0)

        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils.ray.get', side_effect=_mock_ray_get):
            with patch('mindspeed_rl.utils.utils.get_current_dp_range_indexes', return_value=[]):
                with patch('mindspeed_rl.utils.pad_process.remove_padding_tensor_dict_to_dict', return_value={}):
                    with patch('mindspeed_rl.trainer.utils.transfer_dock.pad_experience', return_value={}):
                        with patch('mindspeed_rl.utils.utils.generate_mask', return_value=torch.randn(2, 10)):
                            with patch('mindspeed_rl.trainer.utils.compute_utils.compute_gae_advantage_return',
                                      return_value=(torch.randn(2, 10), torch.randn(2, 10))):
                                with patch('mindspeed_rl.utils.pad_process.truncate_rows', return_value=torch.randn(2, 10)):
                                    with patch('mindspeed_rl.utils.pad_process.padding_dict_to_tensor_dict', return_value={}):
                                        from aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils import compute_advantage_utils

                                        compute_advantage_utils(
                                            td=mock_td,
                                            gamma=0.99,
                                            lam=0.95,
                                            adv_estimator="gae",
                                            experience_count=8,
                                            tokenizer=mock_tokenizer,
                                            global_batch_size=32,
                                            guarantee_order=False,
                                            n_sample_per_prompt=4,
                                            use_stepwise_advantage=False,
                                        )

    def test_compute_advantage_utils_group_norm(self):
        mock_td = MagicMock()
        mock_td.get_experience_len.remote.return_value = 8
        mock_td.all_consumed.remote.return_value = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad = 0

        _ray_get_call_count = [0]
        def _mock_ray_get(val):
            _ray_get_call_count[0] += 1
            if _ray_get_call_count[0] == 1:
                return 8
            elif _ray_get_call_count[0] == 2:
                return True
            else:
                return ({}, 0)

        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils.ray.get', side_effect=_mock_ray_get):
            with patch('mindspeed_rl.utils.utils.get_current_dp_range_indexes', return_value=[]):
                with patch('mindspeed_rl.utils.pad_process.remove_padding_tensor_dict_to_dict', return_value={}):
                    with patch('mindspeed_rl.trainer.utils.transfer_dock.pad_experience', return_value={}):
                        with patch('mindspeed_rl.utils.utils.generate_mask', return_value=torch.randn(2, 10)):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils_patch.compute_group_norm_advantage_return_patch',
                                      return_value=(torch.randn(2, 10), torch.randn(2, 10))):
                                with patch('mindspeed_rl.utils.pad_process.truncate_rows', return_value=torch.randn(2, 10)):
                                    with patch('mindspeed_rl.utils.pad_process.padding_dict_to_tensor_dict', return_value={}):
                                        from aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils import compute_advantage_utils

                                        compute_advantage_utils(
                                            td=mock_td,
                                            gamma=0.99,
                                            lam=0.95,
                                            adv_estimator="group_norm",
                                            experience_count=8,
                                            tokenizer=mock_tokenizer,
                                            global_batch_size=32,
                                            guarantee_order=False,
                                            n_sample_per_prompt=4,
                                            use_stepwise_advantage=False,
                                        )



    def test_compute_advantage_utils_gae_with_kl(self):
        mock_td = MagicMock()
        mock_td.get_experience_len.remote.return_value = 8
        mock_td.all_consumed.remote.return_value = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad = 0

        _ray_get_call_count = [0]
        def _mock_ray_get(val):
            _ray_get_call_count[0] += 1
            if _ray_get_call_count[0] == 1:
                return 8
            elif _ray_get_call_count[0] == 2:
                return True
            else:
                return ({"values": torch.randn(2, 10), "responses": torch.randn(2, 10),
                        "token_level_rewards": torch.randn(2, 10), "response_length": torch.tensor([10, 10])}, 0)

        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils.ray.get', side_effect=_mock_ray_get):
            with patch('mindspeed_rl.utils.utils.get_current_dp_range_indexes', return_value=[]):
                with patch('mindspeed_rl.utils.pad_process.remove_padding_tensor_dict_to_dict',
                          return_value={"values": torch.randn(2, 10), "responses": torch.randn(2, 10),
                                       "token_level_rewards": torch.randn(2, 10), "response_length": torch.tensor([10, 10])}):
                    with patch('mindspeed_rl.trainer.utils.transfer_dock.pad_experience', return_value={}):
                        with patch('mindspeed_rl.utils.utils.generate_mask', return_value=torch.randn(2, 10)):
                            with patch('mindspeed_rl.trainer.utils.compute_utils.compute_gae_advantage_return',
                                      return_value=(torch.randn(2, 10), torch.randn(2, 10))):
                                with patch('mindspeed_rl.utils.pad_process.truncate_rows', return_value=torch.randn(2, 10)):
                                    with patch('mindspeed_rl.utils.pad_process.padding_dict_to_tensor_dict', return_value={}):
                                        from aura.trainer.train_adapter.mindspeed_rl.patch.compute_utils import compute_advantage_utils

                                        compute_advantage_utils(
                                            td=mock_td,
                                            gamma=0.99,
                                            lam=0.95,
                                            adv_estimator="gae",
                                            experience_count=8,
                                            tokenizer=mock_tokenizer,
                                            global_batch_size=32,
                                            guarantee_order=False,
                                            n_sample_per_prompt=4,
                                            use_stepwise_advantage=False,
                                            use_kl_in_reward=True,
                                        )
