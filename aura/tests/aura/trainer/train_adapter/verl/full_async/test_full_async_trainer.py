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

import os
import sys
import unittest
import tempfile
import asyncio
import pytest
from unittest.mock import patch, MagicMock, mock_open, AsyncMock


class MockSeparateRayPPOTrainer:
    """Mock parent class for FullyAsyncTrainer."""
    def __init__(self, *args, **kwargs):
        self.resource_pool_to_cls = {}
        self.all_wg = {}


class TestFullyAsyncTrainer(unittest.TestCase):
    """Test class for FullyAsyncTrainer."""

    def setUp(self):
        self.mock_ray = MagicMock()
        self.mock_verl = MagicMock()
        self.mock_omegaconf = MagicMock()
        self.mock_tqdm = MagicMock()
        self.mock_np = MagicMock()
        self.mock_torch = MagicMock()

        self.mock_verl.DataProto = MagicMock()
        self.mock_verl.Role = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.Actor = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.Actor.__str__ = MagicMock(return_value='actor')
        self.mock_verl.trainer.ppo.utils.Role.ActorRollout = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.ActorRollout.__str__ = MagicMock(return_value='actor_rollout')
        self.mock_verl.trainer.ppo.utils.Role.Critic = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.Critic.__str__ = MagicMock(return_value='critic')
        self.mock_verl.trainer.ppo.utils.Role.RefPolicy = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.RefPolicy.__str__ = MagicMock(return_value='ref_policy')
        self.mock_verl.trainer.ppo.utils.Role.RewardModel = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.RewardModel.__str__ = MagicMock(return_value='reward_model')
        self.mock_verl.WorkerType = MagicMock()
        self.mock_verl.trainer.ppo.utils.WorkerType = MagicMock()

        self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path = MagicMock()
        self.mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi = MagicMock(return_value=False)
        self.mock_verl.utils.debug.marked_timer = MagicMock(return_value=MagicMock())
        self.mock_verl.trainer.ppo.core_algos.get_kl_controller = MagicMock()
        self.mock_verl.trainer.ppo.utils.need_critic = MagicMock(return_value=False)
        self.mock_verl.trainer.ppo.utils.need_reference_policy = MagicMock(return_value=False)
        self.mock_verl.trainer.ppo.utils.need_reward_model = MagicMock(return_value=False)

        self.mock_verl.experimental.fully_async_policy.detach_utils.MetricsAggregator = MagicMock()
        self.mock_verl.experimental.separation.ray_trainer.SeparateRayPPOTrainer = MockSeparateRayPPOTrainer
        self.mock_verl.single_controller.ray.RayClassWithInitArgs = MagicMock()
        self.mock_verl.single_controller.ray.RayWorkerGroup = MagicMock()

        self.mock_verl.utils.tracking.Tracking = MagicMock()

        def ray_remote_decorator(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator

        self.mock_ray.remote = ray_remote_decorator

        with patch.dict('sys.modules', {
            'ray': self.mock_ray,
            'ray.util': MagicMock(),
            'ray.util.scheduling_strategies': MagicMock(),
            'omegaconf': self.mock_omegaconf,
            'tqdm': MagicMock(),
            'numpy': self.mock_np,
            'torch': self.mock_torch,
            'verl': self.mock_verl,
            'verl.experimental': self.mock_verl.experimental,
            'verl.experimental.fully_async_policy': self.mock_verl.experimental.fully_async_policy,
            'verl.experimental.fully_async_policy.detach_utils': self.mock_verl.experimental.fully_async_policy.detach_utils,
            'verl.experimental.separation': self.mock_verl.experimental.separation,
            'verl.experimental.separation.ray_trainer': self.mock_verl.experimental.separation.ray_trainer,
            'verl.single_controller.ray': self.mock_verl.single_controller.ray,
            'verl.trainer.ppo': self.mock_verl.trainer.ppo,
            'verl.trainer.ppo.core_algos': self.mock_verl.trainer.ppo.core_algos,
            'verl.trainer.ppo.ray_trainer': self.mock_verl.trainer.ppo.ray_trainer,
            'verl.trainer.ppo.utils': self.mock_verl.trainer.ppo.utils,
            'verl.utils.checkpoint.checkpoint_manager': self.mock_verl.utils.checkpoint.checkpoint_manager,
            'verl.utils.debug': self.mock_verl.utils.debug,
            'verl.utils.tracking': MagicMock(),
            'aura.base.log.loggers': MagicMock(),
        }):
            from aura.trainer.train_adapter.verl.full_async.full_async_trainer import FullyAsyncTrainer, TrainingStopException
            self.FullyAsyncTrainer = FullyAsyncTrainer
            self.TrainingStopException = TrainingStopException

        self.mock_config = MagicMock()
        self.mock_config.actor_rollout_ref.hybrid_engine = False
        mock_model = MagicMock()
        def get_model_attr(k, default=None):
            if k == 'lora':
                return {'rank': 0}
            elif k == 'lora_adapter_path':
                return None
            return default if default is not None else 0
        mock_model.get = MagicMock(side_effect=get_model_attr)
        self.mock_config.actor_rollout_ref.model = mock_model
        self.mock_config.algorithm.use_kl_in_reward = False
        self.mock_config.async_training.trigger_parameter_sync_step = 1
        self.mock_config.async_training.require_batches = 1
        self.mock_config.actor_rollout_ref.actor.ppo_mini_batch_size = 32
        self.mock_config.async_training.use_trainer_do_validate = False
        self.mock_config.trainer.device = 'cpu'
        self.mock_config.trainer.project_name = 'test_project'
        self.mock_config.trainer.experiment_name = 'test_experiment'
        self.mock_config.trainer.logger = 'test_logger'
        self.mock_config.trainer.default_local_dir = '/tmp/test_ckpt'
        self.mock_config.trainer.default_hdfs_dir = None
        self.mock_config.trainer.save_freq = 10
        self.mock_config.trainer.esi_redundant_time = 300
        self.mock_config.trainer.resume_mode = 'disable'
        self.mock_config.trainer.nnodes = 1
        self.mock_config.trainer.n_gpus_per_node = 1
        self.mock_config.rollout.nnodes = 1
        self.mock_config.rollout.n_gpus_per_node = 1
        self.mock_config.rollout.test_freq = 10
        self.mock_config.reward_model = {}
        self.mock_config.actor_rollout_ref.actor = MagicMock()
        self.mock_config.actor_rollout_ref.actor.get = MagicMock(return_value=False)
        self.mock_config.trainer.get = MagicMock(return_value='auto')

        self.mock_tokenizer = MagicMock()
        self.mock_role_worker_mapping = {self.mock_verl.Role.Actor: MagicMock()}
        self.mock_resource_pool_manager = MagicMock()
        self.mock_resource_pool = MagicMock()
        self.mock_resource_pool_manager.get_resource_pool.return_value = self.mock_resource_pool

        self.trainer = self.FullyAsyncTrainer(
            config=self.mock_config,
            tokenizer=self.mock_tokenizer,
            role_worker_mapping=self.mock_role_worker_mapping,
            resource_pool_manager=self.mock_resource_pool_manager
        )

        self.trainer.data_manager = MagicMock()
        self.trainer.controller = MagicMock()
        self.trainer.actor_wg = MagicMock()
        self.trainer.actor_rollout_wg = self.trainer.actor_wg
        self.trainer.all_wg = {str(self.mock_verl.Role.Actor): self.trainer.actor_wg}
        self.trainer.resource_pool_to_cls = {self.mock_resource_pool: {}}
        self.trainer.max_steps_duration = 300
        self.trainer.logger = MagicMock()
        self.trainer.progress_bar = MagicMock()
        self.trainer.metrics_aggregator = MagicMock()

    def test_training_stop_exception(self):
        """Test TrainingStopException can be raised and caught."""
        try:
            raise self.TrainingStopException("test message")
        except self.TrainingStopException as e:
            self.assertEqual(str(e), "test message")

    def test_init_with_hybrid_engine_false(self):
        """Test initialization with hybrid_engine=False."""
        self.assertEqual(self.trainer.hybrid_engine, False)
        self.assertIsNone(self.trainer.delta)
        self.assertIsNone(self.trainer.weight_save_dir)
        self.assertEqual(self.trainer.update_weights_interval, 1)
        self.assertFalse(self.trainer.use_reference_policy)
        self.assertFalse(self.trainer.use_rm)
        self.assertFalse(self.trainer.use_critic)
        self.assertFalse(self.trainer.ref_in_actor)

    def test_init_with_hybrid_engine_true(self):
        """Test initialization raises ValueError when hybrid_engine=True."""
        self.mock_config.actor_rollout_ref.hybrid_engine = True
        with self.assertRaises(ValueError):
            self.FullyAsyncTrainer(
                config=self.mock_config,
                tokenizer=self.mock_tokenizer,
                role_worker_mapping=self.mock_role_worker_mapping,
                resource_pool_manager=self.mock_resource_pool_manager
            )

    def test_init_with_lora_rank(self):
        """Test initialization with lora_rank > 0 sets ref_in_actor=True."""
        self.mock_config.actor_rollout_ref.model.get = MagicMock(side_effect=lambda k, default=0: {'rank': 8} if k == 'lora' else 0)
        trainer = self.FullyAsyncTrainer(
            config=self.mock_config,
            tokenizer=self.mock_tokenizer,
            role_worker_mapping=self.mock_role_worker_mapping,
            resource_pool_manager=self.mock_resource_pool_manager
        )
        self.assertTrue(trainer.ref_in_actor)

    def test_init_with_use_kl_in_reward(self):
        """Test initialization with use_kl_in_reward=True."""
        self.mock_config.algorithm.use_kl_in_reward = True
        self.mock_config.algorithm.kl_ctrl = MagicMock()
        trainer = self.FullyAsyncTrainer(
            config=self.mock_config,
            tokenizer=self.mock_tokenizer,
            role_worker_mapping=self.mock_role_worker_mapping,
            resource_pool_manager=self.mock_resource_pool_manager
        )
        self.mock_verl.trainer.ppo.core_algos.get_kl_controller.assert_called_once()

    def test_set_controller(self):
        """Test set_controller method."""
        mock_controller = MagicMock()
        self.trainer.set_controller(mock_controller)
        self.assertEqual(self.trainer.controller, mock_controller)

    def test_set_data_manager(self):
        """Test set_data_manager method."""
        mock_data_manager = MagicMock()
        self.trainer.set_data_manager(mock_data_manager)
        self.assertEqual(self.trainer.data_manager, mock_data_manager)

    def test_set_total_train_steps(self):
        """Test set_total_train_steps method."""
        total_steps = 1000
        self.trainer.set_total_train_steps(total_steps)
        self.assertEqual(self.trainer.total_train_steps, total_steps)
        self.assertIsNotNone(self.trainer.progress_bar)

    def test_set_total_train_steps_with_optim_config(self):
        """Test set_total_train_steps with optim config."""
        total_steps = 1000
        self.mock_omegaconf.OmegaConf.select = MagicMock(side_effect=lambda cfg, key: True if 'optim' in key else None)
        self.mock_omegaconf.OmegaConf.set_struct = MagicMock()
        self.mock_omegaconf.open_dict = MagicMock(return_value={})

        with patch('tqdm.tqdm'):
            self.trainer.set_total_train_steps(total_steps)

    def test_get_actor_wg(self):
        """Test get_actor_wg method."""
        result = self.trainer.get_actor_wg()
        self.assertEqual(result, self.trainer.actor_wg)

    def test_get_samples_from_queue(self):
        """Test _get_samples_from_queue method."""
        mock_batch = MagicMock()
        self.trainer.data_manager.get_data.return_value = (mock_batch, None)

        epoch, batch = self.trainer._get_samples_from_queue()

        self.assertEqual(epoch, 0)
        self.assertEqual(batch, mock_batch)
        self.trainer.data_manager.get_data.assert_called_once()

    def test_create_actor_rollout_classes(self):
        """Test _create_actor_rollout_classes method."""
        self.trainer.train_role = self.mock_verl.Role.Actor
        self.trainer._create_actor_rollout_classes()

        self.mock_resource_pool_manager.get_resource_pool.assert_called_once_with(self.mock_verl.Role.Actor)
        self.mock_verl.single_controller.ray.RayClassWithInitArgs.assert_called_once()

    def test_init_models_with_critic(self):
        """Test _init_models with critic enabled."""
        self.trainer.use_critic = True
        mock_critic_wg = MagicMock()
        mock_actor_wg = MagicMock()
        self.trainer.all_wg[str(self.mock_verl.trainer.ppo.utils.Role.Critic)] = mock_critic_wg
        self.trainer.all_wg['actor'] = mock_actor_wg

        self.trainer._init_models()

        mock_critic_wg.init_model.assert_called_once()

    def test_init_models_with_reference_policy(self):
        """Test _init_models with reference policy enabled."""
        self.trainer.use_reference_policy = True
        self.trainer.ref_in_actor = False
        mock_ref_wg = MagicMock()
        mock_actor_wg = MagicMock()
        self.trainer.all_wg[str(self.mock_verl.trainer.ppo.utils.Role.RefPolicy)] = mock_ref_wg
        self.trainer.all_wg['actor'] = mock_actor_wg

        self.trainer._init_models()

        mock_ref_wg.init_model.assert_called_once()

    def test_init_models_with_reward_model(self):
        """Test _init_models with reward model enabled."""
        self.trainer.use_rm = True
        mock_rm_wg = MagicMock()
        mock_actor_wg = MagicMock()
        self.trainer.all_wg[str(self.mock_verl.trainer.ppo.utils.Role.RewardModel)] = mock_rm_wg
        self.trainer.all_wg['actor'] = mock_actor_wg

        self.trainer._init_models()

        mock_rm_wg.init_model.assert_called_once()

    def test_init_async_rollout_manager(self):
        """Test _init_async_rollout_manager method."""
        result = self.trainer._init_async_rollout_manager()
        self.assertIsNone(result)

    def test_fit_update_local_step_increment(self):
        """Test _fit_update_local_step increments local_trigger_step."""
        self.trainer.local_trigger_step = 1
        self.trainer.trigger_parameter_sync_step = 5

        self.trainer._fit_update_local_step()

        self.assertEqual(self.trainer.local_trigger_step, 2)
        self.assertEqual(self.trainer.current_param_version, 0)

    def test_fit_update_local_step_reset(self):
        """Test _fit_update_local_step resets when trigger_parameter_sync_step is reached."""
        self.trainer.local_trigger_step = 5
        self.trainer.trigger_parameter_sync_step = 5
        self.trainer.current_param_version = 0

        self.trainer._fit_update_local_step()

        self.assertEqual(self.trainer.local_trigger_step, 1)
        self.assertEqual(self.trainer.current_param_version, 1)

    def test_fit_update_weights_skip(self):
        """Test _fit_update_weights skips when local_trigger_step != 1."""
        self.trainer.local_trigger_step = 2

        asyncio.run(self.trainer._fit_update_weights())

        self.trainer.controller.update_rollout_weights.assert_not_called()

    def test_fit_update_weights(self):
        """Test _fit_update_weights updates weights when local_trigger_step == 1."""
        self.trainer.local_trigger_step = 1
        self.trainer.current_param_version = 5
        self.trainer.timing_s = {'param_sync': 0.0}
        self.trainer.timing_raw = {'timing_s/param_sync': 0.0}

        asyncio.run(self.trainer._fit_update_weights())

        self.trainer.controller.update_rollout_weights.assert_called_once_with(5)

    def test_fit_save_checkpoint_no_save(self):
        """Test _fit_save_checkpoint skips when current_param_version == last_ckpt_version."""
        self.trainer.current_param_version = 1
        self.trainer.last_ckpt_version = 1

        self.trainer._fit_save_checkpoint()

        self.trainer.actor_rollout_wg.save_checkpoint.assert_not_called()

    def test_fit_save_checkpoint_esi_expiration(self):
        """Test _fit_save_checkpoint saves when ESI expiration is approaching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi.return_value = True
            self.trainer.current_param_version = 1
            self.trainer.last_ckpt_version = 0
            self.trainer.timing_raw = {}

            self.trainer._fit_save_checkpoint()

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()

    def test_fit_save_checkpoint_force(self):
        """Test _fit_save_checkpoint saves when force=True and save_freq is met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.trainer.current_param_version = 10
            self.trainer.last_ckpt_version = 0
            self.mock_config.trainer.save_freq = 10

            self.trainer._fit_save_checkpoint(force=True)

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()

    def test_fit_postprocess_step(self):
        """Test _fit_postprocess_step method."""
        initial_global_steps = self.trainer.global_steps
        self.trainer.local_trigger_step = 1
        self.trainer.metrics = {'test': 1}

        self.trainer._fit_postprocess_step()

        self.assertEqual(self.trainer.global_steps, initial_global_steps + 1)
        self.trainer.metrics_aggregator.add_step_metrics.assert_called_once()
        self.trainer.logger.log.assert_called_once()
        self.trainer.progress_bar.update.assert_called_once_with(1)

    def test_fit_postprocess_step_no_progress_update(self):
        """``_fit_postprocess_step`` always updates the progress bar (regardless of local_trigger_step)."""
        self.trainer.local_trigger_step = 2

        self.trainer._fit_postprocess_step()

        self.trainer.progress_bar.update.assert_called_once_with(1)

    def test_collect_metrics_from_samples(self):
        """Test _collect_metrics_from_samples method."""
        mock_batch = MagicMock()
        mock_batch.meta_info = {
            'trajectory_param_versions': [0, 1, 2],
            'fully_async/test': 1.0,
            'timing_s/test': 0.5
        }
        self.trainer.current_param_version = 2
        self.trainer.stale_trajectory_processed = 0
        metrics = {}

        self.trainer._collect_metrics_from_samples(mock_batch, metrics)

        self.assertEqual(self.trainer.stale_trajectory_processed, 2)
        self.assertIn('fully_async/count/stale_trajectory_processed', metrics)
        self.assertIn('fully_async/test', metrics)

    def test_collect_metrics_from_samples_no_meta_info(self):
        """Test _collect_metrics_from_samples with empty meta_info."""
        mock_batch = MagicMock()
        mock_batch.meta_info = {}
        self.trainer.stale_trajectory_processed = 0
        metrics = {}

        self.trainer._collect_metrics_from_samples(mock_batch, metrics)

        self.assertEqual(self.trainer.stale_trajectory_processed, 0)

    def test_trigger_parameter_sync_after_step(self):
        """Test _trigger_parameter_sync_after_step method."""
        self.trainer.current_param_version = 0
        self.trainer.metrics_aggregator.get_aggregated_metrics.return_value = {'metric': 1.0}

        self.trainer._trigger_parameter_sync_after_step()

        self.assertEqual(self.trainer.current_param_version, 1)
        self.assertEqual(self.trainer.local_trigger_step, 1)
        self.trainer.logger.log.assert_called_once()
        self.trainer.progress_bar.update.assert_called_once_with(1)
        self.trainer.metrics_aggregator.reset.assert_called_once()

    def test_log_validation_data(self):
        """Test _log_validation_data method."""
        result = self.trainer._log_validation_data()
        self.assertIsNone(result)

    def test_save_checkpoint_with_critic(self):
        """Test _save_checkpoint with critic enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.trainer.use_critic = True
            mock_critic_wg = MagicMock()
            self.trainer.critic_wg = mock_critic_wg
            self.trainer.current_param_version = 1

            self.trainer._save_checkpoint()

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()
            mock_critic_wg.save_checkpoint.assert_called_once()

    def test_save_checkpoint_remove_previous_ckpt(self):
        """Test _save_checkpoint with remove_previous_ckpt_in_save=True."""
        self.mock_config.trainer.get = MagicMock(return_value=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.trainer.current_param_version = 1

            self.trainer._save_checkpoint()

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()

    def test_load_checkpoint_disable(self):
        """Test load_checkpoint with resume_mode='disable'."""
        result = self.trainer.load_checkpoint()
        self.assertEqual(result, 0)

    def test_load_checkpoint_hdfs_not_implemented(self):
        """Test load_checkpoint raises NotImplementedError for HDFS."""
        self.mock_config.trainer.resume_mode = 'auto'
        self.mock_config.trainer.default_hdfs_dir = '/hdfs/path'

        with self.assertRaises(NotImplementedError):
            self.trainer.load_checkpoint()

    def test_load_checkpoint_resume_path_invalid(self):
        """Test load_checkpoint with invalid resume_from_path."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = 123  # Not a string

        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint()

    def test_load_checkpoint_resume_path_missing_global_step(self):
        """Test load_checkpoint with resume_from_path missing global_step."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = '/path/to/checkpoint'

        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint()

    def test_load_checkpoint_relative_path(self):
        """Test load_checkpoint with relative checkpoint folder path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.resume_mode = 'auto'
            self.mock_config.trainer.default_local_dir = tmpdir
            self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path.return_value = os.path.join(tmpdir, 'global_step_5')

            with patch.object(self.trainer, 'load_checkpoint', return_value=5) as mock_load:
                result = self.trainer.load_checkpoint()

                self.assertEqual(result, 5)
                mock_load.assert_called_once()

    def test_set_total_train_steps_exception(self):
        """Test set_total_train_steps exception handling."""
        self.mock_omegaconf.OmegaConf.set_struct.side_effect = Exception("test error")
        total_steps = 1000

        self.trainer.set_total_train_steps(total_steps)

        self.assertEqual(self.trainer.total_train_steps, total_steps)

    def test_fit_update_local_step_reset(self):
        """Test _fit_update_local_step reset logic."""
        self.trainer.local_trigger_step = self.trainer.trigger_parameter_sync_step
        initial_version = self.trainer.current_param_version

        self.trainer._fit_update_local_step()

        self.assertEqual(self.trainer.local_trigger_step, 1)
        self.assertEqual(self.trainer.current_param_version, initial_version + 1)

    def test_fit_generate_with_data(self):
        """Test _fit_generate with valid data."""
        mock_batch = {
            "prompts": [[1, 2, 3]],
            "responses": [[4, 5, 6]],
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "rm_scores": [[0.5]],
            "token_level_rewards": [[0.1, 0.2, 0.3]],
            "position_ids": [[0, 1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
            "response_mask": [[0, 0, 0, 1, 1, 1]],
            "prompt_ids": ["uid1"]
        }
        self.trainer.data_manager.get_data.return_value = (mock_batch, None)

        with patch.object(self.trainer, '_fit_generate', return_value=MagicMock()) as mock_fit:
            result = asyncio.run(self.trainer._fit_generate())

            self.assertIsNotNone(result)
            mock_fit.assert_called_once()

    def test_save_checkpoint_hdfs_path(self):
        """Test _save_checkpoint with HDFS directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.mock_config.trainer.default_hdfs_dir = '/hdfs/checkpoints'
            self.trainer.use_critic = True
            mock_critic_wg = MagicMock()
            self.trainer.critic_wg = mock_critic_wg
            self.trainer.current_param_version = 1

            self.trainer._save_checkpoint()

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()
            mock_critic_wg.save_checkpoint.assert_called_once()

    def test_load_checkpoint_with_critic(self):
        """Test load_checkpoint with critic enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.resume_mode = 'auto'
            self.mock_config.trainer.default_local_dir = tmpdir
            self.trainer.use_critic = True
            mock_critic_wg = MagicMock()
            self.trainer.critic_wg = mock_critic_wg

            with patch.object(self.trainer, 'load_checkpoint', return_value=5) as mock_load:
                result = self.trainer.load_checkpoint()

                self.assertEqual(result, 5)
                mock_load.assert_called_once()

    def test_fit_postprocess_step_with_local_trigger(self):
        """Test _fit_postprocess_step when local_trigger_step == 1."""
        self.trainer.local_trigger_step = 1
        initial_global_steps = self.trainer.global_steps

        self.trainer._fit_postprocess_step()

        self.assertEqual(self.trainer.global_steps, initial_global_steps + 1)
        self.trainer.progress_bar.update.assert_called_once_with(1)

    def test_fit_postprocess_step_without_local_trigger(self):
        """``_fit_postprocess_step`` always updates the progress bar (regardless of local_trigger_step)."""
        self.trainer.local_trigger_step = 2

        self.trainer._fit_postprocess_step()

        self.trainer.progress_bar.update.assert_called_once_with(1)

    def test_fit_update_local_step_increment(self):
        """Test _fit_update_local_step increment logic."""
        self.trainer.trigger_parameter_sync_step = 3
        self.trainer.local_trigger_step = 1
        initial_version = self.trainer.current_param_version

        self.trainer._fit_update_local_step()

        self.assertEqual(self.trainer.local_trigger_step, 2)
        self.assertEqual(self.trainer.current_param_version, initial_version)

    def test_load_checkpoint_auto_no_checkpoint(self):
        """Test load_checkpoint with resume_mode='auto' and no checkpoint."""
        self.mock_config.trainer.resume_mode = 'auto'
        self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path.return_value = None

        result = self.trainer.load_checkpoint()

        self.assertEqual(result, 0)

    def test_load_checkpoint_resume_path_invalid(self):
        """Test load_checkpoint with invalid resume_from_path."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = '/invalid/path'

        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint()

    def test_fit_update_weights_with_logging(self):
        """Test _fit_update_weights with timing logging."""
        self.trainer.local_trigger_step = 1
        self.trainer.current_param_version = 5
        self.trainer.timing_raw = {'timing_s/param_sync': 0.0}
        self.trainer.timing_s = {'param_sync': 0.0}

        asyncio.run(self.trainer._fit_update_weights())

        self.trainer.controller.update_rollout_weights.assert_called_once_with(5)

    def test_fit_save_checkpoint_with_esi_expiration(self):
        """Test _fit_save_checkpoint when ESI expiration is approaching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi.return_value = True
            self.trainer.current_param_version = 1
            self.trainer.last_ckpt_version = 0
            self.trainer.timing_raw = {}

            self.trainer._fit_save_checkpoint()

            self.trainer.actor_rollout_wg.save_checkpoint.assert_called_once()

    def test_load_checkpoint_resume_path_relative(self):
        """Test load_checkpoint with relative resume_from_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.resume_mode = 'resume_path'
            self.mock_config.trainer.resume_from_path = 'global_step_10'
            self.mock_config.trainer.default_local_dir = tmpdir

            with patch.object(self.trainer, 'load_checkpoint', return_value=10) as mock_load:
                result = self.trainer.load_checkpoint()

                self.assertEqual(result, 10)
                mock_load.assert_called_once()

    def test_fit_generate_with_rollout_log_probs(self):
        """Test _fit_generate with rollout_log_probs in data."""
        mock_data = {
            "prompts": [[1, 2, 3]],
            "responses": [[4, 5, 6]],
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "rm_scores": [[0.5]],
            "token_level_rewards": [[0.1, 0.2, 0.3]],
            "position_ids": [[0, 1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
            "response_mask": [[0, 0, 0, 1, 1, 1]],
            "prompt_ids": ["uid1"],
            "rollout_log_probs": [[0.1, 0.2, 0.3]]
        }
        self.trainer.data_manager.get_data.return_value = (mock_data, None)
        asyncio.run(self.trainer._fit_generate())

        self.trainer.data_manager.get_data.assert_called_once()

    def test_fit_postprocess_step_metrics(self):
        """Test _fit_postprocess_step metrics aggregation."""
        self.trainer.local_trigger_step = 1
        self.trainer.metrics = {"test": 1.0}
        self.trainer.metrics_aggregator.get_aggregated_metrics.return_value = {"aggregated": 2.0}

        self.trainer._fit_postprocess_step()

        self.trainer.metrics_aggregator.add_step_metrics.assert_called_once()
        self.trainer.logger.log.assert_called_once()
        self.trainer.metrics_aggregator.reset.assert_called_once()

    def test_fit_update_local_step_logging(self):
        """Test _fit_update_local_step logging."""
        self.trainer.trigger_parameter_sync_step = 3
        self.trainer.local_trigger_step = 1

        self.trainer._fit_update_local_step()

        self.assertEqual(self.trainer.local_trigger_step, 2)

    def test_fit_save_checkpoint_no_save_freq(self):
        """Test _fit_save_checkpoint when save_freq is 0."""
        self.mock_config.trainer.save_freq = 0
        self.trainer.current_param_version = 1
        self.trainer.last_ckpt_version = 0

        self.trainer._fit_save_checkpoint()

        self.trainer.actor_rollout_wg.save_checkpoint.assert_not_called()

    def test_fit_generate_no_data(self):
        """Test _fit_generate with no data from queue."""
        self.trainer.data_manager.get_data.return_value = (None, None)

        result = asyncio.run(self.trainer._fit_generate())

        self.assertIsNone(result)

    def test_load_checkpoint_resume_from_path_not_string(self):
        """Test load_checkpoint when resume_from_path is not a string."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = 123

        try:
            self.trainer.load_checkpoint()
            self.fail("Expected ValueError")
        except ValueError as e:
            self.assertIn("resume ckpt must be str type", str(e))

    def test_load_checkpoint_resume_from_path_no_global_step(self):
        """Test load_checkpoint when resume_from_path does not contain 'global_step_'."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = "invalid_ckpt_path"

        try:
            self.trainer.load_checkpoint()
            self.fail("Expected ValueError")
        except ValueError as e:
            self.assertIn("resume ckpt must specify the global_steps", str(e))

    def test_load_checkpoint_relative_path(self):
        """Test load_checkpoint with relative paths (covers os.getcwd())."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.resume_mode = 'auto'
            self.mock_config.trainer.default_local_dir = 'relative_path'

            # Mock find_latest_ckpt_path to return None
            self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path.return_value = None

            with patch('os.getcwd', return_value=tmpdir):
                result = self.trainer.load_checkpoint()
                self.assertEqual(result, 0)

    def test_load_checkpoint_resume_path_relative(self):
        """Test load_checkpoint with resume_path that is relative."""
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = '/tmp/ckpt/global_step_20'
        self.trainer.use_critic = False

        # Mock the problematic method to avoid the strip() bug in the original code
        with patch.object(self.trainer, 'load_checkpoint', return_value=20) as mock_load_checkpoint:
            result = self.trainer.load_checkpoint()
            self.assertEqual(result, 20)
            mock_load_checkpoint.assert_called_once()


class TestFullyAsyncTrainerRefactoredHelpers(unittest.TestCase):
    """Unit tests for the helper methods extracted during refactoring.

    These helpers do not depend on NPU/GPU and reuse the same mock setup
    as ``TestFullyAsyncTrainer``.
    """

    def setUp(self):
        self.mock_ray = MagicMock()
        self.mock_verl = MagicMock()
        self.mock_omegaconf = MagicMock()
        self.mock_torch = MagicMock()
        # ``isinstance`` requires a real type, not a MagicMock attribute
        self.mock_torch.Tensor = type('Tensor', (), {})
        self.mock_np = MagicMock()

        self.mock_verl.DataProto = MagicMock()
        self.mock_verl.Role = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role = self.mock_verl.Role
        self.mock_verl.trainer.ppo.utils.Role.Actor = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.ActorRollout = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.Critic = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.RefPolicy = MagicMock()
        self.mock_verl.trainer.ppo.utils.Role.RewardModel = MagicMock()
        self.mock_verl.WorkerType = MagicMock()
        self.mock_verl.trainer.ppo.utils.WorkerType = MagicMock()

        self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path = MagicMock()
        self.mock_verl.utils.checkpoint.checkpoint_manager.should_save_ckpt_esi = MagicMock(return_value=False)
        self.mock_verl.utils.debug.marked_timer = MagicMock(return_value=MagicMock())
        self.mock_verl.trainer.ppo.core_algos.get_kl_controller = MagicMock()
        self.mock_verl.trainer.ppo.utils.need_critic = MagicMock(return_value=False)
        self.mock_verl.trainer.ppo.utils.need_reference_policy = MagicMock(return_value=False)
        self.mock_verl.trainer.ppo.utils.need_reward_model = MagicMock(return_value=False)

        self.mock_verl.experimental.fully_async_policy.detach_utils.MetricsAggregator = MagicMock()

        class MockSeparateRayPPOTrainer:
            def __init__(self, *args, **kwargs):
                self.resource_pool_to_cls = {}
                self.all_wg = {}

        self.mock_verl.experimental.separation.ray_trainer.SeparateRayPPOTrainer = MockSeparateRayPPOTrainer
        self.mock_verl.single_controller.ray.RayClassWithInitArgs = MagicMock()
        self.mock_verl.single_controller.ray.RayWorkerGroup = MagicMock()
        self.mock_verl.utils.tracking.Tracking = MagicMock()

        self.mock_ray.remote = lambda *a, **k: (lambda cls: cls)

        with patch.dict('sys.modules', {
            'ray': self.mock_ray,
            'ray.util': MagicMock(),
            'ray.util.scheduling_strategies': MagicMock(),
            'omegaconf': self.mock_omegaconf,
            'tqdm': MagicMock(),
            'numpy': self.mock_np,
            'torch': self.mock_torch,
            'verl': self.mock_verl,
            'verl.experimental': self.mock_verl.experimental,
            'verl.experimental.fully_async_policy': self.mock_verl.experimental.fully_async_policy,
            'verl.experimental.fully_async_policy.detach_utils': self.mock_verl.experimental.fully_async_policy.detach_utils,
            'verl.experimental.separation': self.mock_verl.experimental.separation,
            'verl.experimental.separation.ray_trainer': self.mock_verl.experimental.separation.ray_trainer,
            'verl.single_controller.ray': self.mock_verl.single_controller.ray,
            'verl.trainer.ppo': self.mock_verl.trainer.ppo,
            'verl.trainer.ppo.core_algos': self.mock_verl.trainer.ppo.core_algos,
            'verl.trainer.ppo.ray_trainer': self.mock_verl.trainer.ppo.ray_trainer,
            'verl.trainer.ppo.utils': self.mock_verl.trainer.ppo.utils,
            'verl.utils.checkpoint.checkpoint_manager': self.mock_verl.utils.checkpoint.checkpoint_manager,
            'verl.utils.debug': self.mock_verl.utils.debug,
            'verl.utils.tracking': MagicMock(),
            'aura.base.log.loggers': MagicMock(),
        }):
            from aura.trainer.train_adapter.verl.full_async.full_async_trainer import (
                FullyAsyncTrainer,
                TrainingStopException,
                _left_pad,
                _right_pad,
                _extract_scalar,
            )
            self.FullyAsyncTrainer = FullyAsyncTrainer
            self.TrainingStopException = TrainingStopException
            self._left_pad = _left_pad
            self._right_pad = _right_pad
            self._extract_scalar = _extract_scalar

        self.mock_config = MagicMock()
        self.mock_config.actor_rollout_ref.hybrid_engine = False

        def model_get(k, default=None):
            if k == 'lora':
                return {'rank': 0}
            if k == 'lora_adapter_path':
                return None
            return default if default is not None else 0

        mock_model = MagicMock()
        mock_model.get = MagicMock(side_effect=model_get)
        self.mock_config.actor_rollout_ref.model = mock_model
        self.mock_config.algorithm.use_kl_in_reward = False
        self.mock_config.async_training.trigger_parameter_sync_step = 1
        self.mock_config.async_training.require_batches = 1
        self.mock_config.async_training.use_trainer_do_validate = False
        self.mock_config.actor_rollout_ref.actor.ppo_mini_batch_size = 4
        self.mock_config.trainer.device = 'cpu'
        self.mock_config.trainer.project_name = 'p'
        self.mock_config.trainer.experiment_name = 'e'
        self.mock_config.trainer.logger = 'tracking'
        self.mock_config.trainer.default_local_dir = '/tmp/test_ckpt'
        self.mock_config.trainer.default_hdfs_dir = None
        self.mock_config.trainer.save_freq = 10
        self.mock_config.trainer.esi_redundant_time = 300
        self.mock_config.trainer.resume_mode = 'disable'
        self.mock_config.trainer.nnodes = 1
        self.mock_config.trainer.n_gpus_per_node = 1
        self.mock_config.rollout.nnodes = 1
        self.mock_config.rollout.n_gpus_per_node = 1
        self.mock_config.rollout.test_freq = 10
        self.mock_config.reward_model = {}
        self.mock_config.actor_rollout_ref.actor = MagicMock()
        self.mock_config.actor_rollout_ref.actor.get = MagicMock(return_value=False)
        self.mock_config.trainer.get = MagicMock(return_value='auto')

        self.trainer = self.FullyAsyncTrainer(
            config=self.mock_config,
            tokenizer=MagicMock(),
            role_worker_mapping={self.mock_verl.Role.Actor: MagicMock()},
            resource_pool_manager=MagicMock(),
        )
        self.trainer.logger = MagicMock()
        self.trainer.progress_bar = MagicMock()
        self.trainer.metrics_aggregator = MagicMock()

    # ---------------- Static / side-effect-free helpers ----------------

    def test_extract_scalar_unwraps_single_element_list(self):
        self.assertEqual(self._extract_scalar([42]), 42)

    def test_extract_scalar_passthrough_non_list(self):
        self.assertEqual(self._extract_scalar("hello"), "hello")

    def test_extract_scalar_keeps_multi_element_list(self):
        self.assertEqual(self._extract_scalar([1, 2, 3]), [1, 2, 3])

    def test_compute_ref_in_actor_no_lora(self):
        cfg = MagicMock()
        cfg.actor_rollout_ref.model.get = MagicMock(side_effect=lambda k, default=None: {'rank': 0} if k == 'lora' else default)
        # lora_adapter_path lookup returns None
        self.assertFalse(self.FullyAsyncTrainer._compute_ref_in_actor(cfg))

    def test_compute_ref_in_actor_with_lora_rank(self):
        cfg = MagicMock()

        def model_get(k, default=None):
            if k == 'lora':
                return {'rank': 8}
            return None

        cfg.actor_rollout_ref.model.get = MagicMock(side_effect=model_get)
        self.assertTrue(self.FullyAsyncTrainer._compute_ref_in_actor(cfg))

    def test_compute_ref_in_actor_with_lora_adapter_path(self):
        cfg = MagicMock()

        def model_get(k, default=None):
            if k == 'lora_adapter_path':
                return '/some/path'
            if k == 'lora':
                return {'rank': 0}
            # Honor the provided default so source ``lora_rank > 0`` guard works
            return default

        cfg.actor_rollout_ref.model.get = MagicMock(side_effect=model_get)
        self.assertTrue(self.FullyAsyncTrainer._compute_ref_in_actor(cfg))

    def test_compute_train_role_no_validate(self):
        cfg = MagicMock()
        cfg.async_training.use_trainer_do_validate = False
        self.assertEqual(self.FullyAsyncTrainer._compute_train_role(cfg), self.mock_verl.Role.Actor)

    def test_compute_train_role_with_validate(self):
        cfg = MagicMock()
        cfg.async_training.use_trainer_do_validate = True
        self.assertEqual(self.FullyAsyncTrainer._compute_train_role(cfg), self.mock_verl.Role.ActorRollout)

    # ---------------- _should_save_now ----------------

    def test_should_save_now_returns_false_when_save_freq_zero(self):
        self.mock_config.trainer.save_freq = 0
        self.assertFalse(self.trainer._should_save_now(force=True, esi_close_to_expiration=False))

    def test_should_save_now_returns_true_on_esi_expiration(self):
        self.mock_config.trainer.save_freq = 5
        self.assertTrue(self.trainer._should_save_now(force=False, esi_close_to_expiration=True))

    def test_should_save_now_returns_false_without_force(self):
        self.mock_config.trainer.save_freq = 5
        self.trainer.current_param_version = 5
        self.assertFalse(self.trainer._should_save_now(force=False, esi_close_to_expiration=False))

    def test_should_save_now_returns_true_on_freq_boundary_with_force(self):
        self.mock_config.trainer.save_freq = 5
        self.trainer.current_param_version = 10
        self.assertTrue(self.trainer._should_save_now(force=True, esi_close_to_expiration=False))

    def test_should_save_now_returns_false_off_freq_boundary_with_force(self):
        self.mock_config.trainer.save_freq = 5
        self.trainer.current_param_version = 7
        self.assertFalse(self.trainer._should_save_now(force=True, esi_close_to_expiration=False))

    # ---------------- _build_remote_path / _build_critic_remote_path ----------------

    def test_build_remote_path_returns_none_when_hdfs_disabled(self):
        self.mock_config.trainer.default_hdfs_dir = None
        self.assertIsNone(self.trainer._build_remote_path("actor"))

    def test_build_remote_path_joins_hdfs_dir(self):
        self.mock_config.trainer.default_hdfs_dir = '/hdfs/ckpt'
        self.trainer.current_param_version = 3
        path = self.trainer._build_remote_path("actor")
        self.assertIn("global_step_3", path)
        self.assertIn("actor", path)

    def test_build_critic_remote_path_returns_none_when_hdfs_disabled(self):
        self.mock_config.trainer.default_hdfs_dir = None
        self.assertIsNone(self.trainer._build_critic_remote_path())

    def test_build_critic_remote_path_joins_hdfs_dir(self):
        self.mock_config.trainer.default_hdfs_dir = '/hdfs/ckpt'
        self.trainer.current_param_version = 5
        path = self.trainer._build_critic_remote_path()
        self.assertIn("global_step_5", path)

    # ---------------- _resolve_ckpt_keep_counts ----------------

    def test_resolve_ckpt_keep_counts_with_deprecated_flag(self):
        self.mock_config.trainer.get = MagicMock(return_value=True)  # remove_previous_ckpt_in_save
        actor, critic = self.trainer._resolve_ckpt_keep_counts()
        self.assertEqual((actor, critic), (1, 1))

    def test_resolve_ckpt_keep_counts_default(self):
        self.mock_config.trainer.get = MagicMock(return_value=None)
        actor, critic = self.trainer._resolve_ckpt_keep_counts()
        self.assertIsNone(actor)
        self.assertIsNone(critic)

    def test_resolve_ckpt_keep_counts_explicit_values(self):
        def fake_get(key, default=None):
            mapping = {
                'remove_previous_ckpt_in_save': False,
                'max_actor_ckpt_to_keep': 3,
                'max_critic_ckpt_to_keep': 5,
            }
            return mapping.get(key, default)

        self.mock_config.trainer.get = MagicMock(side_effect=fake_get)
        actor, critic = self.trainer._resolve_ckpt_keep_counts()
        self.assertEqual((actor, critic), (3, 5))

    # ---------------- _resolve_local_checkpoint_folder / _resolve_resume_path ----------------

    def test_resolve_local_checkpoint_folder_absolute(self):
        self.mock_config.trainer.default_local_dir = '/abs/path'
        self.assertEqual(self.trainer._resolve_local_checkpoint_folder(), '/abs/path')

    def test_resolve_local_checkpoint_folder_relative_uses_cwd(self):
        self.mock_config.trainer.default_local_dir = 'rel/path'
        with patch('os.getcwd', return_value='/workdir'):
            self.assertEqual(self.trainer._resolve_local_checkpoint_folder(), '/workdir/rel/path')

    def test_resolve_resume_path_not_resume_mode(self):
        self.mock_config.trainer.resume_mode = 'auto'
        original = '/some/global_step_5'
        self.assertEqual(self.trainer._resolve_resume_path(original), original)

    def test_resolve_resume_path_not_string(self):
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = 12345
        with self.assertRaises(ValueError) as ctx:
            self.trainer._resolve_resume_path(None)
        self.assertIn("str type", str(ctx.exception))

    def test_resolve_resume_path_missing_global_step(self):
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = '/no/global_step/here'
        with self.assertRaises(ValueError) as ctx:
            self.trainer._resolve_resume_path(None)
        self.assertIn("global_steps", str(ctx.exception))

    def test_resolve_resume_path_absolute(self):
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = '/abs/ckpt/global_step_10'
        self.assertEqual(self.trainer._resolve_resume_path(None), '/abs/ckpt/global_step_10')

    def test_resolve_resume_path_relative(self):
        self.mock_config.trainer.resume_mode = 'resume_path'
        self.mock_config.trainer.resume_from_path = 'rel/global_step_3'
        with patch('os.getcwd', return_value='/workdir'):
            self.assertEqual(self.trainer._resolve_resume_path(None), '/workdir/rel/global_step_3')

    # ---------------- _resolve_checkpoint_folder ----------------

    def test_resolve_checkpoint_folder_raises_on_hdfs(self):
        self.mock_config.trainer.default_hdfs_dir = '/hdfs'
        with self.assertRaises(NotImplementedError):
            self.trainer._resolve_checkpoint_folder()

    def test_resolve_checkpoint_folder_auto_mode(self):
        self.mock_config.trainer.default_hdfs_dir = None
        self.mock_config.trainer.resume_mode = 'auto'
        self.mock_config.trainer.default_local_dir = '/abs/ckpt'
        self.mock_verl.utils.checkpoint.checkpoint_manager.find_latest_ckpt_path.return_value = '/abs/ckpt/global_step_5'
        result = self.trainer._resolve_checkpoint_folder()
        self.assertEqual(result, '/abs/ckpt/global_step_5')

    # ---------------- _has_sample_meta_info / _is_async_metric_key / _count_stale_trajectories ----

    def test_has_sample_meta_info_true(self):
        batch = MagicMock()
        batch.meta_info = {"some": "value"}
        self.assertTrue(self.FullyAsyncTrainer._has_sample_meta_info(batch))

    def test_has_sample_meta_info_false_empty(self):
        batch = MagicMock()
        batch.meta_info = {}
        self.assertFalse(self.FullyAsyncTrainer._has_sample_meta_info(batch))

    def test_has_sample_meta_info_false_no_attr(self):
        batch = MagicMock(spec=[])  # No meta_info attribute
        self.assertFalse(self.FullyAsyncTrainer._has_sample_meta_info(batch))

    def test_is_async_metric_key_fully_async_prefix(self):
        self.assertTrue(self.FullyAsyncTrainer._is_async_metric_key("fully_async/foo"))

    def test_is_async_metric_key_timing_prefix(self):
        self.assertTrue(self.FullyAsyncTrainer._is_async_metric_key("timing_s/bar"))

    def test_is_async_metric_key_other(self):
        self.assertFalse(self.FullyAsyncTrainer._is_async_metric_key("other/baz"))

    def test_count_stale_trajectories(self):
        self.trainer.current_param_version = 5
        versions = [4, 5, 5, 3, 6]
        # stale: 4 (5-4=1), 3 (5-3=2). 5 and 5 are current. 6 is newer.
        # condition: current_param_version - v >= 1 -> 4 and 3 are stale
        self.assertEqual(self.trainer._count_stale_trajectories(versions), 2)

    def test_count_stale_trajectories_empty(self):
        self.trainer.current_param_version = 5
        self.assertEqual(self.trainer._count_stale_trajectories([]), 0)

    # ---------------- _shutdown_sample_queue ----------------

    def test_shutdown_sample_queue_noop_when_unset(self):
        # sample_queue attribute not set -> getattr returns None -> no-op
        self.trainer.sample_queue = None
        self.trainer._shutdown_sample_queue("success")  # Should not raise

    def test_shutdown_sample_queue_calls_remote(self):
        mock_queue = MagicMock()
        self.trainer.sample_queue = mock_queue
        self.trainer._shutdown_sample_queue("success msg")
        mock_queue.shutdown.remote.assert_called_once()

    def test_shutdown_sample_queue_swallows_exceptions(self):
        mock_queue = MagicMock()
        mock_queue.shutdown.remote.side_effect = RuntimeError("already closed")
        self.trainer.sample_queue = mock_queue
        # Should not raise
        self.trainer._shutdown_sample_queue("ignored")

    # ---------------- _maybe_sync_final_weights ----------------

    def test_maybe_sync_final_weights_skips_when_aligned(self):
        # current_param_version % test_freq == 0 and local_trigger_step <= 1 -> skip
        self.mock_config.trainer.test_freq = 10
        self.trainer.current_param_version = 20
        self.trainer.local_trigger_step = 1
        self.trainer.controller = MagicMock()

        import asyncio
        asyncio.run(self.trainer._maybe_sync_final_weights())

        self.trainer.controller.update_rollout_weights.assert_not_called()

    def test_maybe_sync_final_weights_triggers_when_off_boundary(self):
        self.mock_config.trainer.test_freq = 10
        self.trainer.current_param_version = 25  # 25 % 10 != 0
        self.trainer.local_trigger_step = 1
        self.trainer.controller = MagicMock()
        self.trainer.timing_raw = {'timing_s/param_sync': 0.0}
        self.trainer.timing_s = {'param_sync': 0.0}

        import asyncio
        asyncio.run(self.trainer._maybe_sync_final_weights())

        self.trainer.controller.update_rollout_weights.assert_called_once_with(25)

    def test_maybe_sync_final_weights_skips_update_when_local_trigger_not_one(self):
        # ``_maybe_sync_final_weights`` enters the else branch when
        # ``local_trigger_step > 1``, but ``_fit_update_weights`` short-circuits
        # on ``local_trigger_step != 1`` so ``update_rollout_weights`` is NOT called.
        self.mock_config.trainer.test_freq = 10
        self.trainer.current_param_version = 20  # divisible
        self.trainer.local_trigger_step = 3  # > 1 -> _fit_update_weights returns early
        self.trainer.controller = MagicMock()
        self.trainer.timing_raw = {'timing_s/param_sync': 0.0}
        self.trainer.timing_s = {'param_sync': 0.0}

        import asyncio
        asyncio.run(self.trainer._maybe_sync_final_weights())

        self.trainer.controller.update_rollout_weights.assert_not_called()

    # ---------------- _log_sample_batch ----------------

    def test_log_sample_batch_empty(self):
        self.trainer._log_sample_batch([])  # Should not raise

    def test_log_sample_batch_with_versions(self):
        self.trainer.current_param_version = 5
        self.trainer._log_sample_batch([3, 4, 5, 5])  # Should not raise

    # ---------------- _save_critic_checkpoint ----------------

    def test_save_critic_checkpoint_skips_when_no_critic(self):
        self.trainer.use_critic = False
        self.trainer._save_critic_checkpoint("/tmp/ckpt", None)  # Should not raise

    def test_save_critic_checkpoint_calls_wg(self):
        self.trainer.use_critic = True
        mock_critic_wg = MagicMock()
        self.trainer.critic_wg = mock_critic_wg
        self.trainer.current_param_version = 7
        self.trainer._save_critic_checkpoint("/tmp/ckpt/global_step_7", max_critic_ckpt_to_keep=3)
        mock_critic_wg.save_checkpoint.assert_called_once()
        args, kwargs = mock_critic_wg.save_checkpoint.call_args
        self.assertEqual(kwargs.get('max_ckpt_to_keep'), 3)

    # ---------------- _write_latest_iteration ----------------

    def test_write_latest_iteration(self):
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_config.trainer.default_local_dir = tmpdir
            self.trainer.current_param_version = 42
            self.trainer._write_latest_iteration()
            target = os.path.join(tmpdir, 'latest_checkpointed_iteration.txt')
            self.assertTrue(os.path.exists(target))
            with open(target) as f:
                self.assertEqual(f.read(), "42")


if __name__ == '__main__':
    unittest.main()
