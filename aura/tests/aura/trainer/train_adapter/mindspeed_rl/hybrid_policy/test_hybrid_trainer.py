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

import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

_PATCH_MODULE = 'aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.hybrid_trainer'


class TestAgentGRPOTrainer:

    def _make_fit_trainer(self, train_iters=1, validate_freq=10, test_before_train=False,
                          test_only=False, use_stepwise_advantage=False, save_interval=10,
                          skip_actor_log_prob=False, tensorboard=None, gen_sequences_return=4):
        from aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.hybrid_trainer import AgentGRPOTrainer

        rollout_worker = MagicMock()
        rollout_worker.generate_sequences.remote.return_value = gen_sequences_return

        kwargs = {
            "validate_freq": validate_freq,
            "test_before_train": test_before_train,
            "test_only": test_only,
            "use_stepwise_advantage": use_stepwise_advantage,
            "save_interval": save_interval,
        }
        trainer = AgentGRPOTrainer(rollout_worker, **kwargs)

        trainer.actor_worker = MagicMock()
        trainer.actor_worker.get_iteration.return_value = 0
        trainer.train_iters = train_iters
        trainer.blocking = True
        trainer._validate_agent = MagicMock(return_value={"metric1": 0.5})
        trainer.tensorboard = tensorboard
        trainer.n_samples_per_prompt = 4
        trainer.global_batch_size = 8
        trainer.dataset_additional_keys = []
        trainer.guarantee_order = False
        trainer.skip_actor_log_prob = skip_actor_log_prob
        trainer.kl_ctrl = MagicMock()
        trainer.reward_list = []
        trainer.ref_worker = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.transfer_dock = MagicMock()
        trainer.transfer_dock.put_experience.remote.return_value = MagicMock()
        trainer.transfer_dock.get_metrics.remote.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}
        trainer.transfer_dock.clear.remote.return_value = MagicMock()
        trainer.save_interval = save_interval
        trainer.compute_advantage = MagicMock()
        trainer.save_checkpoint = MagicMock()

        return trainer

    def _setup_fit_patches(self, stack, ray_get_side_effect=None):
        mock_ray = stack.enter_context(patch(f'{_PATCH_MODULE}.ray'))
        if ray_get_side_effect:
            mock_ray.get.side_effect = ray_get_side_effect
        else:
            mock_ray.get.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}

        mock_metric = MagicMock()
        mock_metric.metric = {}
        mock_metric.update = MagicMock()
        stack.enter_context(patch(f'{_PATCH_MODULE}.Metric', return_value=mock_metric))

        mock_put = stack.enter_context(patch('mindspeed_rl.trainer.utils.transfer_dock.put_prompts_experience'))
        mock_put.return_value = ({"data": "test"}, [0, 1, 2])

        mock_compute_metrics = stack.enter_context(patch('mindspeed_rl.trainer.utils.compute_grpo_data_metrics'))
        mock_tps = stack.enter_context(patch('mindspeed_rl.utils.utils.compute_tps'))
        mock_post = stack.enter_context(patch('mindspeed_rl.utils.utils.metrics_post_processing'))
        mock_sort = stack.enter_context(patch('mindspeed_rl.utils.utils.metrics_sort'))

        return mock_ray, mock_metric, mock_put, mock_compute_metrics, mock_tps, mock_post, mock_sort

    def test_init(self):
        rollout_worker = MagicMock()
        kwargs = {
            "validate_freq": 10,
            "test_before_train": False,
            "test_only": False,
        }
        from aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.hybrid_trainer import AgentGRPOTrainer
        trainer = AgentGRPOTrainer(rollout_worker, **kwargs)
        assert trainer is not None
        assert trainer.rollout_worker is not None
        assert trainer.validate_freq == 10
        assert trainer.test_before_train == False
        assert trainer.test_only == False

    def test_transfer_dock_init(self):
        rollout_worker = MagicMock()
        rollout_worker.init_data_manager = MagicMock()
        rollout_worker.init_data_manager.remote = MagicMock(return_value=MagicMock())

        kwargs = {
            "validate_freq": 10,
            "test_before_train": False,
            "test_only": False,
        }
        from aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.hybrid_trainer import AgentGRPOTrainer
        trainer = AgentGRPOTrainer(rollout_worker, **kwargs)
        trainer.dataset_additional_keys = ["key1", "key2"]
        trainer.transfer_dock = MagicMock()

        with patch(f'{_PATCH_MODULE}.ray') as mock_ray, \
             patch.object(trainer.__class__.__bases__[0], 'transfer_dock_init'):
            mock_ray.get.return_value = None
            trainer.transfer_dock_init()

        assert trainer.dataset_additional_keys == ["key1", "key2"]

    def test_fit_with_test_before_train(self):
        trainer = self._make_fit_trainer(train_iters=10, test_before_train=True, test_only=True)

        mock_test_dataloader = MagicMock()

        with patch(f'{_PATCH_MODULE}.ray') as mock_ray:
            mock_ray.get.return_value = None
            trainer.fit(iter([]), None, mock_test_dataloader)

        trainer._validate_agent.assert_called_once()

    def test_fit_full_loop(self):
        trainer = self._make_fit_trainer(train_iters=2, validate_freq=1, save_interval=1, tensorboard=MagicMock())

        mock_batch = {"prompt": [1, 2, 3]}
        mock_data_iters = iter([mock_batch, mock_batch])

        def ray_get_side_effect(*args, **kwargs):
            if args and hasattr(args[0], 'remote'):
                return {"timing/update": 0.1, "timing/rollout": 0.2}
            return {"timing/update": 0.1, "timing/rollout": 0.2}

        with ExitStack() as stack:
            _, _, _, mock_compute_metrics, mock_tps, mock_post, mock_sort = \
                self._setup_fit_patches(stack, ray_get_side_effect=ray_get_side_effect)
            mock_compute_metrics.return_value = {"sample_count": 32}
            mock_tps.return_value = 100.0
            mock_post.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}
            mock_sort.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}

            trainer.fit(mock_data_iters, MagicMock(), None)

            assert trainer.actor_worker.update.call_count == 2
            assert trainer.save_checkpoint.call_count == 2

    def test_fit_with_stepwise_advantage(self):
        trainer = self._make_fit_trainer(use_stepwise_advantage=True, gen_sequences_return=8)
        trainer.transfer_dock.reset_experience_len.remote.return_value = MagicMock()

        mock_batch = {"prompt": [1, 2, 3]}
        mock_data_iters = iter([mock_batch])

        def ray_get_side_effect(*args, **kwargs):
            if args and hasattr(args[0], 'remote'):
                return 8
            return {"timing/update": 0.1, "timing/rollout": 0.2}

        with ExitStack() as stack:
            _, _, _, mock_compute_metrics, mock_tps, mock_post, mock_sort = \
                self._setup_fit_patches(stack, ray_get_side_effect=ray_get_side_effect)
            mock_compute_metrics.return_value = {"sample_count": 32}
            mock_tps.return_value = 100.0
            mock_post.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}
            mock_sort.return_value = {"timing/update": 0.1, "timing/rollout": 0.2}

            trainer.fit(mock_data_iters, None, None)

            trainer.ref_worker.update_ref_dispatch_size.assert_called_once()
            trainer.actor_worker.update_actor_logprob_dispatch_size.assert_called_once()
            trainer.actor_worker.update_actor_update_dispatch_size.assert_called_once()
            trainer.transfer_dock.reset_experience_len.remote.assert_called_once()

    def test_fit_with_validation(self):
        trainer = self._make_fit_trainer(train_iters=2, validate_freq=1)

        mock_batch = {"prompt": [1, 2, 3]}
        mock_data_iters = iter([mock_batch, mock_batch])
        mock_val_dataloader = MagicMock()

        with ExitStack() as stack:
            self._setup_fit_patches(stack)

            trainer.fit(mock_data_iters, mock_val_dataloader, None)

            assert trainer._validate_agent.call_count == 2

    def test_fit_with_skip_actor_log_prob(self):
        trainer = self._make_fit_trainer(skip_actor_log_prob=True)

        mock_batch = {"prompt": [1, 2, 3]}
        mock_data_iters = iter([mock_batch])

        with ExitStack() as stack:
            self._setup_fit_patches(stack)

            trainer.fit(mock_data_iters, None, None)

            trainer.actor_worker.compute_log_prob.assert_not_called()

    def test_fit_with_reward_wait(self):
        trainer = self._make_fit_trainer()

        mock_reward1 = MagicMock()
        mock_reward1.wait_all_ref_objs_run_over = MagicMock()
        mock_reward2 = MagicMock()
        trainer.reward_list = [mock_reward1, mock_reward2]

        mock_batch = {"prompt": [1, 2, 3]}
        mock_data_iters = iter([mock_batch])

        with ExitStack() as stack:
            self._setup_fit_patches(stack)

            trainer.fit(mock_data_iters, None, None)

            mock_reward1.wait_all_ref_objs_run_over.assert_called_once()
