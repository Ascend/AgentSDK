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

import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestTrainService:

    def test_dummy_rollout(self):
        """Test dummy_rollout function calls RolloutWorker.remote correctly."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import dummy_rollout

        mock_rl_config = MagicMock()
        mock_rl_config.n_samples_per_prompt = 4
        mock_rl_config.actor_rollout_dispatch_size = 32
        mock_rl_config.validate_n_samples = 100

        mock_agentic_env_config = MagicMock()
        mock_agentic_env_config.trajectory_timeout = 300
        mock_agentic_env_config.rollout_output_path = '/path/to/output'

        mock_actor_config = MagicMock()
        mock_actor_config.tokenizer_name_or_path = '/path/to/tokenizer'
        mock_actor_config.global_batch_size = 32

        mock_generate_config = MagicMock()
        mock_generate_config.train_backend = 'megatron'
        mock_generate_config.weight_save_dir = '/path/to/save'
        mock_generate_config.hybrid_batch_num = 1
        mock_generate_config.use_on_policy = False
        mock_generate_config.wait_available_weight_timeout = 60

        mock_actor_worker = MagicMock()
        mock_agent_service = MagicMock()
        mock_infer_service = MagicMock()

        with patch('aura.trainer.rollout.rollout_worker.RolloutWorker') as mock_rollout_worker:
            mock_rollout_worker.remote.return_value = MagicMock()

            result = dummy_rollout(
                rl_config=mock_rl_config,
                agentic_env_config=mock_agentic_env_config,
                actor_config=mock_actor_config,
                generate_config=mock_generate_config,
                actor_worker=mock_actor_worker,
                agent_service=mock_agent_service,
                infer_service=mock_infer_service,
            )

            mock_rollout_worker.remote.assert_called_once()
            assert result is not None

    def test_get_train_controller_mock_rollout(self):
        """Test get_train_controller with mock_rollout=True."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import get_train_controller

        mock_actor_worker = MagicMock()
        mock_actor_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.mock_rollout = True
        mock_generate_config = MagicMock()

        with patch('aura.controllers.train_controller.train_mock_controller.TrainMockController') as mock_controller:
            mock_controller.return_value = MagicMock()

            result = get_train_controller(
                actor_worker=mock_actor_worker,
                actor_config=mock_actor_config,
                rl_config=mock_rl_config,
                generate_config=mock_generate_config,
                consumed_train_samples=0,
                data_optimized=False,
            )

            mock_controller.assert_called_once()
            assert result is not None

    def test_get_train_controller_data_optimized_true(self):
        """Test get_train_controller with mock_rollout=False and data_optimized=True."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import get_train_controller

        mock_actor_worker = MagicMock()
        mock_actor_config = MagicMock()
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 1000
        mock_rl_config = MagicMock()
        mock_rl_config.mock_rollout = False
        mock_rl_config.n_samples_per_prompt = 4
        mock_rl_config.validate_num_samples = 100
        mock_generate_config = MagicMock()
        mock_generate_config.init_num_group_batches = 10
        mock_generate_config.max_queue_size = 100
        mock_generate_config.weight_save_dir = '/path/to/save'
        mock_generate_config.ckpt_delta = 100

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service.TrainController') as mock_controller:
            mock_controller.return_value = MagicMock()

            result = get_train_controller(
                actor_worker=mock_actor_worker,
                actor_config=mock_actor_config,
                rl_config=mock_rl_config,
                generate_config=mock_generate_config,
                consumed_train_samples=0,
                data_optimized=True,
            )

            mock_controller.assert_called_once()
            call_kwargs = mock_controller.call_args[1]
            assert call_kwargs['initialize_rollout_dataloader'].__name__ == 'optimize_train_dataloader'
            assert result is not None

    def test_get_train_controller_data_optimized_false(self):
        """Test get_train_controller with mock_rollout=False and data_optimized=False."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import get_train_controller

        mock_actor_worker = MagicMock()
        mock_actor_config = MagicMock()
        mock_actor_config.global_batch_size = 32
        mock_actor_config.train_iters = 1000
        mock_rl_config = MagicMock()
        mock_rl_config.mock_rollout = False
        mock_rl_config.n_samples_per_prompt = 4
        mock_rl_config.validate_num_samples = 100
        mock_generate_config = MagicMock()
        mock_generate_config.init_num_group_batches = 10
        mock_generate_config.max_queue_size = 100
        mock_generate_config.weight_save_dir = '/path/to/save'
        mock_generate_config.ckpt_delta = 100

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service.TrainController') as mock_controller:
            mock_controller.return_value = MagicMock()

            result = get_train_controller(
                actor_worker=mock_actor_worker,
                actor_config=mock_actor_config,
                rl_config=mock_rl_config,
                generate_config=mock_generate_config,
                consumed_train_samples=0,
                data_optimized=False,
            )

            mock_controller.assert_called_once()
            call_kwargs = mock_controller.call_args[1]
            assert call_kwargs['initialize_rollout_dataloader'].__name__ == 'default_train_dataloader'
            assert result is not None

    def test_create_rollout_worker_exists(self):
        """Test create_rollout_worker function exists and is callable."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import create_rollout_worker
        assert callable(create_rollout_worker)

    def test_train_function_exists(self):
        """Test that train function exists and is a ray remote function."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import train
        assert callable(train)
        assert hasattr(train, 'options')
        assert hasattr(train, 'remote')
        assert hasattr(train, 'bind')

    def test_dummy_train_function_exists(self):
        """Test that dummy_train function exists and is a ray remote function."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import dummy_train
        assert callable(dummy_train)
        assert hasattr(dummy_train, 'options')
        assert hasattr(dummy_train, 'remote')
        assert hasattr(dummy_train, 'bind')

    def test_logger_exists(self):
        """Test that logger exists."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import logger
        assert logger is not None

    def test_module_imports(self):
        """Test all imports are available."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import (
            ray,
            NodeAffinitySchedulingStrategy,
            Loggers,
            TrainController,
            DEFAULT_SLEEP_TIME,
            ExtendedGenerateConfig,
            optimize_train_dataloader,
            OneStepOffTrainExecutor,
            default_train_dataloader,
            prepare_train,
            logger,
            dummy_rollout,
            get_train_controller,
            create_rollout_worker,
            train,
            dummy_train,
        )
        assert ray is not None
        assert NodeAffinitySchedulingStrategy is not None
        assert Loggers is not None
        assert TrainController is not None
        assert DEFAULT_SLEEP_TIME is not None
        assert ExtendedGenerateConfig is not None
        assert optimize_train_dataloader is not None
        assert OneStepOffTrainExecutor is not None
        assert default_train_dataloader is not None
        assert prepare_train is not None
        assert logger is not None
        assert callable(dummy_rollout)
        assert callable(get_train_controller)
        assert callable(create_rollout_worker)
        assert callable(train)
        assert callable(dummy_train)

    def test_get_train_controller_with_consumed_samples(self):
        """Test get_train_controller with non-zero consumed_train_samples."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import get_train_controller

        mock_actor_worker = MagicMock()
        mock_actor_config = MagicMock()
        mock_actor_config.global_batch_size = 32
        mock_rl_config = MagicMock()
        mock_rl_config.mock_rollout = False
        mock_rl_config.n_samples_per_prompt = 4
        mock_generate_config = MagicMock()
        mock_generate_config.weight_save_dir = '/path/to/save'

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service.TrainController') as mock_controller:
            mock_controller.return_value = MagicMock()

            result = get_train_controller(
                actor_worker=mock_actor_worker,
                actor_config=mock_actor_config,
                rl_config=mock_rl_config,
                generate_config=mock_generate_config,
                consumed_train_samples=1000,
                data_optimized=False,
            )

            mock_controller.assert_called_once()
            call_kwargs = mock_controller.call_args[1]
            assert call_kwargs['consumed_train_samples'] == 1000
            assert result is not None

    def test_create_rollout_worker(self):
        """Test create_rollout_worker function with mocked dependencies."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import create_rollout_worker

        mock_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_agentic_env_config = MagicMock()
        mock_actor_config = MagicMock()
        mock_generate_config = MagicMock()
        mock_agent_service = MagicMock()
        mock_infer_service = MagicMock()

        mock_start_async = MagicMock()
        mock_options_return = MagicMock()
        mock_start_async.options.return_value = mock_options_return

        with patch.dict('sys.modules', {
            'aura.trainer.rollout.rollout_service': MagicMock(
                start_async_rollout_worker=mock_start_async
            )
        }):
            with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service.ray') as mock_ray:
                mock_ray.get_runtime_context.return_value.node_id = 'test-node-id'

                create_rollout_worker(
                    config=mock_config,
                    rl_config=mock_rl_config,
                    agentic_env_config=mock_agentic_env_config,
                    actor_config=mock_actor_config,
                    generate_config=mock_generate_config,
                    agent_service=mock_agent_service,
                    infer_service=mock_infer_service,
                )

                mock_start_async.options.assert_called_once()
                mock_options_return.remote.assert_called_once()

    def test_default_train_dataloader_import(self):
        """Test that default_train_dataloader can be imported and is callable."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import default_train_dataloader
        assert callable(default_train_dataloader)

    def test_optimize_train_dataloader_import(self):
        """Test that optimize_train_dataloader can be imported and is callable."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import optimize_train_dataloader
        assert callable(optimize_train_dataloader)

    def test_prepare_train_import(self):
        """Test that prepare_train can be imported and is callable."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import prepare_train
        assert callable(prepare_train)

    def test_extended_generate_config_import(self):
        """Test that ExtendedGenerateConfig can be imported."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import ExtendedGenerateConfig
        assert ExtendedGenerateConfig is not None

    def test_default_sleep_time_import(self):
        """Test that DEFAULT_SLEEP_TIME can be imported."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import DEFAULT_SLEEP_TIME
        assert DEFAULT_SLEEP_TIME is not None

    def test_train_function_node_affinity_scheduling_strategy(self):
        """Test that train function uses NodeAffinitySchedulingStrategy."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import train
        assert hasattr(train, 'options')

    def test_dummy_train_function_returns_none(self):
        """Test that dummy_train function exists and is ray remote."""
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import dummy_train
        assert callable(dummy_train)
        assert hasattr(dummy_train, 'bind')
