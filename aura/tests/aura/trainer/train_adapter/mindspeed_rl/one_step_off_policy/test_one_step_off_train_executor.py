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
import torch
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True, scope="session")
def ensure_aura_src_on_sys_path():
    project_root = Path(__file__).resolve().parents[7]
    aura_src = project_root / "aura"
    aura_src_str = str(aura_src)
    if aura_src_str not in sys.path:
        sys.path.insert(0, aura_src_str)

    yield


@pytest.fixture(autouse=True)
def clean_polluted_modules_before_test():
    """Remove cross-test polluted module entries that can break imports/mocks."""
    polluted_prefixes = (
        "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy",
        "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy",
    )
    polluted_exact = {
        "mindspeed_rl",
        "mindspeed_rl.trainer",
        "mindspeed_rl.trainer.utils",
        "mindspeed_rl.utils",
        "mindspeed_rl.utils.utils",
    }

    for name in list(sys.modules.keys()):
        if name in polluted_exact or name.startswith(polluted_prefixes):
            sys.modules.pop(name, None)

    yield


@pytest.fixture(autouse=True)
def clean_polluted_modules_before_test():
    """Clear polluted trainer modules so imports resolve to real test target modules."""
    polluted_prefixes = (
        "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy",
        "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy",
    )
    for name in list(sys.modules.keys()):
        if name.startswith(polluted_prefixes):
            sys.modules.pop(name, None)

    yield


@pytest.fixture(autouse=True)
def clean_polluted_modules_before_test():
    """Remove cross-test polluted trainer modules but keep global dependency mocks."""
    polluted_prefixes = (
        "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy",
        "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy",
        "aura.trainer.train_adapter.mindspeed_rl.utils.trainer_utils",
        "aura.trainer.train_adapter.mindspeed_rl.patch",
    )

    for name in list(sys.modules.keys()):
        if name.startswith(polluted_prefixes):
            sys.modules.pop(name, None)

    yield


@pytest.fixture(autouse=True)
def isolate_mindspeed_rl_modules(monkeypatch):
    """Prevent importing real mindspeed_rl dependency tree during tests."""
    mock_mindspeed_rl = types.ModuleType("mindspeed_rl")
    mock_mindspeed_rl.Metric = MagicMock

    mock_trainer = types.ModuleType("mindspeed_rl.trainer")
    mock_trainer_utils = types.ModuleType("mindspeed_rl.trainer.utils")
    mock_trainer_utils.compute_grpo_data_metrics = MagicMock(return_value={})

    mock_utils_pkg = types.ModuleType("mindspeed_rl.utils")
    mock_utils_utils = types.ModuleType("mindspeed_rl.utils.utils")
    mock_utils_utils.metrics_post_processing = MagicMock(side_effect=lambda x: x)
    mock_utils_utils.compute_tps = MagicMock(return_value=0.0)
    mock_utils_utils.metrics_sort = MagicMock(side_effect=lambda x, _: x)

    mock_patch_module = types.ModuleType("aura.trainer.train_adapter.mindspeed_rl.patch")
    mock_train_controller_module = types.ModuleType("aura.controllers.train_controller.train_controller")
    mock_train_controller_module.TrainController = MagicMock

    monkeypatch.setitem(sys.modules, "mindspeed_rl", mock_mindspeed_rl)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.trainer", mock_trainer)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.trainer.utils", mock_trainer_utils)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.utils", mock_utils_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.utils.utils", mock_utils_utils)
    monkeypatch.setitem(sys.modules, "aura.trainer.train_adapter.mindspeed_rl.patch", mock_patch_module)
    monkeypatch.setitem(sys.modules, "aura.controllers.train_controller.train_controller", mock_train_controller_module)


class TestOneStepOffTrainExecutor:

    def test_init(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )

        assert executor.validate_freq == 10
        assert executor.test_before_train == False
        assert executor.test_only == False
        assert executor.weight_save_dir == '/path/to/save'
        assert executor.update_weights_interval == 5
        assert executor.delta == 100
        assert executor.data_optimized == True

    def test_transfer_dock_init_data_optimized(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.dataset_additional_keys = ['key1', 'key2']

        with patch.object(executor, 'transfer_dock_init', wraps=executor.transfer_dock_init):
            executor.transfer_dock_init()

            assert executor.dataset_additional_keys == ['key1', 'key2']

    def test_transfer_dock_init_not_optimized(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=False,
        )
        executor.dataset_additional_keys = ['key1', 'key2']

        with patch.object(executor, 'transfer_dock_init', wraps=executor.transfer_dock_init):
            executor.transfer_dock_init()

            assert executor.dataset_additional_keys == ['key1', 'key2']

    def test_update_rollout_metrics(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()
        mock_transfer_dock = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.transfer_dock = mock_transfer_dock

        rollout_metric = {
            'rollout_cost': 1.0,
            'resharding_to_infer': 0.5,
            'res_reward_mean': 0.8,
            'res_reward_min': 0.0,
            'res_reward_max': 1.0,
            'toolcall_reward_mean': 0.9,
            'toolcall_reward_min': 0.5,
            'toolcall_reward_max': 1.0,
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.ray') as mock_ray:
            mock_ray.get.return_value = None
            executor.update_rollout_metrics(rollout_metric)
            assert mock_transfer_dock.update_metrics.remote.call_count == 8

    def test_update_weights_to_rollout_unit(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )

        executor.update_weights_to_rollout_unit(last_iteration=False, iteration=5)
        mock_controller.update_rollout_weights.assert_called_once_with(6)

        mock_controller.reset_mock()
        executor.update_weights_to_rollout_unit(last_iteration=False, iteration=3)
        mock_controller.update_rollout_weights.assert_not_called()

        mock_controller.reset_mock()
        executor.update_weights_to_rollout_unit(last_iteration=True, iteration=0)
        mock_controller.update_rollout_weights.assert_not_called()

    def test_put_data_to_td(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()
        mock_transfer_dock = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.transfer_dock = mock_transfer_dock

        output = {
            'input_ids': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            'attention_mask': [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])],
            'prompt_id': [torch.tensor([0]), torch.tensor([1])],
        }
        index = [0, 1]

        executor.put_data_to_td(output, index)
        mock_transfer_dock.put_experience.remote.assert_called_once()

    def test_collect_iteration_timer_metrics(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()
        mock_transfer_dock = MagicMock()

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.transfer_dock = mock_transfer_dock
        executor.global_batch_size = 32
        executor.n_samples_per_prompt = 4
        executor.tokenizer = MagicMock()
        executor.guarantee_order = False

        rollout_metric = {
            'rollout_cost': 1.0,
            'resharding_to_infer': 0.5,
            'res_reward_mean': 0.8,
            'res_reward_min': 0.0,
            'res_reward_max': 1.0,
            'toolcall_reward_mean': 0.9,
            'toolcall_reward_min': 0.5,
            'toolcall_reward_max': 1.0,
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.ray') as mock_ray:
            mock_ray.get.return_value = {'timing': {'rollout': 0.5}}

            with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.compute_grpo_data_metrics', return_value={'metric1': 0.5}) as mock_compute:
                grpo_metrics, metrics_result = executor.collect_iteration_timer_metrics(0, rollout_metric)

                assert grpo_metrics == {'metric1': 0.5}
                assert metrics_result == {'timing': {'rollout': 0.5}}
                mock_compute.assert_called_once()
                mock_controller.finish_training_iteration.assert_called_once_with(iteration=1)

    def test_collect_iteration_metrics(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()
        mock_transfer_dock = MagicMock()
        mock_tensorboard = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.metric = {
            'timing/rollout': 0.5,
            'timing/update': 0.3,
            'actor/loss': 0.1,
            'tokens_generated': 100,
            'tokens/p/s': 100.0,
            'update_tps': 100.0,
            'vllm_throughput': 100.0
        }

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.transfer_dock = mock_transfer_dock
        executor.tensorboard = mock_tensorboard
        executor.kwargs = {}
        executor.global_batch_size = 32
        executor.n_samples_per_prompt = 4
        executor.train_iters = 10

        class MockTimer:
            last = 1.0

        grpo_data_metrics = {'tokens_generated': 100}
        metrics_result = {
            'timing/rollout': 0.5,
            'timing/update': 0.3,
            'actor/loss': 0.1
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.ray') as mock_ray:
            mock_ray.get.return_value = None

            with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.metrics_post_processing', return_value=metrics_result) as mock_post:
                with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.metrics_sort', return_value=metrics_result) as mock_sort:
                    with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.compute_tps', return_value=100.0) as mock_tps:
                        mock_log = MagicMock()
                        with patch.dict('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.__dict__', {'log': mock_log}):
                            executor.collect_iteration_metrics(0, MockTimer(), mock_metrics, grpo_data_metrics, metrics_result)

                            mock_post.assert_called_once()
                            mock_sort.assert_called_once()
                            assert mock_tps.call_count == 3
                            mock_transfer_dock.clear.remote.assert_called_once()
                            assert mock_tensorboard.add_scalar.call_count >= 1

    def test_collect_iteration_metrics_no_tensorboard(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor

        mock_controller = MagicMock()
        mock_transfer_dock = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.metric = {}

        executor = OneStepOffTrainExecutor(
            controller=mock_controller,
            validate_freq=10,
            test_before_train=False,
            test_only=False,
            weight_save_dir='/path/to/save',
            update_weights_interval=5,
            ckpt_delta=100,
            data_optimized=True,
        )
        executor.transfer_dock = mock_transfer_dock
        executor.tensorboard = None
        executor.kwargs = {}
        executor.global_batch_size = 32
        executor.n_samples_per_prompt = 4
        executor.train_iters = 10

        class MockTimer:
            last = 1.0

        grpo_data_metrics = {'tokens_generated': 100}
        metrics_result = {
            'timing/rollout': 0.5,
            'timing/update': 0.3,
            'actor/loss': 0.1
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.ray') as mock_ray:
            mock_ray.get.return_value = None

            with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.metrics_post_processing', return_value=metrics_result):
                with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.metrics_sort', return_value=metrics_result):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.compute_tps', return_value=100.0):
                        mock_log = MagicMock()
                        with patch.dict('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor.__dict__', {'log': mock_log}):
                            executor.collect_iteration_metrics(0, MockTimer(), mock_metrics, grpo_data_metrics, metrics_result)

                            mock_transfer_dock.clear.remote.assert_called_once()
