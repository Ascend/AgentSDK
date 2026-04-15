#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import sys
import os
import io
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)

mock_requests = MagicMock()
mock_torch = MagicMock()
mock_fastapi = MagicMock()

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

mock_http_server_module = MagicMock()
mock_start_server = MagicMock()
mock_http_server_module.start_server = mock_start_server

mock_dispatch_actor_module = MagicMock()
mock_dispatch_actor_class = MagicMock()
mock_dispatch_actor_module.DispatchActor = mock_dispatch_actor_class

mock_train_server_module = MagicMock()
mock_train_server_class = MagicMock()
mock_train_server_module.TrainServer = mock_train_server_class

mock_weight_updater_module = MagicMock()
mock_weight_update_actor_class = MagicMock()
mock_weight_updater_module.WeightUpdateActor = mock_weight_update_actor_class

mock_controller_config_module = MagicMock()
mock_controller_config_instance = MagicMock()
mock_controller_config_instance.train_server_addr = "127.0.0.1:4002"
mock_controller_config_instance.rollout_server_addr = "127.0.0.1:4001"
mock_controller_config_module.ControllerConfig.return_value = mock_controller_config_instance

mock_http_status_module = MagicMock()
mock_http_status_module.HTTP_OK_200 = 200

mock_utils_module = MagicMock()
mock_create_actor = MagicMock()
mock_utils_module.create_actor = mock_create_actor
mock_utils_module.DEFAULT_SLEEP_TIME = 0.001  # Fast for tests
mock_utils_module.MAX_CPUS = 4
mock_utils_module.MAX_TIMEOUT = 1800
mock_utils_module.DEFAULT_URL_METHOD = "http"

with patch.dict('sys.modules', {
    'ray': mock_ray,
    'ray.util': MagicMock(),
    'ray.util.scheduling_strategies': MagicMock(),
    'requests': mock_requests,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'fastapi': mock_fastapi,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.base.utils.http_server': mock_http_server_module,
    'agentic_rl.controllers.train_controller.dispatch_actor': mock_dispatch_actor_module,
    'agentic_rl.controllers.train_controller.train_server': mock_train_server_module,
    'agentic_rl.controllers.train_controller.train_weight_updater': mock_weight_updater_module,
    'agentic_rl.controllers.utils.controller_config': mock_controller_config_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status_module,
    'agentic_rl.controllers.utils.utils': mock_utils_module,
}):
    from agentic_rl.controllers.train_controller.train_controller import TrainController
    import agentic_rl.controllers.train_controller.train_controller as _train_controller_mod


class TestTrainController(unittest.TestCase):
    """Tests for TrainController covering init, pre_initialize, rollout, training flow."""

    def _make_controller(self, **overrides):
        defaults = dict(
            global_batch_size=8,
            n_samples_per_prompt=2,
            validate_num_samples=4,
            init_num_group_batches=3,
            max_queue_size=10,
            train_iters=5,
            weight_save_dir="/tmp/test_weights",
            delta=2,
            data_loader=MagicMock(),
            actor_worker=MagicMock(),
            initialize_rollout_dataloader=MagicMock(),
            consumed_train_samples=0,
            data_optimized=False,
        )
        defaults.update(overrides)
        return TrainController(**defaults)

    def setUp(self):
        mock_controller_config_module.ControllerConfig.return_value = mock_controller_config_instance
        mock_create_actor.reset_mock()
        mock_ray.reset_mock()
        mock_requests.reset_mock()
        mock_torch.reset_mock()
        mock_logger.reset_mock()
        mock_start_server.reset_mock()
        mock_train_server_class.reset_mock()
        self.ctrl = self._make_controller()

    def tearDown(self):
        pass

    # ---- __init__ -----------------------------------------------------------

    def test_init_sets_global_batch_size(self):
        self.assertEqual(self.ctrl.global_batch_size, 8)

    def test_init_sets_n_samples_per_prompt(self):
        self.assertEqual(self.ctrl.n_samples_per_prompt, 2)

    def test_init_sets_train_iters(self):
        self.assertEqual(self.ctrl.train_iters, 5)

    def test_init_sets_weight_save_dir(self):
        self.assertEqual(self.ctrl.weight_save_dir, "/tmp/test_weights")

    def test_init_sets_delta(self):
        self.assertEqual(self.ctrl.delta, 2)

    def test_init_dispatch_actor_none(self):
        self.assertIsNone(self.ctrl.dispatch_actor)

    def test_init_train_server_none(self):
        self.assertIsNone(self.ctrl.train_server)

    def test_init_weight_update_actor_none(self):
        self.assertIsNone(self.ctrl.weight_update_actor)

    def test_init_reads_controller_config(self):
        self.assertEqual(self.ctrl.train_server_addr, "127.0.0.1:4002")
        self.assertEqual(self.ctrl.rollout_server_addr, "127.0.0.1:4001")

    def test_init_timing_training_unit_empty(self):
        self.assertEqual(self.ctrl.timing_training_unit, [])

    # ---- initialize_dispatch ------------------------------------------------

    def test_initialize_dispatch_calls_create_actor(self):
        mock_dispatch_actor_ref = MagicMock()
        mock_create_actor.return_value = mock_dispatch_actor_ref
        mock_scheduling = MagicMock()
        with patch.object(_train_controller_mod, 'ray', mock_ray), \
             patch.dict('sys.modules', {'ray.util.scheduling_strategies': mock_scheduling}):
            self.ctrl.initialize_dispatch()
        mock_create_actor.assert_called_once()
        call_kwargs = mock_create_actor.call_args[1]
        self.assertEqual(call_kwargs['name'], 'dispatch')
        self.assertEqual(call_kwargs['cls'], mock_dispatch_actor_class)
        self.assertEqual(call_kwargs['namespace'], 'controller_raygroup')

    def test_initialize_dispatch_calls_init_done(self):
        mock_dispatch_actor_ref = MagicMock()
        mock_create_actor.return_value = mock_dispatch_actor_ref
        mock_scheduling = MagicMock()
        with patch.object(_train_controller_mod, 'ray', mock_ray), \
             patch.dict('sys.modules', {'ray.util.scheduling_strategies': mock_scheduling}):
            self.ctrl.initialize_dispatch()
        mock_dispatch_actor_ref.init_done.remote.assert_called_once()
        mock_ray.get.assert_called()

    # ---- initialize_train_server --------------------------------------------

    def test_initialize_train_server_creates_train_server(self):
        with patch.object(_train_controller_mod, 'threading') as mock_threading:
            self.ctrl.initialize_train_server()
            mock_train_server_class.assert_called_once_with(
                max_queue_size=10,
                global_batch_size=8,
                n_samples_per_prompt=2,
            )

    def test_initialize_train_server_starts_thread(self):
        with patch.object(_train_controller_mod, 'threading') as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            self.ctrl.initialize_train_server()
            mock_thread.start.assert_called_once()

    # ---- initialize_weight_updater ------------------------------------------

    def test_initialize_weight_updater_calls_create_actor(self):
        mock_weight_ref = MagicMock()
        mock_create_actor.return_value = mock_weight_ref
        self.ctrl.dispatch_actor = MagicMock()
        self.ctrl.initialize_weight_updater()
        mock_create_actor.assert_called_once()
        call_kwargs = mock_create_actor.call_args[1]
        self.assertEqual(call_kwargs['name'], 'weight_updater')
        self.assertEqual(call_kwargs['cls'], mock_weight_update_actor_class)

    # ---- send_initial_batch_groups_to_rollout --------------------------------

    def test_send_initial_batch_groups(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        self.ctrl.send_initial_batch_groups_to_rollout()
        mock_da.send_batch_groups.remote.assert_called_once_with(3)

    # ---- unlock_rollout_unit ------------------------------------------------

    def test_unlock_rollout_unit(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        self.ctrl.unlock_rollout_unit()
        mock_da.unlock_rollout_unit.remote.assert_called_once()

    # ---- clean_train_updated_weights ----------------------------------------

    def test_clean_train_updated_weights_nonexistent_dir(self):
        with patch('os.path.exists', return_value=False):
            # Should not raise, no deletion attempted
            self.ctrl.clean_train_updated_weights()

    def test_clean_train_updated_weights_removes_files(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=[("/tmp/test_weights", [], ["file1.pt", "file2.pt"])]), \
             patch('os.remove') as mock_remove:
            self.ctrl.clean_train_updated_weights()
            self.assertEqual(mock_remove.call_count, 2)
            mock_remove.assert_any_call("/tmp/test_weights/file1.pt")
            mock_remove.assert_any_call("/tmp/test_weights/file2.pt")

    # ---- _training_batch_queue_ready ----------------------------------------

    def test_training_batch_queue_ready_true(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_ready": True}
        mock_requests.post.return_value = mock_response
        result = self.ctrl._training_batch_queue_ready()
        self.assertTrue(result)

    def test_training_batch_queue_ready_false_status(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests.post.return_value = mock_response
        result = self.ctrl._training_batch_queue_ready()
        self.assertFalse(result)

    def test_training_batch_queue_ready_false_not_ready(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_ready": False}
        mock_requests.post.return_value = mock_response
        result = self.ctrl._training_batch_queue_ready()
        self.assertFalse(result)

    # ---- finish_training_iteration ------------------------------------------

    def test_finish_training_iteration_appends_time(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        self.ctrl.finish_training_iteration(3)
        self.assertEqual(len(self.ctrl.timing_training_unit), 1)
        mock_da.set_current_training_iter.remote.assert_called_once_with(4)

    def test_finish_training_iteration_increments_iter_by_one(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        self.ctrl.finish_training_iteration(0)
        mock_da.set_current_training_iter.remote.assert_called_with(1)

    # ---- finish_training ----------------------------------------------------

    def test_finish_training_shuts_down_actors(self):
        mock_da = MagicMock()
        mock_wa = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        self.ctrl.weight_update_actor = mock_wa
        self.ctrl.finish_training()
        mock_da.shutdown.remote.assert_called_once()
        mock_ray.get.assert_called()
        mock_ray.kill.assert_any_call(mock_da)
        mock_ray.kill.assert_any_call(mock_wa)

    # ---- _get_old_iter_dirs -------------------------------------------------

    def test_get_old_iter_dirs(self):
        with patch.object(Path, 'glob') as mock_glob:
            mock_dir1 = MagicMock()
            mock_dir1.is_dir.return_value = True
            mock_dir1.stat.return_value.st_mtime = 100
            mock_dir2 = MagicMock()
            mock_dir2.is_dir.return_value = True
            mock_dir2.stat.return_value.st_mtime = 200
            mock_glob.return_value = [mock_dir2, mock_dir1]
            ckpt_dir, all_iters = self.ctrl._get_old_iter_dirs()
            self.assertEqual(len(all_iters), 2)
            # sorted by mtime ascending
            self.assertIs(all_iters[0], mock_dir1)
            self.assertIs(all_iters[1], mock_dir2)

    # ---- _clean_old_iters_than_delta ----------------------------------------

    def test_clean_old_iters_removes_excess(self):
        ckpt_dir = Path("/tmp/test_weights")
        mock_paths = [MagicMock() for _ in range(4)]
        with patch('shutil.rmtree') as mock_rmtree:
            self.ctrl._clean_old_iters_than_delta(ckpt_dir, mock_paths)
            # delta=2, 4 iters, should remove first 2
            self.assertEqual(mock_rmtree.call_count, 2)

    def test_clean_old_iters_no_removal_when_under_delta(self):
        ckpt_dir = Path("/tmp/test_weights")
        mock_paths = [MagicMock()]
        with patch('shutil.rmtree') as mock_rmtree:
            self.ctrl._clean_old_iters_than_delta(ckpt_dir, mock_paths)
            mock_rmtree.assert_not_called()

    # ---- _create_weight_dir -------------------------------------------------

    def test_create_weight_dir_returns_path(self):
        with patch.object(self.ctrl, '_get_old_iter_dirs', return_value=(Path("/tmp/test_weights"), [])), \
             patch.object(self.ctrl, '_clean_old_iters_than_delta'):
            result = self.ctrl._create_weight_dir(5)
            self.assertEqual(result, Path("/tmp/test_weights") / "iter_0000005")

    def test_create_weight_dir_zero_padded(self):
        with patch.object(self.ctrl, '_get_old_iter_dirs', return_value=(Path("/tmp/w"), [])), \
             patch.object(self.ctrl, '_clean_old_iters_than_delta'):
            result = self.ctrl._create_weight_dir(42)
            self.assertEqual(result.name, "iter_0000042")

    # ---- pre_initialize -----------------------------------------------------

    def test_pre_initialize_calls_all_three(self):
        self.ctrl.initialize_dispatch = MagicMock()
        self.ctrl.initialize_train_server = MagicMock()
        self.ctrl.initialize_weight_updater = MagicMock()
        self.ctrl.pre_initialize()
        self.ctrl.initialize_dispatch.assert_called_once()
        self.ctrl.initialize_train_server.assert_called_once()
        self.ctrl.initialize_weight_updater.assert_called_once()

    # ---- initialize_rollout -------------------------------------------------

    def test_initialize_rollout_calls_sequence(self):
        self.ctrl.send_initial_batch_groups_to_rollout = MagicMock()
        self.ctrl.unlock_rollout_unit = MagicMock()
        self.ctrl.clean_train_updated_weights = MagicMock()
        self.ctrl.initialize_rollout()
        self.ctrl.send_initial_batch_groups_to_rollout.assert_called_once()
        self.ctrl.unlock_rollout_unit.assert_called_once()
        self.ctrl.clean_train_updated_weights.assert_called_once()

    # ---- wait_for_rollout_unit_ready ----------------------------------------

    def test_wait_for_rollout_unit_ready_immediate(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        mock_ray.get.return_value = True
        self.ctrl.wait_for_rollout_unit_ready()
        mock_da.check_rollout_unit_ready.remote.assert_called()

    def test_wait_for_rollout_unit_ready_timeout(self):
        mock_da = MagicMock()
        self.ctrl.dispatch_actor = mock_da
        mock_ray.get.return_value = False
        self.ctrl.initialization_timeout = 0.001
        with self.assertRaises(TimeoutError):
            self.ctrl.wait_for_rollout_unit_ready()


if __name__ == '__main__':
    unittest.main()
