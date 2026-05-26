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
import types
import time
import io
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for train_controller
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build isolated fake modules so train_controller can be imported safely."""

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray.get_actor = MagicMock()
    fake_ray.get = MagicMock()
    fake_ray.remote = lambda fn: fn
    fake_ray.get_runtime_context = MagicMock(return_value=MagicMock(node_id="node1"))
    fake_ray.util = types.ModuleType("ray.util")
    fake_ray.util.scheduling_strategies = types.ModuleType("ray.util.scheduling_strategies")
    fake_ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy = MagicMock

    # ---- fake requests ----
    fake_requests = types.ModuleType("requests")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"is_ready": True}
    fake_response.headers = {}
    fake_response.content = b"mock_content"
    fake_requests.post = MagicMock(return_value=fake_response)
    fake_requests.get = MagicMock(return_value=fake_response)

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.load = MagicMock(return_value="torch_data")

    # ---- fake fastapi ----
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = MagicMock()

    # ---- fake threading ----
    fake_threading = types.ModuleType("threading")
    fake_threading.Thread = MagicMock()

    # ---- fake pathlib (with Path as a real class so that .glob exists) ----
    fake_pathlib = types.ModuleType("pathlib")
    class FakePath:
        def __init__(self, *args, **kwargs):
            self.name = ""
            self._args = args
        def __truediv__(self, other):
            return FakePath()
        def glob(self, pattern):
            return []
        def is_dir(self):
            return True
    fake_pathlib.Path = FakePath

    # ---- fake shutil ----
    fake_shutil = types.ModuleType("shutil")
    fake_shutil.rmtree = MagicMock()

    # ---- fake aura modules ----
    # Loggers
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger)))

    # http_server
    fake_http_server = types.ModuleType("aura.base.utils.http_server")
    fake_http_server.start_server = MagicMock()

    # dispatch_actor
    fake_dispatch_actor_mod = types.ModuleType("aura.controllers.train_controller.dispatch_actor")
    fake_dispatch_actor_cls = MagicMock()
    fake_dispatch_actor_mod.DispatchActor = fake_dispatch_actor_cls

    # train_server
    fake_train_server_mod = types.ModuleType("aura.controllers.train_controller.train_server")
    fake_train_server_cls = MagicMock()
    fake_train_server_mod.TrainServer = fake_train_server_cls

    # train_weight_updater
    fake_weight_updater_mod = types.ModuleType("aura.controllers.train_controller.train_weight_updater")
    fake_weight_updater_cls = MagicMock()
    fake_weight_updater_mod.WeightUpdateActor = fake_weight_updater_cls

    # controller_config
    fake_controller_config_mod = types.ModuleType("aura.controllers.utils.controller_config")
    fake_controller_config_mod.ControllerConfig = MagicMock(
        return_value=MagicMock(
            train_server_addr="127.0.0.1:8000",
            rollout_server_addr="rollout:9000",
        )
    )

    # http_status
    fake_http_status_mod = types.ModuleType("aura.controllers.utils.http_status")
    fake_http_status_mod.HTTP_OK_200 = 200

    # utils (create_actor, kill_actor, DEFAULT_SLEEP_TIME, etc.)
    fake_utils_mod = types.ModuleType("aura.controllers.utils.utils")
    fake_utils_mod.create_actor = MagicMock()
    fake_utils_mod.kill_actor = MagicMock()
    fake_utils_mod.DEFAULT_SLEEP_TIME = 0.01
    fake_utils_mod.MAX_CPUS = 1
    fake_utils_mod.MAX_TIMEOUT = 30
    fake_utils_mod.DEFAULT_URL_METHOD = "http"
    fake_utils_mod.TRAIN_CONTROLLER_NAMESPACE = "test_ns"

    # aura packages to locate the real file
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_base_utils = types.ModuleType("aura.base.utils")
    fake_aura_base_utils.__path__ = []
    fake_aura_controllers = types.ModuleType("aura.controllers")
    fake_aura_controllers.__path__ = []
    fake_aura_controllers_train_controller = types.ModuleType("aura.controllers.train_controller")
    fake_aura_controllers_train_controller.__path__ = [
        os.path.join(base, "controllers/train_controller")
    ]
    fake_aura_controllers_utils = types.ModuleType("aura.controllers.utils")
    fake_aura_controllers_utils.__path__ = []

    fakes = {
        "ray": fake_ray,
        "ray.util": fake_ray.util,
        "ray.util.scheduling_strategies": fake_ray.util.scheduling_strategies,
        "requests": fake_requests,
        "torch": fake_torch,
        "fastapi": fake_fastapi,
        "threading": fake_threading,
        "pathlib": fake_pathlib,
        "shutil": fake_shutil,
        "aura.base.log.loggers": fake_loggers,
        "aura.base.utils.http_server": fake_http_server,
        "aura.controllers.train_controller.dispatch_actor": fake_dispatch_actor_mod,
        "aura.controllers.train_controller.train_server": fake_train_server_mod,
        "aura.controllers.train_controller.train_weight_updater": fake_weight_updater_mod,
        "aura.controllers.utils.controller_config": fake_controller_config_mod,
        "aura.controllers.utils.http_status": fake_http_status_mod,
        "aura.controllers.utils.utils": fake_utils_mod,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.base.utils": fake_aura_base_utils,
        "aura.controllers": fake_aura_controllers,
        "aura.controllers.train_controller": fake_aura_controllers_train_controller,
        "aura.controllers.utils": fake_aura_controllers_utils,
    }

    fake_aura_base_utils.__path__ = []

    target = "aura.controllers.train_controller.train_controller"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.controllers.train_controller.train_controller as mod
        yield {
            "mod": mod,
            "TrainController": mod.TrainController,
            "mock_logger": mock_logger,
            "fake_ray": fake_ray,
            "fake_requests": fake_requests,
            "fake_torch": fake_torch,
            "fake_threading": fake_threading,
            "fake_pathlib": fake_pathlib,
            "fake_shutil": fake_shutil,
            "fake_utils_mod": fake_utils_mod,
            "fake_dispatch_actor_mod": fake_dispatch_actor_mod,
            "fake_train_server_mod": fake_train_server_mod,
            "fake_weight_updater_mod": fake_weight_updater_mod,
            "fake_controller_config_mod": fake_controller_config_mod,
            "fake_http_server": fake_http_server,
            "fake_fastapi": fake_fastapi,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_controller(fake_env, **overrides):
    """Create a TrainController with default valid arguments."""
    default_args = {
        "global_batch_size": 4,
        "n_samples_per_prompt": 2,
        "validate_num_samples": 10,
        "init_num_group_batches": 3,
        "max_queue_size": 20,
        "train_iters": 100,
        "weight_save_dir": "/tmp/weights",
        "delta": 3,
        "data_loader": MagicMock(),
        "actor_worker": MagicMock(),
        "initialize_rollout_dataloader": MagicMock(),
        "consumed_train_samples": MagicMock(),
        "data_optimized": MagicMock(),
    }
    default_args.update(overrides)
    return fake_env["TrainController"](**default_args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestInit:
    def test_init_sets_attributes(self, fake_env):
        """__init__ correctly assigns all constructor arguments and default None attrs."""
        ctrl = make_controller(fake_env)
        assert ctrl.global_batch_size == 4
        assert ctrl.n_samples_per_prompt == 2
        assert ctrl.dispatch_actor is None
        assert ctrl.train_server is None
        assert ctrl.weight_update_actor is None
        assert ctrl.initialization_timeout == fake_env["mod"].MAX_TIMEOUT
        assert ctrl.train_server_addr == "127.0.0.1:8000"
        assert ctrl.rollout_server_addr == "rollout:9000"


class TestPreInitialize:
    def test_calls_three_initializers(self, fake_env):
        """pre_initialize invokes dispatch, train_server and weight_updater init."""
        ctrl = make_controller(fake_env)
        ctrl.initialize_dispatch = MagicMock()
        ctrl.initialize_train_server = MagicMock()
        ctrl.initialize_weight_updater = MagicMock()
        ctrl.pre_initialize()
        ctrl.initialize_dispatch.assert_called_once()
        ctrl.initialize_train_server.assert_called_once()
        ctrl.initialize_weight_updater.assert_called_once()


class TestInitializeRollout:
    def test_calls_three_methods(self, fake_env):
        """initialize_rollout calls the three rollout methods."""
        ctrl = make_controller(fake_env)
        ctrl.send_initial_batch_groups_to_rollout = MagicMock()
        ctrl.unlock_rollout_unit = MagicMock()
        ctrl.clean_train_updated_weights = MagicMock()
        ctrl.initialize_rollout()
        ctrl.send_initial_batch_groups_to_rollout.assert_called_once()
        ctrl.unlock_rollout_unit.assert_called_once()
        ctrl.clean_train_updated_weights.assert_called_once()


class TestUpdateRolloutWeights:
    def test_update_weights_loop(self, fake_env):
        """Poll update_weights_finished until it returns True, then exit."""
        ctrl = make_controller(fake_env)
        ctrl._create_weight_dir = MagicMock(return_value=MagicMock())
        ctrl.weight_update_actor = MagicMock()
        ctrl.weight_update_actor.update_weights_to_file.remote = MagicMock()
        ctrl.weight_update_actor.update_weights_finished.remote.side_effect = [False, True]
        fake_env["fake_ray"].get.side_effect = lambda ref: ref

        with patch.object(fake_env["mod"], "time") as mock_time:
            mock_time.time.return_value = 1000.0
            ctrl.update_rollout_weights(iteration=1)

        assert ctrl.weight_update_actor.update_weights_to_file.remote.called
        assert fake_env["fake_ray"].get.call_count == 2


class TestDirHelpers:
    def test_get_old_iter_dirs(self, fake_env):
        """_get_old_iter_dirs returns sorted iteration numbers."""
        ctrl = make_controller(fake_env)
        FakePath = fake_env["fake_pathlib"].Path
        with patch.object(FakePath, "glob") as mock_glob:
            mock_path1 = MagicMock()
            mock_path1.is_dir.return_value = True
            mock_path1.name = "iter_0000003"
            mock_path2 = MagicMock()
            mock_path2.is_dir.return_value = True
            mock_path2.name = "iter_0000001"
            mock_glob.return_value = [mock_path1, mock_path2]

            ckpt_dir, all_iters = ctrl._get_old_iter_dirs()
            assert all_iters == [1, 3]

    def test_clean_old_iters_less_than_delta(self, fake_env):
        """When number of iters <= delta, nothing is deleted."""
        ctrl = make_controller(fake_env, delta=3)
        ckpt_dir = MagicMock()
        all_iters = [1, 2, 3]
        ctrl._clean_old_iters_than_delta(ckpt_dir, all_iters)
        fake_env["fake_shutil"].rmtree.assert_not_called()

    def test_clean_old_iters_more_than_delta(self, fake_env):
        """Old iterations beyond delta are removed."""
        ctrl = make_controller(fake_env, delta=2)
        ckpt_dir = MagicMock()
        all_iters = [1, 2, 3, 4]
        with patch("os.path.exists", return_value=True):
            ctrl._clean_old_iters_than_delta(ckpt_dir, all_iters)
        assert fake_env["fake_shutil"].rmtree.call_count == 2

    def test_create_weight_dir(self, fake_env):
        """_create_weight_dir returns the correct path and cleans old iters."""
        ctrl = make_controller(fake_env)
        ctrl._get_old_iter_dirs = MagicMock(return_value=(MagicMock(), [1, 2, 3]))
        ctrl._clean_old_iters_than_delta = MagicMock()
        result = ctrl._create_weight_dir(iteration=4)
        assert result is not None
        ctrl._clean_old_iters_than_delta.assert_called_once()

    def test_clean_weight_files(self, fake_env):
        """clean_weight_files deletes iterations >= given iteration."""
        ctrl = make_controller(fake_env)
        ctrl._get_old_iter_dirs = MagicMock(return_value=(MagicMock(), [1, 2, 3, 4]))
        ctrl.clean_weight_files(iteration=3)
        assert fake_env["fake_shutil"].rmtree.call_count == 2


class TestInitializeDispatch:
    def test_create_actor_called(self, fake_env):
        """initialize_dispatch uses create_actor with correct arguments."""
        ctrl = make_controller(fake_env)
        fake_utils = fake_env["fake_utils_mod"]
        fake_ray = fake_env["fake_ray"]
        mock_actor = MagicMock()
        fake_utils.create_actor.return_value = mock_actor
        fake_ray.get.return_value = None
        ctrl.initialize_dispatch()
        fake_utils.create_actor.assert_called_once()
        mock_actor.init_done.remote.assert_called_once()
        assert fake_ray.get.called


class TestInitializeWeightUpdater:
    def test_create_actor_called(self, fake_env):
        """initialize_weight_updater creates the weight updater actor."""
        ctrl = make_controller(fake_env)
        fake_utils = fake_env["fake_utils_mod"]
        fake_ray = fake_env["fake_ray"]
        ctrl.dispatch_actor = MagicMock()
        mock_actor = MagicMock()
        fake_utils.create_actor.return_value = mock_actor
        fake_ray.get.return_value = None
        ctrl.initialize_weight_updater()
        fake_utils.create_actor.assert_called_once()
        mock_actor.init_done.remote.assert_called_once()


class TestWaitForRolloutUnitReady:
    def test_ready_immediately(self, fake_env):
        """check_rollout_unit_ready returns True immediately."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.dispatch_actor.check_rollout_unit_ready.remote.return_value = True
        fake_env["fake_ray"].get.side_effect = lambda ref: ref
        ctrl.wait_for_rollout_unit_ready()
        assert fake_env["fake_ray"].get.called_once()

    def test_timeout_raises(self, fake_env):
        """If timeout exceeded, raise TimeoutError."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.dispatch_actor.check_rollout_unit_ready.remote.return_value = False
        fake_env["fake_ray"].get.side_effect = lambda ref: ref
        with patch.object(fake_env["mod"], "time") as mock_time:
            mock_time.time.side_effect = [0, 0, 31]
            with pytest.raises(TimeoutError, match="rollout unit did not signal"):
                ctrl.wait_for_rollout_unit_ready()


class TestDataIterComplete:
    def test_data_iter_complete(self, fake_env):
        """data_iter_complete returns the remote value."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.dispatch_actor.data_iter_finished.remote.return_value = "finished"
        fake_env["fake_ray"].get.return_value = "finished"
        result = ctrl.data_iter_complete()
        assert result == "finished"


class TestInitializeTrainServer:
    def test_server_creation_and_thread(self, fake_env):
        """initialize_train_server creates TrainServer, FastAPI app, and starts a thread."""
        ctrl = make_controller(fake_env)
        fake_threading = fake_env["fake_threading"]
        fake_fastapi = fake_env["fake_fastapi"]
        fake_train_server = fake_env["fake_train_server_mod"]
        ctrl.initialize_train_server()
        fake_train_server.TrainServer.assert_called_once()
        fake_fastapi.FastAPI.assert_called_once()
        app_instance = fake_fastapi.FastAPI.return_value
        app_instance.include_router.assert_called_once_with(ctrl.train_server.router)
        fake_threading.Thread.return_value.start.assert_called_once()


class TestRemoteMethods:
    def test_send_initial_batch_groups(self, fake_env):
        """send_initial_batch_groups_to_rollout calls dispatch actor."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.send_initial_batch_groups_to_rollout()
        ctrl.dispatch_actor.send_batch_groups.remote.assert_called_once_with(ctrl.init_num_group_batches)

    def test_unlock_rollout_unit(self, fake_env):
        """unlock_rollout_unit calls dispatch actor."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.unlock_rollout_unit()
        ctrl.dispatch_actor.unlock_rollout_unit.remote.assert_called_once()


class TestTrainingBatchQueueReady:
    def test_ready(self, fake_env):
        """Returns True when status 200 and is_ready true."""
        ctrl = make_controller(fake_env)
        result = ctrl.training_batch_queue_ready()
        assert result is True
        fake_env["fake_requests"].post.assert_called_once()

    def test_not_ready_status(self, fake_env):
        """Returns False when status code is not 200."""
        ctrl = make_controller(fake_env)
        fake_env["fake_requests"].post.return_value.status_code = 500
        result = ctrl.training_batch_queue_ready()
        assert result is False

    def test_not_ready_value(self, fake_env):
        """Returns False when is_ready is false."""
        ctrl = make_controller(fake_env)
        fake_env["fake_requests"].post.return_value.json.return_value = {"is_ready": False}
        result = ctrl.training_batch_queue_ready()
        assert result is False


class TestGetNextTrainingBatch:
    def test_successful_get(self, fake_env):
        """Successfully retrieves a training batch and unlocks rollout unit."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()          # avoid NoneType error
        fake_requests = fake_env["fake_requests"]
        fake_response = fake_requests.get.return_value
        fake_response.status_code = 200
        fake_response.headers = {"X-Metrics-Metadata": '{"key":"val"}'}
        fake_response.content = b"serialized_data"
        outputs, metric = ctrl.get_next_training_batch()
        fake_env["fake_torch"].load.assert_called_once()
        assert outputs == "torch_data"
        assert metric == {"key": "val"}

    def test_successful_last_iteration_no_unlock(self, fake_env):
        """When last_iteration=True, does not call unlock_rollout_unit."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.unlock_rollout_unit = MagicMock()
        fake_requests = fake_env["fake_requests"]
        fake_response = fake_requests.get.return_value
        fake_response.status_code = 200
        fake_response.headers = {}
        fake_response.content = b"serialized_data"
        outputs, metric = ctrl.get_next_training_batch(last_iteration=True)
        assert outputs == "torch_data"
        assert metric == {}
        ctrl.unlock_rollout_unit.assert_not_called()

    def test_request_exception_returns_none(self, fake_env):
        """When request fails, returns None, None."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        fake_env["fake_requests"].get.side_effect = Exception("network error")
        outputs, metric = ctrl.get_next_training_batch()
        assert outputs is None
        assert metric is None


class TestFinishTrainingIteration:
    def test_updates_timing_and_iter(self, fake_env):
        """Updates timing_training_unit and calls set_current_training_iter."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        with patch.object(fake_env["mod"], "time") as mock_time:
            mock_time.time.return_value = 2000.0
            ctrl.finish_training_iteration(5)
        assert ctrl.timing_training_unit == [2000.0]
        ctrl.dispatch_actor.set_current_training_iter.remote.assert_called_once_with(6)


class TestWaitForRolloutQuit:
    def test_waits_until_true(self, fake_env):
        """Polls until status becomes true."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.dispatch_actor.rollout_already_quit.remote.side_effect = [
            {"Status": "false"},
            {"Status": "true"},
        ]
        fake_env["fake_ray"].get.side_effect = lambda ref: ref
        with patch.object(fake_env["mod"], "time") as mock_time:
            ctrl.wait_for_rollout_quit()
        assert fake_env["fake_ray"].get.call_count == 2


class TestFinishTraining:
    def test_shutdown_and_kill(self, fake_env):
        """Calls shutdown, waits, and kills actors."""
        ctrl = make_controller(fake_env)
        ctrl.dispatch_actor = MagicMock()
        ctrl.weight_update_actor = MagicMock()
        ctrl.wait_for_rollout_quit = MagicMock()
        fake_utils = fake_env["fake_utils_mod"]
        ctrl.finish_training()
        ctrl.dispatch_actor.shutdown.remote.assert_called_once()
        ctrl.wait_for_rollout_quit.assert_called_once()
        fake_utils.kill_actor.assert_any_call(ctrl.dispatch_actor)
        fake_utils.kill_actor.assert_any_call(ctrl.weight_update_actor)


class TestCleanTrainUpdatedWeights:
    def test_directory_exists(self, fake_env):
        """Deletes all files when directory exists."""
        ctrl = make_controller(fake_env)
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("os.walk") as mock_walk, \
             patch("os.remove") as mock_remove:
            mock_walk.return_value = [("/tmp/weights", [], ["file1.txt", "file2.txt"])]
            ctrl.clean_train_updated_weights()
            assert mock_remove.call_count == 2

    def test_directory_not_exists(self, fake_env):
        """Does nothing when directory does not exist."""
        ctrl = make_controller(fake_env)
        with patch("os.path.exists", return_value=False), \
             patch("os.remove") as mock_remove:
            ctrl.clean_train_updated_weights()
            mock_remove.assert_not_called()
