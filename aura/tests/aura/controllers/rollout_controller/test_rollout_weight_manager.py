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
import os
import re
import shutil
import threading
from unittest.mock import MagicMock, patch, ANY
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree for rollout_weight_manager
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Construct isolated fake modules so rollout_weight_manager can be imported safely."""

    # ---- fake shutil ----
    fake_shutil = types.ModuleType("shutil")
    fake_shutil.rmtree = MagicMock()
    fake_shutil.move = MagicMock()

    # ---- fake traceback ----
    fake_traceback = types.ModuleType("traceback")
    fake_traceback.print_exc = MagicMock()

    # ---- fake threading (inherit real threading, only override Lock) ----
    import threading as _real_threading
    fake_threading = types.ModuleType("threading")
    fake_threading.__dict__.update(_real_threading.__dict__)
    class FakeLock:
        def acquire(self, blocking=True, timeout=-1):
            return True
        def release(self):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    fake_threading.Lock = FakeLock

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda fn: fn  # decorator that returns the class unchanged

    # ---- fake transformers.AutoConfig ----
    fake_transformers = types.ModuleType("transformers")
    fake_auto_config = types.ModuleType("transformers.AutoConfig")  # submodule
    fake_auto_config.from_pretrained = MagicMock(return_value=MagicMock())
    fake_transformers.AutoConfig = fake_auto_config

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = types.ModuleType("torch.distributed")
    fake_torch.distributed.checkpoint = types.ModuleType("torch.distributed.checkpoint")
    fake_torch.distributed.checkpoint.FileSystemReader = MagicMock()
    fake_torch.distributed.checkpoint.load = MagicMock()
    fake_torch.empty = MagicMock()
    fake_torch.bfloat16 = "bfloat16"

    # ---- fake safetensors.torch ----
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.save_file = MagicMock()
    fake_safetensors.torch = fake_safetensors_torch

    # ---- fake aura modules ----
    # loggers
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(
        return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger))
    )

    # globals
    fake_globals_mod = types.ModuleType("aura.base.utils.globals")
    fake_globals_mod.ROLLOUT_WEIGHTS_PREFIX = "_rollout"

    # rollout_weight_loader
    fake_loader_mod = types.ModuleType(
        "aura.controllers.rollout_controller.rollout_weight_loader"
    )
    fake_loader_mod.run_distributed_qwen3_assemble = MagicMock()

    # aura packages to locate real file
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
    fake_aura_controllers_rollout = types.ModuleType("aura.controllers.rollout_controller")
    fake_aura_controllers_rollout.__path__ = [
        os.path.join(base, "controllers/rollout_controller")
    ]

    fake_aura.controllers = fake_aura_controllers
    fake_aura_controllers.rollout_controller = fake_aura_controllers_rollout

    fakes = {
        "shutil": fake_shutil,
        "traceback": fake_traceback,
        "threading": fake_threading,
        "ray": fake_ray,
        "transformers": fake_transformers,
        "transformers.AutoConfig": fake_auto_config,
        "torch": fake_torch,
        "torch.distributed": fake_torch.distributed,
        "torch.distributed.checkpoint": fake_torch.distributed.checkpoint,
        "safetensors": fake_safetensors,
        "safetensors.torch": fake_safetensors_torch,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura.base.utils.globals": fake_globals_mod,
        "aura.controllers.rollout_controller.rollout_weight_loader": fake_loader_mod,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.base.utils": fake_aura_base_utils,
        "aura.controllers": fake_aura_controllers,
        "aura.controllers.rollout_controller": fake_aura_controllers_rollout,
    }

    target = "aura.controllers.rollout_controller.rollout_weight_manager"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.controllers.rollout_controller.rollout_weight_manager as mod
        yield {
            "mod": mod,
            "RolloutWeightManager": mod.RolloutWeightManager,
            "mock_logger": mock_logger,
            "fake_shutil": fake_shutil,
            "fake_traceback": fake_traceback,
            "fake_torch": fake_torch,
            "fake_dcp": fake_torch.distributed.checkpoint,
            "fake_safetensors_torch": fake_safetensors_torch,
            "fake_loader_mod": fake_loader_mod,
            "fake_globals_mod": fake_globals_mod,
            "fake_auto_config": fake_auto_config,
            "MAX_RETAIN_WEIGHTS_VERSION": mod.MAX_RETAIN_WEIGHTS_VERSION,
            "PATH_ITER_PATTERN": mod.PATH_ITER_PATTERN,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_manager(fake_env, **overrides):
    """Create a RolloutWeightManager with default valid arguments."""
    default_args = {
        "weight_save_dir": "/tmp/weights",
        "tokenizer_name_or_path": "/models/test",
        "trust_remote_code": True,
        "infer_tensor_parallel_size": 1,
        "train_tensor_parallel_size": 1,
        "infer_expert_parallel_size": 1,
        "enable_version_control": False,
        "use_on_policy": False,
        "model_name": "test_model",
    }
    default_args.update(overrides)
    return fake_env["RolloutWeightManager"](**default_args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestInit:
    def test_init_basic(self, fake_env):
        """Manager initializes with correct attributes and creates inference directory."""
        mgr = make_manager(fake_env)
        assert mgr.model_name == "test_model"
        assert mgr.model_path == "/models/test"
        assert mgr.weights_version == 0
        assert mgr.inference_save_path == "/tmp/weights_rollout"
        fake_env["mock_logger"].info.assert_called()  # at least one info call

    def test_one_step_off_ep_mode_true(self, fake_env):
        """When ONE_STEP_OFF_EP_MODE is 'true', infer_tp is infer_tensor_parallel_size."""
        with patch.dict(os.environ, {"ONE_STEP_OFF_EP_MODE": "true"}):
            mgr = make_manager(fake_env)
        assert mgr.infer_tp == 1  # infer_tensor_parallel_size
        assert mgr.head_dim_scale == 1

    def test_one_step_off_ep_mode_false(self, fake_env):
        """When ONE_STEP_OFF_EP_MODE is not 'true', infer_tp and head_dim_scale are calculated."""
        with patch.dict(os.environ, {"ONE_STEP_OFF_EP_MODE": "false"}):
            mgr = make_manager(fake_env, infer_tensor_parallel_size=2, train_tensor_parallel_size=4)
        # infer_tp = 2 * (4/2) = 4
        assert mgr.infer_tp == 4
        # head_dim_scale = 2 // 4 = 0 (integer division)
        assert mgr.head_dim_scale == 0


class TestGetWeightsVersion:
    def test_returns_current_version(self, fake_env):
        """get_weights_version returns the current version number."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 5
        assert mgr.get_weights_version() == 5


class TestCleanOldWeights:
    def test_weights_version_low_does_nothing(self, fake_env):
        """When weights_version <= MAX_RETAIN_WEIGHTS_VERSION, no directories are removed."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 2
        with patch("os.listdir") as mock_listdir:
            mgr.clean_old_weights()
            mock_listdir.assert_not_called()

    def test_removes_old_weight_dirs(self, fake_env):
        """Old weight directories beyond MAX_RETAIN_WEIGHTS_VERSION are deleted."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 5  # MAX_RETAIN=2, so keep versions 3,4,5, delete 1,2,0
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["weights_0", "weights_1", "weights_2", "weights_3", "other_dir"]
            with patch("os.path.isdir", return_value=True), patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                mgr.clean_old_weights()
        # Should delete weights_0, weights_1, weights_2 (weights_version - MAX_RETAIN = 3, so x < 3)
        assert fake_env["fake_shutil"].rmtree.call_count == 3


class TestUpdateMaxVersion:
    def test_increments_max_possible_version(self, fake_env):
        """update_max_version adds the given number to max_possible_version."""
        mgr = make_manager(fake_env)
        mgr.max_possible_version = 10
        mgr.update_max_version(3)
        assert mgr.max_possible_version == 13


class TestShouldWeightsUpdate:
    def test_input_smaller_returns_false(self, fake_env):
        """When input_weight_version <= current version, return False."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 3
        result = mgr._should_weights_update(3)
        assert result is False

    def test_resume_iteration_equal(self, fake_env):
        """When resume_iteration > 0 and equals input_weight_version, return True and reset max_possible_version."""
        mgr = make_manager(fake_env)
        mgr.resume_iteration = 5
        mgr.max_possible_version = 100
        result = mgr._should_weights_update(5)
        assert result is True
        assert mgr.max_possible_version == 5

    def test_use_on_policy_true(self, fake_env):
        """When use_on_policy is True, always return True."""
        mgr = make_manager(fake_env, use_on_policy=True)
        mgr.weights_version = 0
        result = mgr._should_weights_update(10)
        assert result is True

    def test_enable_version_control_required_version_match(self, fake_env):
        """When enable_version_control is True and input equals required, return True."""
        mgr = make_manager(fake_env, enable_version_control=True)
        mgr.max_possible_version = 5
        result = mgr._should_weights_update(4)  # required = 5-1 = 4
        assert result is True

    def test_enable_version_control_required_version_mismatch(self, fake_env):
        """When enable_version_control is True and input does not equal required, return False."""
        mgr = make_manager(fake_env, enable_version_control=True)
        mgr.max_possible_version = 5
        result = mgr._should_weights_update(3)
        assert result is False

    def test_default_returns_true(self, fake_env):
        """Default case without special flags returns True."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 0
        result = mgr._should_weights_update(5)
        assert result is True


class TestDoWeightsUpdateWithAssemble:
    def test_calls_assemble_and_updates_version(self, fake_env):
        """_do_weights_update_with_assemble calls run_distributed_qwen3_assemble and sets weights_version."""
        mgr = make_manager(fake_env)
        mgr.weights_version = 0
        mgr._do_weights_update_with_assemble("/path", 3)
        fake_env["fake_loader_mod"].run_distributed_qwen3_assemble.assert_called_once()
        assert mgr.weights_version == 3

    def test_assemble_exception_logs_error(self, fake_env):
        """When assemble raises, error is logged and traceback printed."""
        mgr = make_manager(fake_env)
        fake_env["fake_loader_mod"].run_distributed_qwen3_assemble.side_effect = Exception("assemble error")
        mgr._do_weights_update_with_assemble("/path", 3)
        fake_env["mock_logger"].error.assert_called()
        fake_env["fake_traceback"].print_exc.assert_called_once()
        assert mgr.weights_version == 0


class TestAggregateWorker:
    def test_loads_and_saves(self, fake_env):
        """aggregate_worker reads DCP checkpoint and saves as safetensors."""
        fake_dcp = fake_env["fake_dcp"]
        reader_mock = MagicMock()
        fake_dcp.FileSystemReader.return_value = reader_mock
        reader_mock.read_metadata.return_value = MagicMock(
            state_dict_metadata={"key": MagicMock(size=10, properties=MagicMock(dtype="float"))}
        )
        fake_env["fake_torch"].empty.return_value = "tensor"
        fake_dcp.load = MagicMock()
        fake_save = fake_env["fake_safetensors_torch"].save_file

        q = MagicMock()
        fake_env["mod"].aggregate_worker("/shards", "/out.safetensors", q)

        fake_dcp.FileSystemReader.assert_called_once_with("/shards")
        reader_mock.read_metadata.assert_called_once()
        fake_dcp.load.assert_called_once_with(state_dict=ANY, checkpoint_id="/shards")
        fake_save.assert_called_once_with(ANY, "/out.safetensors")
        q.put.assert_called_once_with(None)

    def test_exception_puts_error(self, fake_env):
        """aggregate_worker puts error message on exception."""
        fake_dcp = fake_env["fake_dcp"]
        fake_dcp.FileSystemReader.side_effect = Exception("read error")

        q = MagicMock()
        fake_env["mod"].aggregate_worker("/shards", "/out.safetensors", q)

        q.put.assert_called_once()
        assert "read error" in str(q.put.call_args[0][0])


class TestAggregateHFWeights:
    def test_success(self, fake_env):
        """aggregate_hf_weights spawns process and returns on success."""
        mgr = make_manager(fake_env)

        with patch.object(fake_env["mod"], "Process") as mock_process_class, \
             patch.object(fake_env["mod"], "Queue") as mock_queue_class:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = False
            mock_process.exitcode = 0
            mock_process_class.return_value = mock_process

            mock_q = MagicMock()
            mock_q.get.return_value = None
            mock_queue_class.return_value = mock_q

            mgr.aggregate_hf_weights("/shards", "/out.safetensors")

            mock_process_class.assert_called_once()
            mock_process.start.assert_called_once()
            mock_process.join.assert_called_once_with(timeout=1800)

    def test_timeout_raises(self, fake_env):
        """aggregate_hf_weights raises RuntimeError on timeout and cleans up output_file."""
        mgr = make_manager(fake_env)

        with patch.object(fake_env["mod"], "Process") as mock_process_class, \
             patch.object(fake_env["mod"], "Queue") as mock_queue_class, \
             patch("os.path.exists", return_value=True) as mock_exists, \
             patch("os.remove") as mock_remove:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = True
            mock_process_class.return_value = mock_process

            with pytest.raises(RuntimeError, match="timed out"):
                mgr.aggregate_hf_weights("/shards", "/out.safetensors")

            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()
            mock_process.join.assert_any_call(timeout=5)
            mock_process.join.assert_any_call()
            mock_exists.assert_called_with("/out.safetensors")
            mock_remove.assert_called_with("/out.safetensors")


    def test_nonzero_exit_raises(self, fake_env):
        """aggregate_hf_weights raises RuntimeError on non-zero exit code."""
        mgr = make_manager(fake_env)

        with patch.object(fake_env["mod"], "Process") as mock_process_class, \
             patch.object(fake_env["mod"], "Queue") as mock_queue_class:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = False
            mock_process.exitcode = 1
            mock_process_class.return_value = mock_process

            with pytest.raises(RuntimeError, match="exited with code 1"):
                mgr.aggregate_hf_weights("/shards", "/out.safetensors")

    def test_error_result_raises(self, fake_env):
        """aggregate_hf_weights raises RuntimeError with worker error message."""
        mgr = make_manager(fake_env)

        with patch.object(fake_env["mod"], "Process") as mock_process_class, \
             patch.object(fake_env["mod"], "Queue") as mock_queue_class:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = False
            mock_process.exitcode = 0
            mock_process_class.return_value = mock_process

            mock_q = MagicMock()
            mock_q.get.return_value = "worker failure"
            mock_queue_class.return_value = mock_q

            with pytest.raises(RuntimeError, match="worker failure"):
                mgr.aggregate_hf_weights("/shards", "/out.safetensors")

    def test_no_result_raises(self, fake_env):
        """aggregate_hf_weights raises RuntimeError when queue.get fails."""
        mgr = make_manager(fake_env)

        with patch.object(fake_env["mod"], "Process") as mock_process_class, \
             patch.object(fake_env["mod"], "Queue") as mock_queue_class:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = False
            mock_process.exitcode = 0
            mock_process_class.return_value = mock_process

            mock_q = MagicMock()
            mock_q.get.side_effect = Exception("queue empty")
            mock_queue_class.return_value = mock_q

            with pytest.raises(RuntimeError, match="no result returned"):
                mgr.aggregate_hf_weights("/shards", "/out.safetensors")


class TestMoveWeights:
    def test_move_safetensor_files(self, fake_env):
        """_move_weights creates destination directory, moves only .safetensors files, and updates version."""
        mgr = make_manager(fake_env)
        with patch("os.listdir") as mock_listdir, patch("os.path.isfile", return_value=True), \
             patch("os.path.exists", return_value=False), patch("os.makedirs") as mock_makedirs, \
             patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            mock_listdir.return_value = ["a.safetensors", "b.txt", "c.safetensors"]
            mgr._move_weights("/src", 7)

        # shutil.move should be called for a.safetensors and c.safetensors only
        assert fake_env["fake_shutil"].move.call_count == 2
        # weights_version updated
        assert mgr.weights_version == 7

    def test_move_weights_exception(self, fake_env):
        """When move fails, error is logged and traceback printed."""
        mgr = make_manager(fake_env)
        with patch("os.listdir", side_effect=Exception("listdir error")):
            mgr._move_weights("/src", 7)
        fake_env["mock_logger"].error.assert_called()
        fake_env["fake_traceback"].print_exc.assert_called_once()


class TestDoWeightsUpdateWithMegatron:
    def test_src_exists_not_empty(self, fake_env):
        """When source directory exists and is not empty, call _move_weights."""
        mgr = make_manager(fake_env)
        mgr._move_weights = MagicMock()
        with patch("os.path.exists", return_value=True), \
             patch("aura.controllers.rollout_controller.rollout_weight_manager.is_empty", return_value=False):
            mgr._do_weights_update_with_megatron("/src", 3)
        mgr._move_weights.assert_called_once_with("/src", 3)

    def test_src_missing_logs_error(self, fake_env):
        """When source directory does not exist, error logged and move not called."""
        mgr = make_manager(fake_env)
        mgr._move_weights = MagicMock()
        with patch("os.path.exists", return_value=False):
            mgr._do_weights_update_with_megatron("/src", 3)
        fake_env["mock_logger"].error.assert_called()
        mgr._move_weights.assert_not_called()

    def test_src_empty_logs_error(self, fake_env):
        """When source directory is empty, error logged and move not called."""
        mgr = make_manager(fake_env)
        mgr._move_weights = MagicMock()
        with patch("os.path.exists", return_value=True), \
             patch("aura.controllers.rollout_controller.rollout_weight_manager.is_empty", return_value=True):
            mgr._do_weights_update_with_megatron("/src", 3)
        fake_env["mock_logger"].error.assert_called()
        mgr._move_weights.assert_not_called()


class TestDoWeightsUpdateWithFSDP:
    def test_fsdp_aggregate_and_move(self, fake_env):
        """FSDP strategy aggregates HF weights then moves."""
        mgr = make_manager(fake_env)
        mgr.aggregate_hf_weights = MagicMock()
        mgr._move_weights = MagicMock()
        with patch("os.path.exists", return_value=True), \
             patch("aura.controllers.rollout_controller.rollout_weight_manager.is_empty", return_value=False):
            mgr._do_weights_update_with_fsdp("/src", 5)
        mgr.aggregate_hf_weights.assert_called_once_with("/src", os.path.join("/src", "model.safetensors"))
        mgr._move_weights.assert_called_once_with("/src", 5)

    def test_fsdp_src_missing(self, fake_env):
        """When src dir missing, error logged and no aggregate/move."""
        mgr = make_manager(fake_env)
        mgr.aggregate_hf_weights = MagicMock()
        mgr._move_weights = MagicMock()
        with patch("os.path.exists", return_value=False):
            mgr._do_weights_update_with_fsdp("/src", 5)
        fake_env["mock_logger"].error.assert_called()
        mgr.aggregate_hf_weights.assert_not_called()


class TestDoWeightsUpdate:
    @pytest.mark.parametrize("strategy,method_name", [
        ("megatron", "_do_weights_update_with_megatron"),
        ("fsdp", "_do_weights_update_with_fsdp"),
    ])
    def test_strategy_selection(self, fake_env, strategy, method_name):
        """_do_weights_update delegates to correct strategy method based on WEIGHT_SAVE_STRATEGY."""
        mgr = make_manager(fake_env)
        mgr._do_weights_update_with_megatron = MagicMock()
        mgr._do_weights_update_with_fsdp = MagicMock()
        mgr._do_weights_update_with_assemble = MagicMock()
        with patch.dict(os.environ, {"WEIGHT_SAVE_STRATEGY": strategy}):
            mgr._do_weights_update("/path", 4)
        getattr(mgr, method_name).assert_called_once_with("/path", 4)

    def test_default_assemble(self, fake_env):
        """When no WEIGHT_SAVE_STRATEGY set, assemble method is used."""
        mgr = make_manager(fake_env)
        mgr._do_weights_update_with_assemble = MagicMock()
        with patch.dict(os.environ, {}, clear=True):  # remove the var if set
            mgr._do_weights_update("/path", 4)
        mgr._do_weights_update_with_assemble.assert_called_once_with("/path", 4)


class TestSyncWeightsUpdate:
    def test_sync_weights_update_called_and_lock(self, fake_env):
        """sync_weights_update extracts weight_iter, checks and performs update."""
        mgr = make_manager(fake_env)
        mgr.clean_old_weights = MagicMock()
        mgr._should_weights_update = MagicMock(return_value=True)
        mgr._do_weights_update = MagicMock()
        # simulate path matching regex
        path = "/weights/iter_0000005"
        mgr.sync_weights_update(path)
        mgr.clean_old_weights.assert_called_once()
        mgr._should_weights_update.assert_called_once_with(5)
        mgr._do_weights_update.assert_called_once_with(path, 5)

    def test_should_not_update_skips(self, fake_env):
        """When _should_weights_update returns False, _do_weights_update is not called."""
        mgr = make_manager(fake_env)
        mgr.clean_old_weights = MagicMock()
        mgr._should_weights_update = MagicMock(return_value=False)
        mgr._do_weights_update = MagicMock()
        mgr.sync_weights_update("/weights/iter_0000003")
        mgr._do_weights_update.assert_not_called()


class TestInitDone:
    def test_init_done_noop(self, fake_env):
        """init_done does nothing."""
        mgr = make_manager(fake_env)
        mgr.init_done()  # should not raise
