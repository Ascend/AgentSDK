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

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def rollout_worker_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[5]
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)

    ray = MagicMock()
    ray.remote.side_effect = lambda obj=None, **kwargs: obj if obj is not None else (lambda cls: cls)
    monkeypatch.setitem(sys.modules, "ray", ray)

    class FakeTensor:
        pass

    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.npu = MagicMock()
    torch.nn = MagicMock()
    torch.concat = MagicMock()
    torch.where = MagicMock()
    torch.tensor = MagicMock()
    torch.float64 = "float64"
    monkeypatch.setitem(sys.modules, "torch", torch)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    loggers = types.ModuleType("aura.base.log.loggers")
    loggers.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=MagicMock())))
    monkeypatch.setitem(sys.modules, "aura.base.log.loggers", loggers)

    misc = types.ModuleType("aura.base.misc.misc")
    misc.app_stats = MagicMock()
    monkeypatch.setitem(sys.modules, "aura.base.misc.misc", misc)

    globals_mod = types.ModuleType("aura.base.utils.globals")
    globals_mod.ROLLOUT_WEIGHTS_PREFIX = "/rollout"
    monkeypatch.setitem(sys.modules, "aura.base.utils.globals", globals_mod)

    queue_mod = types.ModuleType("aura.controllers.rollout_controller.rollout_queue")
    queue_mod.get_rollout_queue_actor = MagicMock()
    monkeypatch.setitem(sys.modules, "aura.controllers.rollout_controller.rollout_queue", queue_mod)

    utils_mod = types.ModuleType("aura.controllers.utils.utils")
    utils_mod.DEFAULT_SLEEP_TIME = 0.01
    monkeypatch.setitem(sys.modules, "aura.controllers.utils.utils", utils_mod)

    data_manager = types.ModuleType("aura.data_manager.data_manager")
    data_manager.DataManager = MagicMock()
    monkeypatch.setitem(sys.modules, "aura.data_manager.data_manager", data_manager)

    agent_router = types.ModuleType("aura.runner.agent_router")
    agent_router.AgentRouter = MagicMock()
    monkeypatch.setitem(sys.modules, "aura.runner.agent_router", agent_router)

    async_server = types.ModuleType("aura.runner.infer_adapter.async_server")
    async_server.AsyncServerManager = MagicMock()
    async_server.AsyncServerProxyManager = MagicMock()
    monkeypatch.setitem(sys.modules, "aura.runner.infer_adapter.async_server", async_server)

    sys.modules.pop("aura.trainer.rollout.rollout_worker", None)
    return importlib.import_module("aura.trainer.rollout.rollout_worker")


def test_cfg_get_supports_dict_object_and_none(rollout_worker_module):
    rw = rollout_worker_module

    assert rw._cfg_get({"beam_size": 4}, "beam_size") == 4
    assert rw._cfg_get(SimpleNamespace(beam_size=8), "beam_size") == 8
    assert rw._cfg_get(None, "beam_size", 3) == 3


def test_prompt_id_and_reward_helpers_handle_dict_and_object(rollout_worker_module):
    rw = rollout_worker_module
    dict_traj = {"trajectory": {"application_id": "12-3"}, "reward": 1.0}
    obj_traj = SimpleNamespace(application_id="7-1", reward=2.0)

    assert rw.get_trajectory_prompt_id(dict_traj) == "12"
    assert rw.get_trajectory_prompt_id(obj_traj) == "7"
    assert rw.get_trajectory_reward(dict_traj) == 1.0
    assert rw.get_trajectory_reward(obj_traj) == 2.0

    rw.set_trajectory_reward(dict_traj, 3.5)
    rw.set_trajectory_reward(obj_traj, 4.5)

    assert dict_traj["trajectory_reward"] == 3.5
    assert obj_traj.reward == 4.5


def test_tree_stepwise_pool_and_concurrency_use_beam_config(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.use_stepwise_advantage = True
    worker.trajectory_generation_method = "tree"
    worker.n_samples_per_prompt = 4
    worker.beam_train_n_samples = 0
    worker.agentic_env_config = {"beam_size": 3, "per_beam_expand": 2}

    assert worker._stepwise_layer_pool_size() == 6
    assert worker._samples_per_prompt_for_collection() == 6
    assert worker._rollout_batch_concurrency(agent_task_count=4, actual_batch_num=2) == 12

    worker.use_stepwise_advantage = False

    assert worker._samples_per_prompt_for_collection() == 4
    assert worker._rollout_batch_concurrency(agent_task_count=4, actual_batch_num=2) == 8


def test_stepwise_chain_collection_uses_step_mode(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.use_stepwise_advantage = True
    worker.trajectory_generation_method = "chain"
    worker.n_samples_per_prompt = 2
    worker.agentic_env_config = {}
    worker.beam_train_n_samples = 0
    worker.terminate_trajectories = 0
    worker.iteration = 0

    captured = {}

    async def fake_stream_generate_trajectories(agent_tasks, agent_router, mode, concurrency):
        captured["mode"] = mode
        captured["concurrency"] = concurrency
        if False:
            yield None

    worker.stream_generate_trajectories = fake_stream_generate_trajectories
    worker.multi_batches_final_handle = MagicMock()

    agent_tasks = [SimpleNamespace(prompt_id=0), SimpleNamespace(prompt_id=1)]
    asyncio.run(worker.multi_batches_generate_sequences(agent_tasks, MagicMock(), [21, 22], 1.0, 0.0, 1))

    assert captured["mode"] == "Step"
    assert captured["concurrency"] == 4


def test_select_beam_trajectories_uses_deterministic_seed(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.beam_select_seed = 123
    worker.beam_correct_threshold = 0.8
    worker.beam_partial_threshold = 0.0

    trajectories = [
        {"prompt_index": 0, "idx": idx, "metrics": {"res_reward": 1.0}}
        for idx in range(8)
    ]

    first = worker.select_beam_trajectories(list(trajectories), target_per_prompt=3)
    second = worker.select_beam_trajectories(list(reversed(trajectories)), target_per_prompt=3)

    assert [item["idx"] for item in first] == [item["idx"] for item in second]


def test_trajectory_group_key_prefers_prompt_id_before_idx(rollout_worker_module):
    rw = rollout_worker_module

    assert rw.get_trajectory_group_key({"prompt_id": "p0", "idx": 3}, 0) == "p0"
    assert rw.get_trajectory_group_key({"prompt_index": 2, "idx": 3}, 0) == "2"
    assert rw.get_trajectory_group_key({"idx": 3}, 0) == "3"


def test_default_handle_full_batch_keeps_input_indexes(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.use_stepwise_advantage = False
    worker.trajectory_generation_method = "chain"
    worker.beam_train_n_samples = 0
    worker.n_samples_per_prompt = 2
    worker.global_batch_size = 2
    worker.iteration = 0
    worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    worker.remove_padding_and_split_to_list = None
    worker.data_manager = MagicMock()
    worker.write_file = MagicMock()
    worker.add_output_for_verl = MagicMock()
    worker.add_output_for_msrl = MagicMock()
    worker.hybrid_mode_metrics_handle = MagicMock()
    worker.stepwise_normalization = MagicMock()

    final_gen_batch_output = {
        "responses": [[4], [5]],
        "input_ids": [[1, 4], [2, 5]],
        "prompt_length": [1, 1],
        "token_level_scores": [[0.0], [1.0]],
        "traj_mask": [[1], [1]],
        "index_in_batch_list": [],
    }
    worker._transform_agent_trajectories = MagicMock(return_value=(final_gen_batch_output, {}))

    indexes = [101, 205]
    trajectories = [{"prompt_id": "0", "idx": 0, "reward": 0.0}, {"prompt_id": "1", "idx": 1, "reward": 1.0}]
    worker.handle_full_batch_trajectories(indexes, 10.0, 0.5, trajectories)

    worker.data_manager.put_data.assert_called_once()
    assert worker.data_manager.put_data.call_args.args[1] == indexes


def test_tree_non_stepwise_normalization_resets_rewards(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.use_stepwise_advantage = False
    worker.trajectory_generation_method = "tree"
    worker.reset_trajectory_reward = MagicMock()

    trajectories = [{"prompt_index": 0, "reward": 1.0}]
    worker.stepwise_normalization(trajectories)

    worker.reset_trajectory_reward.assert_called_once_with(trajectories)


def test_tree_stepwise_normalization_keeps_raw_rewards(rollout_worker_module):
    rw = rollout_worker_module
    worker = object.__new__(rw.RolloutWorker)
    worker.use_stepwise_advantage = True
    worker.trajectory_generation_method = "tree"
    worker.data_manager = MagicMock()
    worker.reset_trajectory_reward = MagicMock()

    trajectories = [{"prompt_index": 0, "reward": 1.0}]
    worker.stepwise_normalization(trajectories)

    worker.reset_trajectory_reward.assert_not_called()
    worker.data_manager.update_metrics.assert_called()
