# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
import ast
import asyncio
import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aura.trainer.rollout.trajectory_validation import (
    PromptTooLongError, validate_group_ids, validate_trajectory_batch,
)


def trajectory(prompt_id, idx, prompt=None):
    return {"prompt_id": str(prompt_id), "idx": str(idx),
            "prompt_tokens": prompt if prompt is not None else [int(prompt_id) + 10, 3],
            "response_tokens": [20, 21], "response_masks": [1, 1],
            "logprobs": [-0.1, -0.2], "trajectory_reward": 1.0}


def complete_batch():
    return [trajectory(1, 0), trajectory(1, 1), trajectory(2, 2), trajectory(2, 3)]


@pytest.mark.parametrize("ids", [[], [1], [1, 1, 2], [1, 1, 1, 2], [None, None, 2, 2]])
def test_uid_rejects_incomplete_or_colliding_groups(ids):
    with pytest.raises(ValueError):
        validate_group_ids(ids, 2, 4)


def test_interleaved_valid_uids_and_real_batch():
    validate_group_ids([1, "2", "1", 2], 2, 4)
    validate_trajectory_batch(complete_batch(), 2, 4)


def test_real_overlong_filter_mask_does_not_break_grpo_group():
    batch = complete_batch()
    batch[0]['response_masks'] = [0, 0]
    validate_trajectory_batch(batch, 2, 4)
    assert len(batch) == 4 and batch[0]['response_masks'] == [0, 0]


@pytest.mark.parametrize("mutation", [
    "duplicate_idx", "different_prompt", "dummy", "missing_logprob", "short_logprob", "short_mask", "empty",
])
def test_batch_rejects_semantic_corruption(mutation):
    batch = complete_batch()
    if mutation == "duplicate_idx":
        batch[1]["idx"] = batch[0]["idx"]
    elif mutation == "different_prompt":
        batch[1]["prompt_tokens"] = [99]
    elif mutation == "dummy":
        batch[0]["trajectory"] = {"data_id": "0" * 33}
    elif mutation == "missing_logprob":
        del batch[0]["logprobs"]
    elif mutation == "short_logprob":
        batch[0]["logprobs"] = [-0.1]
    elif mutation == "short_mask":
        batch[0]["response_masks"] = [1]
    else:
        batch[0]["prompt_tokens"] = []
    with pytest.raises(ValueError):
        validate_trajectory_batch(batch, 2, 4)


def worker_type():
    # Compile the actual worker methods without importing Ray/NPU dependencies.
    source = Path(__file__).resolve().parents[4] / "aura/trainer/rollout/rollout_worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RolloutWorker")
    names = {"stream_generate_trajectories", "get_train_batch_traj", "trajectories_collect_done",
             "multi_batches_final_handle", "multi_batches_generate_sequences", "handle_full_batch_trajectories"}
    methods = [n for n in worker.body if getattr(n, "name", "") in names]
    functions = [n for n in tree.body if getattr(n, "name", "") in {"clean_traj_groups", "get_all_prompt_ids"}]
    cls = ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"asyncio": asyncio, "logger": MagicMock(), "defaultdict": defaultdict,
                 "PromptTooLongError": PromptTooLongError, "validate_trajectory_batch": validate_trajectory_batch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions + [cls], type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["Worker"]


def make_worker():
    worker = worker_type()()
    worker.iteration = 0
    worker.use_stepwise_advantage = False
    worker.n_samples_per_prompt = 2
    worker.global_batch_size = 2
    worker.terminate_trajectories = 0
    worker.retry_limit = 3
    worker.trajectory_timeout = 0.1
    worker.rollout_engine = SimpleNamespace(server_addresses=[])
    worker.early_termination_requests = AsyncMock()
    return worker


@pytest.mark.parametrize("rows", [
    [trajectory(1, 0), trajectory(2, 2), trajectory(2, 3)],
    [],
    [trajectory(1, 0), None, trajectory(2, 2), trajectory(2, 3)],
])
def test_partial_or_all_failed_batch_never_publishes_dummy(rows):
    worker = make_worker()
    published = []
    worker.handle_full_batch_trajectories = lambda *args: published.append(args[-1])
    async def stream(*args, **kwargs):
        for row in rows:
            yield row
    worker.stream_generate_trajectories = stream
    tasks = [SimpleNamespace(prompt_id=i) for i in (1, 1, 2, 2)]
    with pytest.raises(ValueError, match="Incomplete GRPO"):
        asyncio.run(worker.multi_batches_generate_sequences(tasks, None, [0, 1, 2, 3], 0, 0, 1))
    assert published == []


def test_out_of_order_completion_preserves_whole_groups():
    worker = make_worker()
    published = []
    def publish(*args):
        validate_trajectory_batch(args[-1], 2, 4)
        published.append(args[-1])
    worker.handle_full_batch_trajectories = publish
    rows = complete_batch()
    async def stream(*args, **kwargs):
        for i in (0, 2, 1, 3):
            yield rows[i]
    worker.stream_generate_trajectories = stream
    tasks = [SimpleNamespace(prompt_id=i) for i in (1, 1, 2, 2)]
    asyncio.run(worker.multi_batches_generate_sequences(tasks, None, [0, 1, 2, 3], 0, 0, 1))
    assert len(published) == 1


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_batch_capacity_cannot_split_a_group(limit):
    with pytest.raises(ValueError, match="multiple"):
        make_worker().get_train_batch_traj({"1": [1, 2]}, limit, 2)


def test_timeout_does_not_count_as_completed_training_sample():
    worker = make_worker()
    worker.terminate_trajectories = 2
    assert not worker.trajectories_collect_done([1, 2], 4, 0, 1)


@pytest.mark.parametrize("failure,calls", [(PromptTooLongError("513 > 512"), 1), (RuntimeError("network"), 3)])
def test_generation_failure_propagates_and_deterministic_input_is_not_retried(failure, calls):
    worker = make_worker()
    task = SimpleNamespace(task_id="1", prompt_id=1)
    router = SimpleNamespace(generate_trajectory=AsyncMock(side_effect=failure))
    async def consume():
        return [row async for row in worker.stream_generate_trajectories([task], router, mode="Token")]
    with pytest.raises(RuntimeError, match="refusing partial"):
        asyncio.run(consume())
    assert router.generate_trajectory.await_count == calls


def test_none_trajectory_is_not_silently_lost():
    worker = make_worker()
    router = SimpleNamespace(generate_trajectory=AsyncMock(return_value=None))
    async def consume():
        return [row async for row in worker.stream_generate_trajectories(
            [SimpleNamespace(task_id="1", prompt_id=1)], router, mode="Token")]
    with pytest.raises(RuntimeError, match="refusing partial"):
        asyncio.run(consume())


def test_invalid_batch_rejected_before_any_transformation_or_put_data():
    worker = make_worker()
    worker.data_manager = SimpleNamespace(put_data=MagicMock())
    with pytest.raises(ValueError):
        worker.handle_full_batch_trajectories([0, 1, 2, 3], 0, 0, complete_batch()[:3])
    worker.data_manager.put_data.assert_not_called()


def test_failed_request_cancels_sibling_inflight_task():
    worker = make_worker()
    cancelled = []
    async def generate_trajectory(task, **kwargs):
        if task.task_id == "bad":
            await asyncio.sleep(0)
            raise PromptTooLongError("513 > 512")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(task.task_id)
    router = SimpleNamespace(generate_trajectory=generate_trajectory)
    tasks = [SimpleNamespace(task_id=t, prompt_id=1) for t in ("bad", "slow")]
    async def consume():
        return [row async for row in worker.stream_generate_trajectories(tasks, router, mode="Token")]
    with pytest.raises(RuntimeError):
        asyncio.run(consume())
    assert cancelled == ["slow"]


@pytest.mark.parametrize("ids", [[1, 1, 2], [1, 1, 1, 2]])
def test_training_consumer_rejects_bad_uid_before_creating_dataproto(ids):
    from contextlib import nullcontext
    source = Path(__file__).resolve().parents[4] / "aura/trainer/train_adapter/verl/full_async/full_async_trainer.py"
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "FullyAsyncTrainer")
    method = next(n for n in cls.body if getattr(n, "name", "") == "_fit_generate")
    class DataProto:
        from_dict = MagicMock()
    namespace = {"DataProto": DataProto, "marked_timer": lambda *a, **k: nullcontext(), "logger": MagicMock()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    worker = SimpleNamespace(timing_raw={}, required_samples=4,
                             config=SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="grpo"),
                                                    extras=SimpleNamespace(work_mode="one_step_off"),
                                                    data=SimpleNamespace(train_batch_size=2),
                                                    actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2))),
                             _get_samples_from_queue=lambda: (0, {"prompt_ids": ids}))
    with pytest.raises(ValueError):
        asyncio.run(namespace["_fit_generate"](worker))
    DataProto.from_dict.assert_not_called()


def test_wrapper_releases_agent_even_when_generator_raises():
    source = Path(__file__).resolve().parents[4] / "aura/runner/agent_engine_wrapper/rllm/rllm_engine_wrapper.py"
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "RLLMEngineWrapper")
    method = next(n for n in cls.body if getattr(n, "name", "") == "generate_trajectory")
    namespace = {"AgentTask": object, "Queue": object, "Trajectory": object, "logger": MagicMock()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    async def fail(**kwargs):
        raise PromptTooLongError("513 > 512")
        yield
    engine = SimpleNamespace(trajectory_generator=fail, release_env_and_agent=MagicMock(),
                             update_env_and_agent=MagicMock(), init_router=MagicMock())
    wrapper = SimpleNamespace(engine=engine, env_class=MagicMock(), env_args={}, agent_class=MagicMock(),
                              agent_args={}, _create_engine=MagicMock())
    task = SimpleNamespace(iteration=0, sample_id=0, task_id="1", model_dump=lambda: {"prompt_id": 1})
    with pytest.raises(PromptTooLongError):
        asyncio.run(namespace["generate_trajectory"](wrapper, task))
    engine.release_env_and_agent.assert_called_once_with("1")
