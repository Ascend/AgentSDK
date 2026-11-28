"""CPU tensor tests; no model or NPU initialization."""
from dataclasses import replace
import io

import pytest

from aura.memory.replay.config import ReplayConfig
from aura.memory.replay.conversion import trajectory_groups, training_batch
from aura.memory.replay.codec import ReplayCodec

torch = pytest.importorskip("torch")


def config(tmp_path, **changes):
    options = dict(replay_ratio=0.3, replay_buffer_size=100, group_size=2, batch_groups=2, max_lag=8, seed=4,
        store_id="store", source_experiment_id="experiment-a", source_run_id="run-a",
        policy_lineage_id="lineage-a", model_signature="model-a", tokenizer_signature="tokenizer-template-a",
        task_signature="gsm8k/train", reward_signature="math-v1", path=str(tmp_path / "store.db"))
    options.update(changes)
    return ReplayConfig(**options)


def records(count=2, prefix=""):
    result = []
    for group in range(count):
        for member in range(2):
            response = [8, 0, 9] if member == 0 else [7, 6]
            result.append(dict(prompt_id=prefix+str(group), idx=f"{prefix}{group}/{member}",
                prompt_tokens=torch.tensor([5, 6+group]), response_tokens=torch.tensor(response),
                response_masks=torch.tensor([1, 0, 1] if member == 0 else [1, 1]),
                logprobs=[-0.25, 0, -0.5] if member == 0 else [-0.125, -0.5],
                trajectory_reward=0.9 if member == 0 else 0.1))
    return result


def test_real_trajectory_conversion_stable_attempt_ids_and_exact_behavior(tmp_path):
    settings = config(tmp_path)
    a = trajectory_groups(records(), settings, "attempt-a:batch-1", 3)
    repeat = trajectory_groups(records(), settings, "attempt-a:batch-1", 3)
    other = trajectory_groups(records(), settings, "attempt-b:batch-1", 3)
    assert a == repeat and a[0].samples[0].trajectory_id != other[0].samples[0].trajectory_id
    assert a[0].samples[0].old_log_probs.data == ReplayCodec.decode(ReplayCodec.encode(a[0].samples[0])).old_log_probs.data
    assert {s.weight_version for g in a for s in g.samples} == {3}


def test_training_repadding_keeps_real_pad_valued_token_and_tool_mask(tmp_path):
    groups = trajectory_groups(records(), config(tmp_path), "batch", 3)
    batch = training_batch(groups, pad_token_id=0)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]
    assert batch["responses"][0].tolist() == [8, 0, 9]
    assert batch["response_mask"][0].tolist() == [1, 0, 1]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 0]
    assert batch["rollout_log_probs"][1].tolist() == [-0.125, -0.5, 0]
    assert batch["token_level_rewards"][0].tolist()[-1] == pytest.approx(0.9)
    assert len(set(batch["prompt_ids"])) == 2
    assert batch["prompt_ids"][0] == batch["prompt_ids"][1]
    # Existing transport's safe primitive references must survive weights_only load.
    batch["replay_refs"] = [{"trajectory_id": "t", "checksum": "a"*64, "created_seq": 1, "source_store_id": "store"}]
    buf = io.BytesIO(); torch.save(batch, buf); buf.seek(0)
    assert torch.load(buf, weights_only=True)["replay_refs"] == batch["replay_refs"]


def test_overlong_filtered_real_trajectory_survives_codec_and_group_conversion(tmp_path):
    raw = records()
    raw[0]['response_masks'] = torch.zeros(3, dtype=torch.long)
    groups = trajectory_groups(raw, config(tmp_path), 'masked', 3)
    sample = ReplayCodec.decode(ReplayCodec.encode(groups[0].samples[0]))
    assert sample.train_eligible
    batch = training_batch(groups, 0)
    assert batch['response_mask'][0].tolist() == [0, 0, 0]
    assert len(batch['prompt_ids']) == 4 and len(set(batch['prompt_ids'])) == 2


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
def test_publisher_byte_chunks_are_complete_and_idempotent(tmp_path, monkeypatch, backend):
    import sys
    from types import SimpleNamespace
    from aura.memory.replay import runtime
    from aura.memory.replay.sqlite_store import SQLiteReplayStore
    from aura.memory.replay.json_store import JsonReplayStore
    cfg = config(tmp_path)
    raw = records()
    groups = trajectory_groups(raw, cfg, 'bytes', 3)
    sizes = [sum(len(ReplayCodec.prepare(s)[1]) for s in g.samples) for g in groups]
    # Small actual encoded payloads exercise the same byte boundary without a large allocation.
    monkeypatch.setattr(runtime, 'MAX_BATCH_BYTES', max(sizes))
    store_type = SQLiteReplayStore if backend == 'sqlite' else JsonReplayStore
    store = store_type(str(tmp_path / ('publish-' + backend)), store_id='store')
    calls = []
    def append(samples, request):
        calls.append((len(samples), request))
        assert len({s.group_id for s in samples}) == 1
        return store.append_trajectories(samples, request)
    actor = SimpleNamespace(append_replay_batch=SimpleNamespace(remote=append))
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(get=lambda value, timeout: value))
    try:
        first = runtime.persist_trajectories(raw, cfg, actor, 'bytes', 3)
        assert first == runtime.persist_trajectories(raw, cfg, actor, 'bytes', 3)
        assert len(calls) == 4 and calls[:2] == calls[2:]
        assert len({r['created_seq'] for r in first}) == 4
        calls.clear()
        monkeypatch.setattr(runtime, 'MAX_BATCH_BYTES', min(sizes) - 1)
        with pytest.raises(ValueError, match='Complete replay group'):
            runtime.persist_trajectories(raw, cfg, actor, 'oversized', 3)
        assert calls == []
    finally:
        store.close()


def test_publisher_count_chunks_keep_groups(tmp_path):
    from aura.memory.replay.runtime import _persistence_chunks
    cfg = config(tmp_path, batch_groups=130, replay_buffer_size=1024)
    groups = trajectory_groups(records(130), cfg, 'count', 3)
    chunks = _persistence_chunks(groups)
    assert [(offset, len(samples)) for offset, samples in chunks] == [(0, 256), (128, 4)]


@pytest.mark.parametrize("mutation", ["missing", "partial", "dummy", "duplicate", "mixed_prompt", "positive"])
def test_invalid_generation_fails_before_publish(tmp_path, mutation):
    raw = records()
    if mutation == "missing": raw[0].pop("logprobs")
    if mutation == "partial": raw.pop()
    if mutation == "dummy": raw[0]["trajectory"] = {"data_id": "0"*33}
    if mutation == "duplicate": raw[1]["idx"] = raw[0]["idx"]
    if mutation == "mixed_prompt": raw[1]["prompt_tokens"] = torch.tensor([99])
    if mutation == "positive": raw[0]["logprobs"][0] = 0.1
    with pytest.raises((ValueError, KeyError)):
        trajectory_groups(raw, config(tmp_path), "bad", 3)


def test_different_generations_of_same_prompt_keep_separate_grpo_uid(tmp_path):
    settings = config(tmp_path)
    a = trajectory_groups(records(), settings, "batch-a", 3)
    b = trajectory_groups(records(), settings, "batch-b", 3)
    mixed = training_batch((a[0], b[0]), 0)
    assert len(set(mixed["prompt_ids"])) == 2


def test_full_input_positions_and_left_padding(tmp_path):
    raw = records()
    raw[2]["prompt_tokens"] = raw[3]["prompt_tokens"] = torch.tensor([5])
    groups = trajectory_groups(raw, config(tmp_path), "batch", 3)
    batch = training_batch(groups, 0)
    assert batch["prompts"][2].tolist() == [0, 5]
    assert batch["attention_mask"][2].tolist() == [0, 1, 1, 1, 1]
    assert batch["position_ids"][2].tolist() == [0, 0, 1, 2, 3]


@pytest.mark.parametrize("failed", [False, True])
def test_real_rollout_publish_waits_for_receipt_before_queue(tmp_path, monkeypatch, failed):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import sys
    from test_replay_backpressure import method
    from aura.trainer.rollout.trajectory_validation import validate_trajectory_batch
    from aura.memory.replay import runtime

    events = []
    def commit(*args):
        events.append("durable")
        if failed:
            raise OSError("injected actor write failure")
        return [{"trajectory_id": "committed-reference"}]
    monkeypatch.setattr(runtime, "persist_trajectories", commit)
    monkeypatch.setitem(sys.modules, "aura.base.analysis.data_analysis", SimpleNamespace(json_save_data=Mock()))
    tensors = {"responses": None, "input_ids": None, "prompt_length": None,
               "token_level_scores": None, "traj_mask": None}
    obj = SimpleNamespace(
        use_stepwise_advantage=False, n_samples_per_prompt=2, global_batch_size=2, iteration=0,
        stepwise_normalization=Mock(), _transform_agent_trajectories=lambda _: (tensors, {}),
        remove_padding_and_split_to_list=None, add_output_for_verl=Mock(), add_output_for_msrl=Mock(),
        _replay_config=config(tmp_path), _replay_actor=object(), _replay_producer_id="attempt",
        _replay_generation_version=3, current_weights_version=3, write_file=Mock(),
        stepwise_pad_datapro=Mock(), data_manager=SimpleNamespace(put_data=Mock(side_effect=lambda *args: events.append("queue"))),
        hybrid_mode_metrics_handle=Mock())
    fn = method("aura/trainer/rollout/rollout_worker.py", "RolloutWorker", "handle_full_batch_trajectories",
                validate_trajectory_batch=validate_trajectory_batch, logger=Mock(),
                time=SimpleNamespace(time=lambda: 0), _stat_rollout_metrics=lambda *args: {}, app_stats=Mock())
    if failed:
        with pytest.raises(OSError):
            fn(obj, [0, 1, 2, 3], 0, 0, records())
        assert events == ["durable"]
        obj.data_manager.put_data.assert_not_called()
    else:
        fn(obj, [0, 1, 2, 3], 0, 0, records())
        assert events == ["durable", "queue"]
        assert obj.data_manager.put_data.call_args.args[0]["replay_refs"] == [{"trajectory_id": "committed-reference"}]
