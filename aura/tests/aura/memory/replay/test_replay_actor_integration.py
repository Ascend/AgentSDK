"""Opt-in real CPU Ray tests; keep separate from legacy tests that mock sys.modules.

RUN_REPLAY_RAY_TESTS=1 PYTHONPATH=aura python -m pytest <this file> -q
Never enable on a memory-constrained host. No NPU or training job is started.
"""

import os
import tempfile
from dataclasses import replace

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REPLAY_RAY_TESTS") != "1",
                                reason="Real Ray integration is opt-in and requires sufficient memory")


@pytest.fixture(scope="module")
def real_ray():
    ray = pytest.importorskip("ray")
    # Keep Linux AF_UNIX sockets below 107 bytes even when RAY_TMPDIR is long.
    temp_dir = tempfile.mkdtemp(prefix="i158ray-", dir="/tmp" if os.name != "nt" else None)
    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
             _temp_dir=temp_dir,
             object_store_memory=80 * 1024 * 1024, _memory=256 * 1024 * 1024,
             runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    try:
        assert ray.cluster_resources().get("NPU", 0) == 0
        print(f"CPU-only Ray runtime: {temp_dir}; resources={ray.cluster_resources()}")
        yield ray
    finally:
        ray.shutdown()  # Only this test's local runtime, never a global `ray stop`.


@pytest.mark.parametrize("backend", ["sqlite", "json"])
def test_real_episode_actor_commit_retry_conflict_and_restart(real_ray, backend, tmp_path, make_group):
    from aura.memory.episode.episode import Episode
    ray = real_ray
    config = {"backend": backend, "path": str(tmp_path / "episodes"), "store_id": "store"}
    actor = Episode.remote("replay-service", replay_store_config=config)
    reopened = reader = None
    try:
        samples = make_group()
        receipts = ray.get([actor.append_replay_batch.remote(samples, "request") for _ in range(2)])
        assert sorted(r.inserted_count for r in receipts) == [0, 2]
        assert receipts[0].sample_refs == receipts[1].sample_refs
        assert ray.get(actor.get_replay_manifest.remote())["store_id"] == "store"
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.append_replay_batch.remote([replace(s, reward=0.1) for s in samples], "request"))
        assert len(ray.get(actor.query_replay.remote(reward_min=0.8, limit=100))) == 2
        stored = ray.get(actor.get_trajectory.remote(samples[0].trajectory_id))
        assert stored.old_log_probs.data == samples[0].old_log_probs.data and stored.created_seq == 1
        assert ray.get(actor.to_dict.remote())["trajectories"] == []
        ray.get(actor.close_replay_store.remote())
        ray.kill(actor, no_restart=True)
        actor = None
        reopened = Episode.remote("reopened-service", replay_store_config=config)
        retry = ray.get(reopened.append_replay_batch.remote(samples, "request"))
        assert retry.replayed and retry.inserted_count == 0 and retry.sample_refs == receipts[0].sample_refs
        page = ray.get(reopened.query_replay_page.remote(limit=1))
        assert page.snapshot_seq == 2 and page.next_cursor
        assert len(ray.get(reopened.query_replay_page.remote(limit=1, cursor=page.next_cursor)).samples) == 1
        # Abrupt actor death after acknowledgement must not lose committed bytes.
        crash_samples = make_group(group_id="crash-committed")
        crash_receipt = ray.get(reopened.append_replay_batch.remote(crash_samples, "before-kill"))
        ray.kill(reopened, no_restart=True)  # Deliberately do not close the store.
        reopened = Episode.remote("after-crash", replay_store_config=config)
        crash_retry = ray.get(reopened.append_replay_batch.remote(crash_samples, "before-kill"))
        assert crash_retry.replayed and crash_retry.sample_refs == crash_receipt.sample_refs
        assert len(ray.get(reopened.query_replay.remote())) == 4
        reader = Episode.remote("readonly-service", replay_store_config={**config, "read_only": True})
        assert len(ray.get(reader.query_replay.remote())) == 4
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(reader.append_replay_batch.remote(samples, "forbidden"))
    finally:
        for target in (actor, reopened, reader):
            if target is not None:
                try:
                    ray.get(target.close_replay_store.remote(), timeout=10)
                finally:
                    ray.kill(target, no_restart=True)


def test_real_legacy_episode_remains_usable_without_a_store(real_ray):
    from aura.memory.episode.episode import Episode
    ray = real_ray
    actor = Episode.remote("old-client")
    try:
        ray.get(actor.set_task.remote({"question": "legacy"}))
        assert ray.get(actor.to_dict.remote())["task"] == {"question": "legacy"}
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.get_replay_manifest.remote())
    finally:
        ray.kill(actor, no_restart=True)


@pytest.mark.parametrize("backend", ["sqlite", "json"])
@pytest.mark.parametrize("ratio", [0.3, 1.0])
def test_p2_real_actor_publish_load_and_mixed_grpo_batch(real_ray, tmp_path, backend, ratio):
    from collections import Counter
    from unittest.mock import Mock
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.config import ReplayConfig
    from aura.memory.replay.runtime import ReplaySession, persist_trajectories
    from aura.memory.replay.mixing import ReplayUnavailableError
    from test_replay_conversion import config, records

    settings = config(tmp_path, replay_ratio=ratio, batch_groups=10, backend=backend,
                      warmup_online_steps=1 if ratio == 1 else 0)
    actor = Episode.remote("p2", replay_store_config={"backend": backend, "path": settings.path, "store_id": "store"})
    session = ReplaySession(settings, 0, actor)
    calls = []
    def source():
        number = len(calls)
        calls.append(number)
        refs = persist_trajectories(records(10, prefix=f"request-{number}"), settings, actor, f"batch-{number}", 3)
        return {"replay_refs": refs}
    try:
        first = session.next_batch(source, 3)
        assert len(first["prompts"]) == 20
        assert session.mixer.window == {}
        session.commit_success(3)
        mixed = session.next_batch(source, 4)
        plan = session.mixer.inflight
        assert plan.metrics["replay/actual_ratio"] == ratio
        assert len(calls) == (1 if ratio == 1 else 2)
        assert len(set(mixed["replay_trajectory_ids"])) == 20
        assert sorted(Counter(mixed["prompt_ids"]).values()) == [2]*10
        assert mixed["rollout_log_probs"].dtype == __import__("torch").float32

        # Build the actual verl DataProto through the production trainer method,
        # without initializing model workers or executing an NPU operation.
        import asyncio
        import contextlib
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        from verl import DataProto
        from test_replay_backpressure import method
        trainer = SimpleNamespace(
            timing_raw={}, _get_samples_from_queue=lambda: (0, mixed), _replay_session=session,
            config=SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="grpo"),
                data=SimpleNamespace(train_batch_size=10),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2, temperature=1.0))))
        generate = method("aura/trainer/train_adapter/verl/full_async/full_async_trainer.py",
            "FullyAsyncTrainer", "_fit_generate", DataProto=DataProto, torch=torch, np=np,
            logger=Mock(), marked_timer=lambda *args, **kwargs: contextlib.nullcontext())
        trainer._extract_generate_tensors = method(
            "aura/trainer/train_adapter/verl/full_async/full_async_trainer.py",
            "FullyAsyncTrainer", "_extract_generate_tensors")
        proto = asyncio.run(generate(trainer))
        assert proto.batch["input_ids"].shape == proto.batch["attention_mask"].shape
        assert proto.meta_info["trajectory_param_versions"] == mixed["replay_weight_versions"]
        assert sorted(Counter(proto.non_tensor_batch["uid"]).values()) == [2]*10
        session.commit_success(4)
        if ratio == 1:
            with pytest.raises(ReplayUnavailableError):
                session.next_batch(source, 11)
            assert len(calls) == 1
        # Forged acknowledgement references must never become training data.
        raw = source()
        raw["replay_refs"][0]["checksum"] = "f"*64
        with pytest.raises(ValueError):
            session.load_online(raw)
    finally:
        real_ray.get(actor.close_replay_store.remote())
        real_ray.kill(actor, no_restart=True)


@pytest.mark.parametrize("strategy", ["priority", "reward_weighted"])
def test_p3_real_actor_weighted_groups_native_advantages_and_probability_metadata(real_ray, tmp_path, strategy):
    import numpy as np
    import torch
    from collections import Counter
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.runtime import ReplaySession, persist_trajectories
    from aura.memory.replay.mixing import group_key
    from test_replay_conversion import config, records

    settings = config(tmp_path, replay_strategy=strategy, replay_ratio=.5)
    actor = Episode.remote("p3-weighted", replay_store_config={"backend": "sqlite", "path": settings.path,
                                                            "store_id": settings.store_id})
    session = ReplaySession(settings, 0, actor)
    calls = []
    def source():
        i = len(calls)
        calls.append(i)
        raw = records(prefix=f"batch-{i}/")
        raw[2]["trajectory_reward"] = .2
        refs = persist_trajectories(raw, settings, actor, f"batch-{i}", 3)
        return {"replay_refs": refs}
    def native(raw):
        tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
        proto = DataProto.from_dict(tensors=tensors, non_tensors={"uid": np.array(raw["prompt_ids"])})
        adv, _ = compute_grpo_outcome_advantage(proto.batch["token_level_rewards"], proto.batch["response_mask"],
            proto.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=False)
        proto.batch["advantages"] = adv
        return proto
    try:
        warm = session.next_batch(source, 3)
        first_keys = tuple(group_key(g) for g in session.mixer.inflight.groups)
        updates = session.priority_updates(native(warm))
        assert session.mixer.priorities == {}
        session.commit_success(3, updates)
        if strategy == "priority":
            assert session.mixer.priorities[first_keys[0]] > 7 * session.mixer.priorities[first_keys[1]]
        mixed = session.next_batch(source, 4)
        plan = session.mixer.inflight
        assert len(plan.replay_groups) == len(plan.online_groups) == 1
        assert sorted(Counter(mixed["prompt_ids"]).values()) == [2, 2]
        assert 0 < plan.draw_probabilities[0] < 1
        before = mixed["rollout_log_probs"].clone()
        session.commit_success(4, session.priority_updates(native(mixed)))
        assert torch.equal(before, mixed["rollout_log_probs"])
        for identity in mixed["replay_trajectory_ids"]:
            stored = real_ray.get(actor.get_trajectory.remote(identity))
            assert stored is not None and stored.weight_version == 3 and stored.group_complete
        assert len(calls) == 2
    finally:
        real_ray.get(actor.close_replay_store.remote())
        real_ray.kill(actor, no_restart=True)


def test_p2_launcher_wires_one_owned_actor_and_zero_ratio_is_bypass(real_ray, tmp_path):
    from omegaconf import OmegaConf
    from aura.memory.replay.launcher import prepare_replay
    from dataclasses import asdict
    from test_replay_conversion import config
    settings = asdict(config(tmp_path))
    train = OmegaConf.create({"replay": settings, "trainer": {"resume_mode": "disable"},
        "algorithm": {"adv_estimator": "grpo"}, "data": {"train_batch_size": 2}, "rollout": {"n": 2},
        "actor_rollout_ref": {"rollout": {"n": 2}, "actor": {"use_rollout_log_probs": True}}})
    rollout = OmegaConf.create({"n_samples_per_prompt": 2, "global_batch_size": 2,
        "hybrid_batch_num": 1, "enable_version_control": True})
    actor = prepare_replay(train, rollout)
    try:
        assert real_ray.get(actor.get_replay_manifest.remote())["store_id"] == "store"
        assert train.replay.actor_name == rollout.replay.actor_name
        assert not rollout.enable_version_control
        disabled = OmegaConf.create({"replay": {"replay_ratio": 0, "path": "/unused"}})
        assert prepare_replay(disabled, object()) is None
        for changes in ({"replay_ratio": 1, "warmup_online_steps": 0}, {"replay_strategy": "unknown"}):
            other = OmegaConf.create(OmegaConf.to_container(train))
            other.replay.actor_name = ""
            for key, value in changes.items():
                other.replay[key] = value
            with pytest.raises(ValueError):
                prepare_replay(other, rollout)
    finally:
        real_ray.get(actor.close_replay_store.remote())
        real_ray.kill(actor, no_restart=True)
