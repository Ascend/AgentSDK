# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Fail-fast launcher wiring; no side effects at replay_ratio=0."""
from dataclasses import asdict
from pathlib import Path
import uuid

from .config import ReplayConfig, replay_ratio


def prepare_replay(train_config, rollout_config):
    if replay_ratio(train_config) == 0:
        return None
    from aura.trainer.train_adapter.verl.full_async.replay_recovery import recovery_settings
    recovery_settings(train_config)  # Validate explicit opt-in before creating actors.
    if train_config.trainer.resume_mode != "disable":
        raise ValueError("Replay trainer resume adapters are not enabled; P4 requires explicit model/source adapters")
    estimator = train_config.algorithm.adv_estimator
    if str(getattr(estimator, "value", estimator)).lower() != "grpo":
        raise ValueError("Replay P2 requires GRPO")
    n, m = train_config.actor_rollout_ref.rollout.n, train_config.data.train_batch_size
    if (rollout_config.n_samples_per_prompt != n or rollout_config.global_batch_size != m
            or rollout_config.hybrid_batch_num != 1 or train_config.rollout.n != n):
        raise ValueError("Replay P2 requires matching group/batch sizes and hybrid_batch_num=1")
    if not train_config.actor_rollout_ref.actor.use_rollout_log_probs:
        raise ValueError("Replay requires preserved behavior rollout log probabilities")
    from aura.trainer.train_adapter.verl.full_async.replay_loss import validate_loss_config
    validate_loss_config(train_config)
    settings = ReplayConfig.from_mapping(train_config.replay, group_size=n, batch_groups=m)
    if settings.replay_ratio == 1 and ((settings.warmup_online_steps == 0 and not settings.reuse_sources)
                                      or settings.replay_buffer_size < m*n):
        raise ValueError("P2 pure replay requires explicit online warmup and a window fitting one full batch")
    if settings.actor_name:
        raise ValueError("Launcher owns the replay actor; actor_name must not be provided")
    if settings.reuse_sources:
        from .reuse import ReadOnlySources, seed_groups
        from .mixing import compatible
        readers = ReadOnlySources(settings)  # Fail before creating any destination writer.
        try:
            if (settings.replay_ratio == 1 and settings.warmup_online_steps == 0
                    and sum(compatible(g, settings, 0) for g in seed_groups(settings, readers.load)) < m):
                raise ValueError('Cold pure reuse needs one full compatible, non-future seed batch at version zero')
        finally:
            readers.close()
    # Only launch-time owner creates a writer, never detached/reused actors.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from aura.memory.episode.episode import Episode
    options = asdict(settings)
    options["actor_name"] = "episode-replay-" + uuid.uuid4().hex
    Path(settings.path).parent.mkdir(parents=True, exist_ok=True)
    actor = Episode.options(name=options["actor_name"], namespace=settings.actor_namespace,
                            num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node_id=ray.get_runtime_context().get_node_id(), soft=False)).remote(
        options["actor_name"], replay_store_config={
            "backend": settings.backend, "path": settings.path, "store_id": settings.store_id})
    try:
        ray.get(actor.get_replay_manifest.remote(), timeout=settings.rpc_timeout)
        if settings.reuse_sources:
            ray.get(actor.configure_replay_sources.remote(asdict(settings)), timeout=settings.rpc_timeout)
    except BaseException:
        ray.kill(actor, no_restart=True)
        raise
    from omegaconf import open_dict
    with open_dict(train_config):
        train_config.replay = options
    with open_dict(rollout_config):
        rollout_config.replay = options
        # Equality against generation-count-derived versions is not valid once
        # replay decouples training steps from generation credits.
        rollout_config.enable_version_control = False
    return actor
