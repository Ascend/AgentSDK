# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Replay settings: disabled means no storage, actors, sampler or RNG."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

from .schema import COMPATIBILITY_FIELDS, identifier, integer


def replay_ratio(config):
    if not isinstance(config, Mapping):
        return 0.0
    value = config.get("replay", {})
    if value is None:
        return 0.0
    if not isinstance(value, Mapping):
        raise ValueError("replay must be a mapping")
    ratio = value.get("replay_ratio", 0.0)
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("replay_ratio must be a finite number in [0,1]")
    return float(ratio)


@dataclass(frozen=True)
class ReplayConfig:
    replay_ratio: float
    replay_buffer_size: int
    group_size: int
    batch_groups: int
    max_lag: int
    seed: int
    store_id: str
    source_experiment_id: str
    source_run_id: str
    policy_lineage_id: str
    model_signature: str
    tokenizer_signature: str
    task_signature: str
    reward_signature: str
    backend: str = "sqlite"
    path: str = ""
    actor_name: str = ""
    actor_namespace: str = "aura-episode-replay"
    replay_strategy: str = "uniform"
    warmup_online_steps: int = 0
    rpc_timeout: float = 120.0
    priority_alpha: float = 1.0
    priority_epsilon: float = 1e-6
    reward_temperature: float = 1.0
    online_prefetch: bool = True
    reuse_sources: tuple = ()

    def __post_init__(self):
        ratio = replay_ratio({"replay": {"replay_ratio": self.replay_ratio}})
        if ratio == 0:
            raise ValueError("Do not construct replay components when replay_ratio=0")
        if type(self.online_prefetch) is not bool:
            raise ValueError("online_prefetch must be boolean")
        for key in ("group_size", "batch_groups", "max_lag", "replay_buffer_size"):
            integer(getattr(self, key), key, 1)
        if self.group_size > 256 or self.replay_buffer_size < self.group_size:
            raise ValueError("Replay capacity must fit at least one bounded complete group")
        integer(self.seed, "seed")
        integer(self.warmup_online_steps, "warmup_online_steps")
        if self.replay_strategy not in ("uniform", "priority", "reward_weighted"):
            raise ValueError("Unknown replay strategy")
        for name in ("priority_alpha", "priority_epsilon", "reward_temperature"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(name + " must be finite")
            if value < 0 or (name != "priority_alpha" and value == 0):
                raise ValueError(name + " is outside its allowed range")
        if self.backend not in ("sqlite", "json") or not Path(self.path).is_absolute():
            raise ValueError("Replay requires sqlite/json and an absolute persistent path")
        if type(self.rpc_timeout) not in (int, float) or not math.isfinite(self.rpc_timeout) or self.rpc_timeout <= 0:
            raise ValueError("rpc_timeout must be positive and finite")
        for key in ("store_id", "source_experiment_id", "source_run_id", "policy_lineage_id",
                    "model_signature", "tokenizer_signature", "task_signature", "reward_signature", "actor_namespace"):
            identifier(getattr(self, key), key)
        if self.actor_name:
            identifier(self.actor_name, "actor_name")
        from .reuse import normalize_sources
        object.__setattr__(self, 'reuse_sources', normalize_sources(self.reuse_sources, self))
        object.__setattr__(self, "_compatibility_key", tuple(
            getattr(self, "store_id" if k == "source_store_id" else k) for k in COMPATIBILITY_FIELDS))

    @classmethod
    def from_mapping(cls, value, *, group_size=None, batch_groups=None):
        options = dict(value)
        if group_size is not None:
            if "group_size" in options and options["group_size"] != group_size:
                raise ValueError("Replay group_size differs from rollout.n")
            options["group_size"] = group_size
        if batch_groups is not None:
            if "batch_groups" in options and options["batch_groups"] != batch_groups:
                raise ValueError("Replay batch_groups differs from train_batch_size")
            options["batch_groups"] = batch_groups
        return cls(**options)
