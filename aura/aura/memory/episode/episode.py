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

from enum import Enum
from typing import Any

import ray

from aura.runner.agent_engine_wrapper.base.agent.base_agent import Trajectory


class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"


@ray.remote
class Episode:
    def __init__(self, episode_id: str = None, replay_store_config: dict = None):
        self.episode_id = episode_id
        self.task: Any = None
        self.termination_reason = None
        # self.is_correct: bool = False
        self.trajectories: list[tuple[str, Trajectory]] = []
        self._replay_store = None
        self._replay_ledger = None
        self._replay_store_config = None
        self._replay_sources = None
        if replay_store_config is not None:
            self.configure_replay_store(replay_store_config)

    def configure_replay_store(self, config):
        """Open connections inside the actor, never send a live store over Ray."""
        if self._replay_store is not None:
            raise RuntimeError("Replay store already configured; close it explicitly first")
        if not isinstance(config, dict) or not {"backend", "path"} <= set(config):
            raise ValueError("Replay store config needs backend and path")
        allowed = {"backend", "path", "store_id", "read_only", "max_batch_size", "max_batch_bytes"}
        if set(config) - allowed:
            raise ValueError("Unknown replay store configuration fields")
        options = {k: v for k, v in config.items() if k not in ("backend", "path")}
        if config["backend"] == "sqlite":
            from aura.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
            store = SQLiteEpisodeStore(config["path"], **options)
        elif config["backend"] == "json":
            from aura.memory.episode.backend.json_episode_store import JsonEpisodeStore
            store = JsonEpisodeStore(config["path"], **options)
        else:
            raise ValueError("Unsupported replay backend")
        try:
            manifest = store.get_replay_manifest()
        except BaseException:
            store.close()
            raise
        self._replay_store = store
        self._replay_store_config = dict(config)
        return manifest

    def _require_replay_store(self):
        if self._replay_store is None:
            raise RuntimeError("Episode has no replay store configured")
        return self._replay_store

    def get_replay_manifest(self):
        return self._require_replay_store().get_replay_manifest()

    def append_replay_batch(self, samples, request_id):
        """Return only after durable batch commit; do not retain payloads on Episode."""
        return self._require_replay_store().append_trajectories(samples, request_id)

    def get_trajectory(self, trajectory_id, source_store_id=None, group_id=None):
        if (self._replay_sources is not None and source_store_id is not None
                and source_store_id != self._require_replay_store().get_replay_manifest()['store_id']):
            return self._replay_sources.load(trajectory_id, source_store_id, group_id)
        if group_id is not None:
            raise ValueError('Group lookup is only supported for mounted foreign sources')
        return self._require_replay_store().get_trajectory(trajectory_id, source_store_id)

    def get_replay_batch(self, ids, source_store_id=None):
        """One bounded actor RPC; preserve request ordering and missing entries."""
        return self._require_replay_store().get_trajectories(ids, source_store_id)

    def query_replay(self, query=None, **filters):
        return self._require_replay_store().query(query, **filters)

    def query_replay_page(self, query=None, *, cursor=None, **filters):
        return self._require_replay_store().query_page(query, cursor=cursor, **filters)

    def configure_replay_sources(self, settings):
        """Mount frozen foreign stores read-only; never migrate or write them."""
        from pathlib import Path
        from aura.memory.replay.config import ReplayConfig
        from aura.memory.replay.reuse import ReadOnlySources
        from aura.memory.replay.state import _contract
        config = ReplayConfig.from_mapping(settings)
        store = self._require_replay_store()
        if (store.get_replay_manifest()['read_only'] or config.store_id != store.get_replay_manifest()['store_id']
                or config.backend != self._replay_store_config['backend']
                or Path(config.path).resolve() != Path(self._replay_store_config['path']).resolve()):
            raise ValueError('Reuse requires this actor to own the destination writer')
        if self._replay_sources is not None:
            if _contract(config) != _contract(self._replay_sources.config):
                raise RuntimeError('Cannot change mounted reuse contract')
            return self._replay_sources.verify()
        readers = ReadOnlySources(config)
        self._replay_sources = readers
        return readers.verify()

    def get_reuse_group(self, source_store_id, group_id):
        from aura.memory.replay.reuse import source_for
        if self._replay_sources is None:
            raise RuntimeError('No read-only reuse sources mounted')
        source = source_for(self._replay_sources.config, source_store_id)
        if source is None or group_id not in source.group_ids:
            raise ValueError('Foreign group not authorized by the reuse contract')
        return self._replay_sources.load(None, source_store_id, group_id)

    def verify_replay_sources(self):
        if self._replay_sources is None:
            return {'sources': 0, 'unchanged': True}
        return self._replay_sources.verify()

    def configure_replay_ledger(self, path, settings):
        """Run-private control DB; immutable trajectory backend stays unchanged."""
        from pathlib import Path
        from aura.memory.replay.config import ReplayConfig
        from aura.memory.replay.ledger import ReplayLedger
        from aura.memory.replay.state import _contract
        store=self._require_replay_store()
        config=ReplayConfig.from_mapping(settings)
        if (config.store_id!=store.get_replay_manifest()['store_id']
                or config.backend!=self._replay_store_config['backend']
                or Path(config.path).resolve()!=Path(self._replay_store_config['path']).resolve()):
            raise ValueError('Ledger/source settings do not match this Episode actor')
        if self._replay_ledger is not None:
            if (Path(path).resolve()!=self._replay_ledger_path
                    or _contract(config)!=self._replay_ledger.contract):
                raise RuntimeError('Episode already owns a different ledger')
        else:
            if config.reuse_sources:
                self.configure_replay_sources(settings)
            self._replay_ledger=ReplayLedger(path,config,self.get_trajectory)
            self._replay_ledger_path=Path(path).resolve()
        return {'contract_digest':self._replay_ledger.contract_digest}

    def replay_ledger_call(self, operation, args=(), kwargs=None):
        """Bounded explicit protocol surface; never expose arbitrary attributes."""
        allowed={'acquire_writer','initial_branch','enqueue','lease','ack',
                 'prepare_commit','lookup_commit','seal_commit','publish_marker','recover_branch','inspect',
                 'seen_online'}
        if operation not in allowed or self._replay_ledger is None:
            raise ValueError('Unknown protocol operation / ledger not configured')
        return getattr(self._replay_ledger,operation)(*args,**(kwargs or {}))

    def close_replay_store(self):
        if self._replay_ledger is not None:
            self._replay_ledger.close()
            self._replay_ledger = None
        if self._replay_sources is not None:
            self._replay_sources.close()
            self._replay_sources = None
        if self._replay_store is not None:
            self._replay_store.close()
            self._replay_store = None

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "termination_reason": self.termination_reason,
            # "is_correct": bool(self.is_correct),
            "trajectories": [(agent_name, trajectory.to_dict()) for agent_name, trajectory in self.trajectories],
        }

    def set_task(self, task: Any):
        self.task = task

    def set_termination_reason(self, termination_reason):
        self.termination_reason = termination_reason

    def add_trajectory(self, agent_name: str, trajectory: Trajectory):
        self.trajectories.append((agent_name, trajectory))

    def get_trajectory_by_agent_name(self, agent_name: str):
        return [trajectory for name, trajectory in self.trajectories if name == agent_name]

    def remove_trajectory_by_agent_name(self, agent_name: str):
        self.trajectories = [trajectory for name, trajectory in self.trajectories if name != agent_name]
