# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Lazy optional capability: legacy construction does not initialize replay data."""


class ReplayStoreCapability:
    def _get_replay_store(self):
        if self._replay_closed:
            raise RuntimeError("EpisodeStore is closed")
        if self._replay_backend is None:
            self._replay_backend = self._create_replay_store()
        return self._replay_backend

    def get_replay_manifest(self):
        return self._get_replay_store().get_replay_manifest()

    def append_trajectories(self, samples, request_id):
        return self._get_replay_store().append_trajectories(samples, request_id)

    def get_trajectory(self, trajectory_id, source_store_id=None):
        return self._get_replay_store().get_trajectory(trajectory_id, source_store_id)

    def get_trajectories(self, ids, source_store_id=None):
        return self._get_replay_store().get_trajectories(ids, source_store_id)

    def query(self, query=None, **filters):
        return self._get_replay_store().query(query, **filters)

    def query_page(self, query=None, *, cursor=None, **filters):
        return self._get_replay_store().query_page(query, cursor=cursor, **filters)

    def _close_replay(self):
        if self._replay_backend is not None:
            self._replay_backend.close()
            self._replay_backend = None
        self._replay_closed = True
