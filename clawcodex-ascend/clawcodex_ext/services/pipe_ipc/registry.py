#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Pipe peer registry with optional JSON persistence."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from .models import PipePeer


class PipeRegistry:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self._peers: dict[str, PipePeer] = {}
        self._data_dir = Path(data_dir) if data_dir is not None else None
        if self._data_dir is not None:
            self._load()

    def register(self, peer: PipePeer) -> None:
        with self._lock:
            self._peers[peer.instance_id] = peer
            self._persist()

    def unregister(self, instance_id: str) -> None:
        with self._lock:
            self._peers.pop(instance_id, None)
            self._persist()

    def get(self, instance_id: str) -> PipePeer | None:
        with self._lock:
            return self._peers.get(instance_id)

    def list_peers(self) -> list[PipePeer]:
        with self._lock:
            return list(self._peers.values())

    def prune_stale(self, max_age_seconds: float, *, now: float | None = None) -> list[str]:
        cutoff = (time.time() if now is None else now) - max_age_seconds
        removed: list[str] = []
        with self._lock:
            for instance_id, peer in list(self._peers.items()):
                if peer.last_seen < cutoff:
                    removed.append(instance_id)
                    del self._peers[instance_id]
            if removed:
                self._persist()
        return removed

    @property
    def peer_count(self) -> int:
        with self._lock:
            return len(self._peers)

    def _registry_path(self) -> Path | None:
        if self._data_dir is None:
            return None
        return self._data_dir / "peers.json"

    def _load(self) -> None:
        path = self._registry_path()
        if path is None or not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid PipeRegistry peers.json") from exc

        if not isinstance(data, list):
            raise TypeError("PipeRegistry peers.json must contain a list")
        self._peers = {peer.instance_id: peer for peer in (PipePeer.from_dict(item) for item in data)}

    def _persist(self) -> None:
        path = self._registry_path()
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        # PID is not unique within a process when many threads call
        # _persist() concurrently, so use a per-call uuid to avoid races on
        # the tmp file path. The caller is expected to hold self._lock so the
        # snapshot of self._peers is consistent with the rename.
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(
                json.dumps([peer.to_dict() for peer in self._peers.values()], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except BaseException:
            # Never leave a stray tmp file behind if the rename failed.
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise
