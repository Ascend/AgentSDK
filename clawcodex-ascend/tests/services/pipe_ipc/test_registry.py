#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=no-name-in-module

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from clawcodex_ext.services.pipe_ipc import PipePeer, PipeRegistry


def make_peer(instance_id: str, *, last_seen: float = 100.0) -> PipePeer:
    return PipePeer(instance_id=instance_id, hostname="host", pid=123, last_seen=last_seen)


def test_register_get_and_unregister() -> None:
    registry = PipeRegistry()
    peer = make_peer("peer-1")

    registry.register(peer)
    assert registry.get("peer-1") is peer
    assert registry.peer_count == 1

    registry.unregister("peer-1")
    assert registry.get("peer-1") is None
    assert registry.peer_count == 0


def test_persistence_round_trip(tmp_path: Path) -> None:
    registry = PipeRegistry(tmp_path)
    registry.register(make_peer("peer-1"))
    registry.register(make_peer("peer-2"))

    restored = PipeRegistry(tmp_path)

    assert {peer.instance_id for peer in restored.list_peers()} == {"peer-1", "peer-2"}


def test_prune_stale_peers(tmp_path: Path) -> None:
    registry = PipeRegistry(tmp_path)
    registry.register(make_peer("old", last_seen=10.0))
    registry.register(make_peer("fresh", last_seen=95.0))

    removed = registry.prune_stale(max_age_seconds=50.0, now=100.0)

    assert removed == ["old"]
    assert registry.get("old") is None
    assert registry.get("fresh") is not None
    assert PipeRegistry(tmp_path).get("old") is None


def test_invalid_persistence_json_raises(tmp_path: Path) -> None:
    (tmp_path / "peers.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="peers.json"):
        PipeRegistry(tmp_path)


def test_concurrent_writes_persist_all_peers(tmp_path: Path) -> None:
    registry = PipeRegistry(tmp_path)

    def writer(i: int) -> None:
        registry.register(PipePeer(instance_id=f"peer-{i}", hostname="h", pid=1234 + i))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    restored = PipeRegistry(tmp_path)
    assert restored.peer_count == 32
    assert {peer.instance_id for peer in restored.list_peers()} == {f"peer-{i}" for i in range(32)}
    # Tmp files should be cleaned up after rename, never leaving stale .tmp.
    assert not list(tmp_path.glob(".peers.json.*.tmp"))
