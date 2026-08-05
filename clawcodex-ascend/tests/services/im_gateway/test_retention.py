#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for gateway persistence retention ownership."""

from __future__ import annotations

# This split test branch is linted without the separately submitted gateway implementation.
# pylint: disable=no-name-in-module

import asyncio
import json
import time

import pytest
from clawcodex_ext.services.im_gateway.config import ReliabilityConfig
from clawcodex_ext.services.im_gateway.retention import run_retention_sweep
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def _write_ndjson(path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_ndjson(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_dead_letter_rotates_on_max_bytes(tmp_path) -> None:
    cfg = ReliabilityConfig(dead_letter_max_bytes=100, dead_letter_backup_count=3)
    store = ReliabilityStore(tmp_path, reliability=cfg)
    path = tmp_path / "dead_letter.ndjson"
    _write_ndjson(path, [{"k": "x" * 200}])

    store.append_dead_letter({"k": "new"})

    assert path.exists()
    assert (tmp_path / "dead_letter.ndjson.1").exists()
    assert _read_ndjson(path) == [{"k": "new"}]


def test_audit_rotates_on_max_bytes(tmp_path) -> None:
    cfg = ReliabilityConfig(audit_max_bytes=100, audit_backup_count=3)
    store = ReliabilityStore(tmp_path, reliability=cfg)
    path = tmp_path / "audit.ndjson"
    _write_ndjson(path, [{"event_type": "old", "payload": "x" * 200}])

    store.audit("test_event", payload="new")

    assert path.exists()
    assert (tmp_path / "audit.ndjson.1").exists()


def test_purge_processed_inbound_uses_seen_at(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    now = time.time()
    _write_ndjson(
        tmp_path / "processed_inbound.ndjson",
        [
            {"key": "old", "seen_at": now - 8 * 86400},
            {"key": "new", "seen_at": now - 100},
        ],
    )

    removed = store.purge_processed_inbound(ttl_seconds=7 * 86400, max_entries=10000)

    assert removed == 1
    assert _read_ndjson(tmp_path / "processed_inbound.ndjson") == [{"key": "new", "seen_at": pytest.approx(now - 100)}]


def test_purge_outbox_uses_at_with_timestamp_fallback(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    now = time.time()
    _write_ndjson(
        tmp_path / "outbox.ndjson",
        [
            {"id": "old-at", "at": now - 31 * 86400},
            {"id": "new-at", "at": now - 100},
            {"id": "old-timestamp", "timestamp": now - 31 * 86400},
            {"id": "new-timestamp", "timestamp": now - 100},
            {"id": "legacy-no-time"},
        ],
    )

    removed = store.purge_outbox(ttl_seconds=30 * 86400, max_entries=50000)

    assert removed == 2
    assert {entry["id"] for entry in _read_ndjson(tmp_path / "outbox.ndjson")} == {
        "new-at",
        "new-timestamp",
        "legacy-no-time",
    }


def test_purge_unsupported_inbound_uses_received_at(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    now = time.time()
    _write_ndjson(
        tmp_path / "unsupported_inbound.ndjson",
        [
            {"id": "old", "received_at": now - 8 * 86400},
            {"id": "new", "received_at": now - 100},
            {"id": "legacy-no-time"},
        ],
    )

    removed = store.purge_unsupported_inbound(ttl_seconds=7 * 86400, max_entries=10000)

    assert removed == 1
    assert {entry["id"] for entry in _read_ndjson(tmp_path / "unsupported_inbound.ndjson")} == {
        "new",
        "legacy-no-time",
    }


def test_purge_unsupported_inbound_caps_legacy_without_timestamps(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    _write_ndjson(
        tmp_path / "unsupported_inbound.ndjson",
        [{"id": f"legacy-{idx}"} for idx in range(5)],
    )

    removed = store.purge_unsupported_inbound(ttl_seconds=7 * 86400, max_entries=2)

    assert removed == 3
    assert [entry["id"] for entry in _read_ndjson(tmp_path / "unsupported_inbound.ndjson")] == [
        "legacy-3",
        "legacy-4",
    ]


def test_purge_all_only_manages_bounded_append_files(tmp_path) -> None:
    cfg = ReliabilityConfig()
    store = ReliabilityStore(tmp_path, reliability=cfg)
    now = time.time()
    _write_ndjson(
        tmp_path / "processed_inbound.ndjson",
        [{"key": "old", "seen_at": now - 8 * 86400}],
    )
    _write_ndjson(tmp_path / "outbox.ndjson", [{"id": "old", "at": now - 31 * 86400}])
    _write_ndjson(
        tmp_path / "unsupported_inbound.ndjson",
        [{"id": "old", "received_at": now - 8 * 86400}],
    )
    _write_json(tmp_path / "im_session_map.json", {"legacy": {"session_id": "old"}})
    _write_json(tmp_path / "wechat_context_tokens.json", {"acct:user": "ctx"})
    _write_json(tmp_path / "feishu_last_senders.json", {"feishu": "ou_user"})
    _write_json(tmp_path / "wechat_accounts.json", {"default": {"get_updates_buf": "cursor"}})
    _write_json(tmp_path / "wechat" / "wechat_pairing.json", {"codes": [{"code": "abc"}]})

    result = store.purge_all(cfg)

    assert result == {
        "processed_inbound.ndjson": 1,
        "outbox.ndjson": 1,
        "unsupported_inbound.ndjson": 1,
    }
    assert _read_json(tmp_path / "im_session_map.json") == {"legacy": {"session_id": "old"}}
    assert _read_json(tmp_path / "wechat_context_tokens.json") == {"acct:user": "ctx"}
    assert _read_json(tmp_path / "feishu_last_senders.json") == {"feishu": "ou_user"}
    assert _read_json(tmp_path / "wechat_accounts.json") == {"default": {"get_updates_buf": "cursor"}}
    assert _read_json(tmp_path / "wechat" / "wechat_pairing.json") == {"codes": [{"code": "abc"}]}


def test_disabled_returns_empty_and_does_not_modify(tmp_path) -> None:
    cfg = ReliabilityConfig(retention_enabled=False)
    store = ReliabilityStore(tmp_path, reliability=cfg)
    _write_ndjson(tmp_path / "processed_inbound.ndjson", [{"key": "k", "seen_at": 0}])

    result = run_retention_sweep(store, cfg)

    assert result == {}
    assert _read_ndjson(tmp_path / "processed_inbound.ndjson") == [{"key": "k", "seen_at": 0}]


def test_run_retention_sweep_enabled(tmp_path) -> None:
    cfg = ReliabilityConfig()
    store = ReliabilityStore(tmp_path, reliability=cfg)
    now = time.time()
    _write_ndjson(
        tmp_path / "processed_inbound.ndjson",
        [{"key": "old", "seen_at": now - 8 * 86400}],
    )

    result = run_retention_sweep(store, cfg)

    assert result.get("processed_inbound.ndjson") == 1


def test_run_retention_sweep_exception_returns_empty(tmp_path, monkeypatch) -> None:
    store = ReliabilityStore(tmp_path)

    def boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "purge_all", boom)

    assert run_retention_sweep(store, ReliabilityConfig()) == {}


def test_corrupted_ndjson_file_does_not_crash_purge(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    path = tmp_path / "processed_inbound.ndjson"
    path.write_text(
        '{"key": "ok", "seen_at": ' + str(time.time()) + "}\n"
        "not json\n"
        '{"key": "ok2", "seen_at": ' + str(time.time()) + "}\n",
        encoding="utf-8",
    )

    assert store.purge_processed_inbound(ttl_seconds=7 * 86400, max_entries=100) == 0


def test_retention_loop_executes_and_cancels(tmp_path, monkeypatch) -> None:
    import extensions.im_gateway.server as server_mod

    store = ReliabilityStore(tmp_path)
    call_count = 0

    def fake_sweep(actual_store, reliability) -> None:
        nonlocal call_count
        assert actual_store is store
        call_count += 1

    monkeypatch.setattr(server_mod, "run_retention_sweep", fake_sweep)

    async def run() -> None:
        task = asyncio.create_task(server_mod._retention_loop(store, 0.01, ReliabilityConfig()))
        await asyncio.sleep(0.05)
        assert call_count >= 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_retention_loop_swallows_exception(tmp_path, monkeypatch) -> None:
    import extensions.im_gateway.server as server_mod

    store = ReliabilityStore(tmp_path)
    call_count = 0

    def flaky_sweep(actual_store, reliability) -> None:
        nonlocal call_count
        assert actual_store is store
        call_count += 1
        raise RuntimeError("flaky")

    monkeypatch.setattr(server_mod, "run_retention_sweep", flaky_sweep)

    async def run() -> None:
        task = asyncio.create_task(server_mod._retention_loop(store, 0.01, ReliabilityConfig()))
        await asyncio.sleep(0.05)
        assert call_count >= 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
