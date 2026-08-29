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

"""Focused regression tests for persistence review fixes."""

from __future__ import annotations

import json
import os
import stat
import time

import pytest

from clawcodex_ext.services.im_gateway.config import ReliabilityConfig
from clawcodex_ext.services.im_gateway.retention import run_retention_sweep
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def _write_lines(path, entries) -> None:
    path.write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in entries),
        encoding="utf-8",
    )


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_retention_sweep_reuses_the_gateway_store(tmp_path, monkeypatch) -> None:
    config = ReliabilityConfig()
    store = ReliabilityStore(tmp_path, reliability=config)
    calls = []

    def purge(actual_config):
        calls.append(actual_config)
        return {"processed_inbound.ndjson": 1}

    monkeypatch.setattr(store, "purge_all", purge)

    assert run_retention_sweep(store, config) == {"processed_inbound.ndjson": 1}
    assert calls == [config]


def test_processed_inbound_skips_invalid_json_values(tmp_path) -> None:
    now = time.time()
    path = tmp_path / "processed_inbound.ndjson"
    _write_lines(
        path,
        [
            None,
            [],
            {"key": "missing-time"},
            {"key": "wrong-time", "seen_at": "yesterday"},
            {"key": "valid", "seen_at": now},
        ],
    )

    store = ReliabilityStore(tmp_path)

    assert store.is_duplicate("valid")
    assert not store.is_duplicate("missing-time")
    assert store.purge_processed_inbound(ttl_seconds=3600, max_entries=10) == 0
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"key": "valid", "seen_at": pytest.approx(now)}
    ]


@pytest.mark.parametrize("invalid_root", [None, [], "text", 1])
def test_json_state_rejects_non_object_roots(tmp_path, invalid_root) -> None:
    store = ReliabilityStore(tmp_path)
    path = tmp_path / "wechat_context_tokens.json"
    path.write_text(json.dumps(invalid_root), encoding="utf-8")

    assert store.get_context_token("account", "user") is None
    store.set_context_token("account", "user", "token")
    assert json.loads(path.read_text(encoding="utf-8")) == {"account:user": "token"}


def test_state_files_remain_private_after_atomic_replace(tmp_path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permissions are not available on Windows")

    store = ReliabilityStore(tmp_path)
    store.set_context_token("account", "user", "first")
    path = tmp_path / "wechat_context_tokens.json"
    path.chmod(0o644)

    store.set_context_token("account", "user", "second")
    store.check_and_record("message")

    assert _mode(tmp_path) == 0o700
    assert _mode(path) == 0o600
    assert _mode(tmp_path / "processed_inbound.ndjson") == 0o600
