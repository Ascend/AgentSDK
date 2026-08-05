#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Regression tests for IM gateway persistence validation and permissions."""

from __future__ import annotations

# This split test branch is linted without the separately submitted gateway implementation.
# pylint: disable=no-name-in-module

import json
import os
import stat
import time

import pytest

from clawcodex_ext.services.im_gateway.config import ReliabilityConfig
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _capture_store_warnings(monkeypatch) -> list[str]:
    from clawcodex_ext.services.im_gateway import store as store_mod

    warnings: list[str] = []

    def capture(message, *args, **kwargs) -> None:
        warnings.append(message % args)

    monkeypatch.setattr(store_mod.logger, "warning", capture)
    return warnings


def test_processed_inbound_skips_invalid_lines_on_load_and_purge(tmp_path, monkeypatch) -> None:
    now = time.time()
    path = tmp_path / "processed_inbound.ndjson"
    entries = [
        None,
        [],
        {"key": "missing-time"},
        {"key": "wrong-time", "seen_at": "yesterday"},
        {"key": ["wrong-key"], "seen_at": now},
        {"key": "old", "seen_at": now - 100},
        {"key": "new", "seen_at": now},
    ]
    path.write_text("".join(f"{json.dumps(entry)}\n" for entry in entries), encoding="utf-8")
    warnings = _capture_store_warnings(monkeypatch)

    store = ReliabilityStore(
        tmp_path,
        ReliabilityConfig(inbound_dedupe_ttl_seconds=10),
    )

    assert store.is_duplicate("new") is True
    assert store.is_duplicate("missing-time") is False
    assert store.is_duplicate("wrong-time") is False
    assert store.purge_processed_inbound(ttl_seconds=10, max_entries=100) == 1
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"key": "new", "seen_at": pytest.approx(now)}
    ]
    warning_text = "\n".join(warnings)
    assert "skipped invalid NDJSON lines" in warning_text
    assert "non_object=2" in warning_text
    assert "schema=3" in warning_text


@pytest.mark.parametrize("value", [None, [], "scalar", 42])
@pytest.mark.parametrize(
    ("filename", "read_value", "expected"),
    [
        (
            "wechat_context_tokens.json",
            lambda store: store.get_context_token("acct", "user"),
            None,
        ),
        (
            "feishu_last_senders.json",
            lambda store: store.get_feishu_last_sender("feishu"),
            None,
        ),
        (
            "wechat_accounts.json",
            lambda store: store.get_wechat_cursor("default"),
            "",
        ),
    ],
)
def test_json_state_rejects_non_object_roots(
    tmp_path,
    monkeypatch,
    filename,
    read_value,
    expected,
    value,
) -> None:
    store = ReliabilityStore(tmp_path)
    _write_json(tmp_path / filename, value)
    warnings = _capture_store_warnings(monkeypatch)

    assert read_value(store) == expected
    assert any(f"expected JSON object file={filename}" in warning for warning in warnings)


def test_json_state_skips_entries_with_invalid_value_schema(tmp_path, monkeypatch) -> None:
    store = ReliabilityStore(tmp_path)
    _write_json(tmp_path / "wechat_context_tokens.json", {"acct:user": ["secret"]})
    _write_json(tmp_path / "feishu_last_senders.json", {"feishu": {"sender": "ou_bad"}})
    _write_json(
        tmp_path / "wechat_accounts.json",
        {"default": {"get_updates_buf": 123, "updated_at": "now"}},
    )
    warnings = _capture_store_warnings(monkeypatch)

    assert store.get_context_token("acct", "user") is None
    assert store.get_feishu_last_sender("feishu") is None
    assert store.get_wechat_cursor("default") == ""
    warning_text = "\n".join(warnings)
    assert "skipped invalid string-map entries" in warning_text
    assert "skipped invalid account entries" in warning_text


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_store_keeps_directory_files_and_atomic_temps_private(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway import store as store_mod

    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o755)
    os.chmod(state_dir, 0o755)
    context_path = state_dir / "wechat_context_tokens.json"
    _write_json(context_path, {"acct:user": "existing-token"})
    os.chmod(context_path, 0o644)

    store = ReliabilityStore(state_dir)
    assert _mode(state_dir) == 0o700
    assert store.get_context_token("acct", "user") == "existing-token"
    assert _mode(context_path) == 0o600

    temporary_modes: list[int] = []
    real_replace = store_mod.os.replace

    def checked_replace(source, destination) -> None:
        temporary_modes.append(_mode(source))
        real_replace(source, destination)

    monkeypatch.setattr(store_mod.os, "replace", checked_replace)

    store.set_context_token("acct", "user", "replacement-token")
    store.set_feishu_last_sender("feishu", "ou_user")
    store.set_wechat_cursor("default", "cursor")
    store.record_processed("dedupe-key")
    store.append_outbox({"idempotency_key": "outbox-key", "status": "pending"})
    store.append_outbox({"idempotency_key": "expired", "status": "pending", "at": 0})
    assert store.purge_outbox(ttl_seconds=1, max_entries=100) == 1
    store.record_unsupported_media({"message_id": "unsupported"})
    store.audit("permission_test")
    store.append_dead_letter({"idempotency_key": "dead-key"})

    assert temporary_modes and set(temporary_modes) == {0o600}
    for filename in (
        "wechat_context_tokens.json",
        "feishu_last_senders.json",
        "wechat_accounts.json",
        "processed_inbound.ndjson",
        "outbox.ndjson",
        "unsupported_inbound.ndjson",
        "audit.ndjson",
        "dead_letter.ndjson",
    ):
        assert _mode(state_dir / filename) == 0o600


def test_processed_purge_rewrites_invalid_lines_when_valid_entries_survive(tmp_path) -> None:
    now = time.time()
    path = tmp_path / "processed_inbound.ndjson"
    path.write_text(
        "null\n" + json.dumps({"key": "valid", "seen_at": now}) + "\n",
        encoding="utf-8",
    )

    store = ReliabilityStore(tmp_path)

    assert store.purge_processed_inbound(ttl_seconds=3600, max_entries=10) == 0
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"key": "valid", "seen_at": pytest.approx(now)}
    ]


def test_processed_purge_tolerates_invalid_utf8(tmp_path, monkeypatch) -> None:
    path = tmp_path / "processed_inbound.ndjson"
    invalid_content = b'\xff{"key":"unreadable"}\n'
    path.write_bytes(invalid_content)
    warnings = _capture_store_warnings(monkeypatch)
    store = ReliabilityStore(tmp_path)

    assert store.purge_processed_inbound(ttl_seconds=3600, max_entries=10) == 0
    assert path.read_bytes() == invalid_content
    assert any("unreadable NDJSON file=processed_inbound.ndjson" in item for item in warnings)
