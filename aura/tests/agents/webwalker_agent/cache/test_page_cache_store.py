#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest


def _ensure_aura_src_on_path():
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)


def test_normalize_cache_url_removes_fragment_and_trailing_slash():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.page_cache_store import normalize_cache_url

    assert normalize_cache_url("HTTPS://Example.COM/path/?a=1#section") == "https://example.com/path?a=1"
    assert normalize_cache_url("https://example.com/path/?b=2&a=1") == "https://example.com/path?a=1&b=2"
    assert normalize_cache_url("https://example.com") == "https://example.com/"
    assert normalize_cache_url("https://example.com/") == "https://example.com/"
    assert normalize_cache_url("example.com/path") == ""
    assert normalize_cache_url("") == ""


def test_page_cache_store_roundtrip_and_downgrade_guard():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.page_cache_store import PageCacheStore

    tmpdir = Path.cwd() / f"webwalker-cache-test-{os.getpid()}"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    db_path = tmpdir / "pages.sqlite"
    store = PageCacheStore(str(db_path))
    try:
        assert store.put(
            "https://example.com/path/#frag",
            "<html>ok</html>",
            "markdown",
            buttons=[{"text": "Next", "url": "https://example.com/next"}],
        )
        assert store.get("https://EXAMPLE.com/path") == ("<html>ok</html>", "markdown")
        assert store.has("https://example.com/path")
        assert store.stats() == {"total": 1, "ok": 1}

        record = store.get_record("https://example.com/path")
        assert json.loads(record["buttons_json"]) == [{"text": "Next", "url": "https://example.com/next"}]

        assert not store.put("https://example.com/path", "", "", status="failed")
        assert store.get_record("https://example.com/path")["status"] == "ok"

        assert store.put("https://example.com/path", "", "", status="failed", force=True)
        assert store.get("https://example.com/path") is None
        assert store.list_urls_by_status(("failed",)) == ["https://example.com/path"]
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_page_cache_store_downgrade_guard_uses_locked_read(monkeypatch):
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.page_cache_store import PageCacheStore

    tmpdir = Path.cwd() / f"webwalker-cache-test-locked-{os.getpid()}"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    db_path = tmpdir / "pages.sqlite"
    store = PageCacheStore(str(db_path))
    try:
        assert store.put("https://example.com/path", "<html>ok</html>", "markdown")

        def _unexpected_get_record(_url):
            raise AssertionError("put should not call lock-taking get_record")

        monkeypatch.setattr(store, "get_record", _unexpected_get_record)
        assert not store.put("https://example.com/path", "", "", status="failed")
        assert store._get_record_nolock("https://example.com/path")["status"] == "ok"
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_page_cache_store_close_closes_read_only_connections():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.page_cache_store import PageCacheStore

    tmpdir = Path.cwd() / f"webwalker-cache-test-readonly-{os.getpid()}"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    db_path = tmpdir / "pages.sqlite"
    writer = PageCacheStore(str(db_path))
    try:
        assert writer.put("https://example.com/path", "<html>ok</html>", "markdown")
        writer.checkpoint()
    finally:
        writer.close()

    reader = PageCacheStore(str(db_path), read_only=True)
    conn = None
    try:
        assert reader.get("https://example.com/path") == ("<html>ok</html>", "markdown")
        conn = reader._read_conn()
    finally:
        reader.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
