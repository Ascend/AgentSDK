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

import sys
import types
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
    sys.modules.setdefault(
        "torch",
        types.SimpleNamespace(distributed=types.SimpleNamespace(is_initialized=lambda: False)),
    )
    sys.modules.setdefault(
        "torch.distributed",
        types.SimpleNamespace(
            is_initialized=lambda: False,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
        ),
    )


def test_crawler_task_helpers_parse_urls_and_golden_depth():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.crawler import golden_max_len, task_root_url, task_source_urls

    task = {
        "website": "https://example.com",
        "info": {
            "source_website": "https://example.com/final",
            "golden_path": ["root -> Products -> Pricing"],
        },
    }

    assert task_root_url(task) == "https://example.com"
    assert task_source_urls(task) == ["https://example.com/final"]
    assert golden_max_len(task) == 2


def test_crawler_stats_formatters_are_stable():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.crawler import format_crawler_stats, format_store_stats

    assert format_store_stats({"total": 4, "ok": 3, "failed": 1, "empty": 0}) == (
        "DB: total=4 ok=3 failed=1 empty=0 success_rate=75.0%"
    )
    assert format_crawler_stats({"fetch_ok": 2, "fetch_fail": 1, "cache_skip": 3, "urls": 6}) == (
        "current_pass: fetch_ok=2 fetch_fail=1 cache_skip=3 urls_touched=6"
    )


def test_crawl_domain_check_rejects_prefix_spoof_domain():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.crawler import _is_in_crawl_domain

    assert _is_in_crawl_domain("https://example.com", "https://example.com/products")
    assert _is_in_crawl_domain("https://example.com", "https://docs.example.com/guide")
    assert not _is_in_crawl_domain("https://example.com", "https://example.com.evil.org/phish")


def test_fetch_contract_error_is_not_retried(monkeypatch):
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache import crawler as crawler_module
    from agents.webwalker_agent.cache.crawler import Crawler, FetchResponseContractError

    calls = 0

    def fake_run(_):
        nonlocal calls
        calls += 1
        return "not-a-pair"

    monkeypatch.setattr(crawler_module, "safe_asyncio_run", fake_run)
    crawler = Crawler(types.SimpleNamespace(), retries=3, timeout=1, refresh=False, max_children=3)
    env = types.SimpleNamespace(get_info=lambda *_, **__: object())

    with pytest.raises(FetchResponseContractError):
        crawler._fetch(env, "https://example.com")

    assert calls == 1


def test_crawl_task_local_max_children_does_not_mutate_shared_limit():
    _ensure_aura_src_on_path()
    from agents.webwalker_agent.cache.crawler import Crawler

    crawler = Crawler(types.SimpleNamespace(), retries=1, timeout=1, refresh=False, max_children=60)
    crawled_urls = []

    def fake_crawl_url(_env, url):
        crawled_urls.append(url)
        if url == "https://example.com":
            return [
                {"url": f"https://example.com/{idx}", "text": str(idx)}
                for idx in range(20)
            ], "ok"
        return [], "ok"

    crawler.crawl_url = fake_crawl_url

    crawler.crawl_task(
        {"root_url": "https://example.com"},
        max_depth=1,
        visited_global=set(),
        visited_lock=None,
        force_source=False,
        max_children=12,
    )

    assert crawler.max_children == 60
    assert len(crawled_urls) == 13
