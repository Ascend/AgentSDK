# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2025 Alibaba-NLP.
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
#
"""Offline page crawler that writes into the SQLite page cache."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agents.webwalker_agent.cache.crawler_env import CrawlerEnv
from agents.webwalker_agent.cache.page_cache_store import normalize_cache_url
from agents.webwalker_agent.environment.webwalker_env import safe_asyncio_run
from agents.webwalker_agent.golden_path_utils import (
    extract_golden_click_paths_from_task,
    normalize_button_label,
)

logger = logging.getLogger("webwalker_crawler")


class FetchResponseContractError(RuntimeError):
    """Raised when env.get_info returns an unexpected non-network response."""


def iter_tasks(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("[skip] failed to parse JSON on line %s: %s", line_no, exc)


def task_root_url(task: dict[str, Any]) -> str:
    return str(task.get("root_url") or task.get("website") or "").strip()


def task_source_urls(task: dict[str, Any]) -> list[str]:
    info = task.get("info") if isinstance(task.get("info"), dict) else {}
    raw = info.get("source_website")
    if raw is None:
        raw = task.get("source_website")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(u).strip() for u in raw if str(u or "").strip()]


def golden_max_len(task: dict[str, Any]) -> int:
    paths = extract_golden_click_paths_from_task(task)
    return max((len(p) for p in paths), default=0)


def format_store_stats(stats: dict) -> str:
    total = int(stats.get("total", 0))
    ok = int(stats.get("ok", 0))
    failed = int(stats.get("failed", 0))
    empty = int(stats.get("empty", 0))
    rate = f"{ok / total * 100:.1f}%" if total else "n/a"
    return f"DB: total={total} ok={ok} failed={failed} empty={empty} success_rate={rate}"


def extract_buttons_list(env: CrawlerEnv, html: str, page_url: str) -> list[dict[str, str]]:
    """Extract in-domain buttons; keep multiple entries that share the same label."""
    soup = __import__("bs4").BeautifulSoup(html, "html.parser")

    links: list[dict[str, str]] = []
    for a_tag in soup.find_all("a", href=True):
        url = a_tag["href"]
        text = "".join(a_tag.stripped_strings)
        if text and "javascript" not in url and not url.endswith(
            (".jpg", ".png", ".gif", ".jpeg", ".pdf")
        ):
            full_url = env.process_url(env.root_url, url)
            if _is_in_crawl_domain(env.root_url, full_url):
                links.append({"url": full_url, "text": text})

    for a_tag in soup.find_all("a", onclick=True):
        onclick_text = a_tag["onclick"]
        text = "".join(a_tag.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text and not url.endswith((".jpg", ".png", ".gif", ".jpeg", ".pdf")):
                full_url = env.process_url(env.root_url, url)
                if _is_in_crawl_domain(env.root_url, full_url):
                    links.append({"url": full_url, "text": text})

    unique: dict[str, dict[str, str]] = {
        f"{item['url']}_{item['text']}": item for item in links
    }
    buttons = list(unique.values())
    env.button_url_dict = {}
    for item in buttons:
        env.button_url_dict[item["text"]] = item["url"]
    return buttons


def _is_in_crawl_domain(root_url: str, candidate_url: str) -> bool:
    root_host = urllib.parse.urlsplit(str(root_url or "")).hostname
    candidate_host = urllib.parse.urlsplit(str(candidate_url or "")).hostname
    if not root_host or not candidate_host:
        return False
    root_host = root_host.lower()
    candidate_host = candidate_host.lower()
    return candidate_host == root_host or candidate_host.endswith(f".{root_host}")


def _coerce_fetch_result(value: Any) -> tuple[str, str]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise FetchResponseContractError(
            f"env.get_info must return (html, markdown), got {type(value).__name__}"
        )
    html, markdown = value
    if not isinstance(html, str) or not isinstance(markdown, str):
        raise FetchResponseContractError(
            "env.get_info must return string html and markdown"
        )
    return html, markdown


def format_crawler_stats(stats: dict) -> str:
    return (
        f"current_pass: fetch_ok={stats.get('fetch_ok', 0)} fetch_fail={stats.get('fetch_fail', 0)} "
        f"cache_skip={stats.get('cache_skip', 0)} urls_touched={stats.get('urls', 0)}"
    )


class Crawler:
    def __init__(
        self,
        store,
        *,
        retries: int,
        timeout: float,
        refresh: bool,
        max_children: int,
        concurrency: int = 1,
    ):
        self.store = store
        self.retries = max(1, int(retries))
        self.timeout = float(timeout)
        self.refresh = bool(refresh)
        self.max_children = max(1, int(max_children))
        self.concurrency = max(1, int(concurrency))
        self.stats = {"fetch_ok": 0, "fetch_fail": 0, "cache_skip": 0, "urls": 0}
        self._stats_lock = threading.Lock()

    def _bump_stat(self, key: str, delta: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + delta

    def _fetch(self, env: CrawlerEnv, url: str) -> tuple[str, str] | None:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                ret = safe_asyncio_run(env.get_info(url, screenshot=False, timeout=self.timeout))
                return _coerce_fetch_result(ret)
            except FetchResponseContractError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        logger.warning(f"[fetch-fail] {url}: {last_error}")
        return None

    def crawl_url(self, env: CrawlerEnv, url: str) -> tuple[list[dict[str, str]], str]:
        norm = normalize_cache_url(url)
        if not norm:
            return [], "invalid"

        if not self.refresh:
            record = self.store.get_record(norm)
            if record is not None and record.get("status") == "ok":
                self._bump_stat("cache_skip")
                try:
                    return json.loads(record.get("buttons_json") or "[]"), "cache_ok"
                except json.JSONDecodeError:
                    return [], "cache_ok"

        try:
            fetched = self._fetch(env, url)
        except FetchResponseContractError as exc:
            logger.error("[fetch-contract-error] %s: %s", url, exc)
            self._bump_stat("fetch_fail")
            return [], "contract_error"
        if fetched is None:
            self.store.put(
                norm, "", "", status="failed", root_url=env.root_url, force=self.refresh
            )
            self._bump_stat("fetch_fail")
            return [], "failed"

        html, markdown = fetched
        buttons = extract_buttons_list(env, html, url)
        page_status = "ok" if (markdown and markdown.strip()) or buttons else "empty"
        self.store.put(
            norm,
            html,
            markdown,
            status=page_status,
            root_url=env.root_url,
            buttons=buttons,
            force=self.refresh,
        )
        self._bump_stat("fetch_ok")
        logger.info(
            f"[ok] depth-fetch {url} -> md={len(markdown or '')}chars buttons={len(buttons)}"
        )
        return buttons, page_status

    def crawl_task(
        self,
        task: dict[str, Any],
        *,
        max_depth: int,
        force_source: bool,
        max_children: int | None = None,
    ) -> None:
        root_url = task_root_url(task)
        if not root_url:
            logger.warning("[skip] task is missing root_url: id=%s", task.get("id") or task.get("task_id"))
            return

        env = CrawlerEnv(root_url)
        queue: deque[tuple[str, int]] = deque()
        queue.append((root_url, 0))
        seen_in_task: set[str] = set()
        child_limit = self.max_children if max_children is None else max(1, int(max_children))

        while queue:
            url, depth = queue.popleft()
            norm = normalize_cache_url(url)
            if not norm or norm in seen_in_task:
                continue
            seen_in_task.add(norm)

            buttons, page_status = self.crawl_url(env, url)
            if page_status != "cache_ok":
                self._bump_stat("urls")

            if depth >= max_depth:
                continue
            for button in buttons[:child_limit]:
                child = button.get("url")
                if not child:
                    continue
                child_norm = normalize_cache_url(child)
                if child_norm and child_norm not in seen_in_task:
                    queue.append((child, depth + 1))

        if force_source:
            for source_url in task_source_urls(task):
                norm = normalize_cache_url(source_url)
                if not norm:
                    continue
                _, page_status = self.crawl_url(env, source_url)
                if page_status != "cache_ok":
                    self._bump_stat("urls")

    def _buttons_for_page(self, norm_url: str) -> list[dict[str, str]]:
        record = self.store.get_record(norm_url)
        if record is not None and str(record.get("status") or "") == "ok":
            self._bump_stat("cache_skip")
            try:
                raw = json.loads(record.get("buttons_json") or "[]")
            except json.JSONDecodeError:
                return []
            if isinstance(raw, list):
                return [b for b in raw if isinstance(b, dict) and b.get("url")]
        return []

    def crawl_task_golden_path(
        self,
        task: dict[str, Any],
        *,
        force_source: bool,
    ) -> None:
        root_url = task_root_url(task)
        if not root_url:
            logger.warning("[skip] task is missing root_url: id=%s", task.get("id") or task.get("task_id"))
            return

        env = CrawlerEnv(root_url)
        paths = extract_golden_click_paths_from_task(task)
        task_id = task.get("id") or task.get("task_id") or ""

        def _ensure_page(url: str) -> list[dict[str, str]]:
            norm = normalize_cache_url(url)
            if not norm:
                return []
            cached = self._buttons_for_page(norm)
            if cached:
                return cached
            buttons, _ = self.crawl_url(env, url)
            return buttons

        for path_idx, click_path in enumerate(paths, start=1):
            current = root_url
            for step_idx, label in enumerate(click_path):
                buttons = _ensure_page(current)
                want = normalize_button_label(label)
                matches = [
                    b for b in buttons if normalize_button_label(b.get("text", "")) == want
                ]
                if not matches:
                    logger.warning(
                        f"[golden_path] task={task_id} path={path_idx} step={step_idx} "
                        f"no matching button {label!r} @ {current}"
                    )
                    break
                if len(matches) > 1:
                    logger.info(
                        f"[golden_path] duplicate label {label!r} x{len(matches)} @ {current}; pre-crawling all"
                    )
                for match in matches:
                    child = str(match.get("url") or "")
                    if child:
                        _ensure_page(child)
                current = str(matches[0].get("url") or "")
                if not current:
                    break

        if force_source:
            for source_url in task_source_urls(task):
                _ensure_page(source_url)
                self._bump_stat("urls")

    def _retry_one_url(self, norm_url: str) -> None:
        record = self.store.get_record(norm_url) or {}
        root_url = str(record.get("root_url") or norm_url).strip() or norm_url
        env = CrawlerEnv(root_url)
        self.crawl_url(env, norm_url)

    def retry_failed_urls(self) -> int:
        pending = self.store.list_urls_by_status(("failed", "empty"))
        if not pending:
            return 0
        logger.info(
            f"[retry-failed] retrying {len(pending)} failed/empty URLs with concurrency={self.concurrency}"
        )
        if self.concurrency <= 1:
            for norm_url in pending:
                self._retry_one_url(norm_url)
            return len(pending)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._retry_one_url, url) for url in pending]
            for future in as_completed(futures):
                future.result()
        return len(pending)
