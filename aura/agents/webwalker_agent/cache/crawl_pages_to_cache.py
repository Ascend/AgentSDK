# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
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
#
"""Offline crawler that pre-fills the WebWalker page cache (SQLite).

Usage::

    python agents/webwalker_agent/cache/crawl_pages_to_cache.py \\
        --jsonl /path/to/tasks.jsonl \\
        --db agents/webwalker_agent/cache/webwalker_pages.sqlite \\
        --strategy golden_plus --max-depth 3 --max-children 60
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.webwalker_agent.cache._bootstrap import default_db_path, ensure_repo_on_path

ensure_repo_on_path()

from agents.webwalker_agent.cache.crawler import (  # noqa: E402
    Crawler,
    format_crawler_stats,
    format_store_stats,
    golden_max_len,
    iter_tasks,
)
from agents.webwalker_agent.cache.page_cache_store import get_page_cache_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("webwalker_crawler")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-crawl WebWalker pages into a SQLite cache.")
    parser.add_argument("--jsonl", required=True, help="Task jsonl with root_url / info.golden_path / source_website")
    parser.add_argument("--db", default=default_db_path(), help="Output SQLite cache path")
    parser.add_argument(
        "--strategy",
        choices=["golden_plus", "golden_only", "golden_path", "bfs_full"],
        default="golden_plus",
    )
    parser.add_argument("--max-depth", type=int, default=-1, help="Max BFS depth; -1 infers from golden path length")
    parser.add_argument("--max-children", type=int, default=60, help="Max buttons expanded per page")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--refresh", action="store_true", help="Force recrawling existing URLs, including ok records")
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Repeat full-task crawling; retry failed/empty URLs after each pass to reduce transient network gaps",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only crawl the first N tasks for debugging; 0 means all")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=max(1, int(os.getenv("WEBWALKER_CRAWL_CONCURRENCY", "8"))),
        help="Pre-crawl concurrency for tasks and retries; defaults to WEBWALKER_CRAWL_CONCURRENCY or 8",
    )
    args = parser.parse_args()

    store = get_page_cache_store(args.db, read_only=False)
    if store is None:
        logger.error("Cannot open or create cache database: %s", args.db)
        sys.exit(1)

    concurrency = max(1, int(args.concurrency))
    crawler = Crawler(
        store,
        retries=args.retries,
        timeout=args.timeout,
        refresh=args.refresh,
        max_children=args.max_children,
        concurrency=concurrency,
    )

    visited_global: set[str] = set()
    visited_lock = threading.Lock() if concurrency > 1 else None
    tasks = list(iter_tasks(args.jsonl))
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]
    passes = max(1, int(args.passes))
    logger.info(
        f"[start] strategy={args.strategy} tasks={len(tasks)} passes={passes} "
        f"concurrency={concurrency} db={args.db}"
    )
    if store.stats().get("total", 0):
        logger.info("[start] existing cache | %s", format_store_stats(store.stats()))
    logger.info(
        "[hint] monitor from another terminal: python agents/webwalker_agent/cache/cache_stats.py "
        f"--db {args.db}"
    )

    def _run_one_task(task: dict) -> None:
        if args.strategy == "golden_path":
            crawler.crawl_task_golden_path(
                task,
                visited_global=visited_global,
                visited_lock=visited_lock,
                force_source=True,
            )
        elif args.strategy == "golden_only":
            depth = args.max_depth if args.max_depth >= 0 else max(golden_max_len(task), 1)
            crawler.crawl_task(
                task,
                max_depth=depth,
                visited_global=visited_global,
                visited_lock=visited_lock,
                force_source=True,
                max_children=min(args.max_children, 12),
            )
        elif args.strategy == "bfs_full":
            depth = args.max_depth if args.max_depth >= 0 else max(golden_max_len(task) + 1, 3)
            crawler.crawl_task(
                task,
                max_depth=depth,
                visited_global=visited_global,
                visited_lock=visited_lock,
                force_source=True,
            )
        else:
            depth = args.max_depth if args.max_depth >= 0 else max(golden_max_len(task), 2)
            crawler.crawl_task(
                task,
                max_depth=depth,
                visited_global=visited_global,
                visited_lock=visited_lock,
                force_source=True,
            )

    for pass_idx in range(1, passes + 1):
        logger.info(
            f"[pass {pass_idx}/{passes}] start | {format_store_stats(store.stats())}"
        )
        if concurrency <= 1:
            for index, task in enumerate(tasks, start=1):
                _run_one_task(task)
                if index % 5 == 0 or index == len(tasks):
                    logger.info(
                        f"[pass {pass_idx}] progress {index}/{len(tasks)} | "
                        f"{format_crawler_stats(crawler.stats)} | {format_store_stats(store.stats())}"
                    )
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_run_one_task, task): task for task in tasks}
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed % 5 == 0 or completed == len(tasks):
                        logger.info(
                            f"[pass {pass_idx}] progress {completed}/{len(tasks)} tasks | "
                            f"{format_crawler_stats(crawler.stats)} | "
                            f"{format_store_stats(store.stats())}"
                        )

        retried = crawler.retry_failed_urls()
        logger.info(
            f"[pass {pass_idx}/{passes}] done | retried failed/empty={retried} | "
            f"{format_crawler_stats(crawler.stats)} | {format_store_stats(store.stats())}"
        )

    store.checkpoint()
    logger.info("=" * 60)
    logger.info(f"[done] {format_crawler_stats(crawler.stats)}")
    logger.info(f"[done] {format_store_stats(store.stats())}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
