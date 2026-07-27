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
"""Print WebWalker page-cache statistics (ok / failed / empty).

Run while crawl_pages_to_cache.py is running (read-only) or after it finishes.

Example:

    python agents/webwalker_agent/cache/cache_stats.py
    python agents/webwalker_agent/cache/cache_stats.py --db agents/webwalker_agent/cache/webwalker_pages.sqlite
    python agents/webwalker_agent/cache/cache_stats.py --list-failed
"""

from __future__ import annotations

import argparse
import os
import sys

from agents.webwalker_agent.cache._bootstrap import default_db_path, ensure_repo_on_path

ensure_repo_on_path()


def main() -> None:
    from aura.base.log.loggers import Loggers
    from agents.webwalker_agent.cache import get_page_cache_store

    logger = Loggers(__name__).get_logger()
    parser = argparse.ArgumentParser(description="Show WebWalker SQLite page cache stats.")
    parser.add_argument(
        "--db",
        default=os.getenv("WEBWALKER_PAGE_CACHE_PATH", default_db_path()),
        help="SQLite cache path",
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="List URLs with status failed or empty",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max failed URLs to print")
    args = parser.parse_args()

    store = get_page_cache_store(args.db, read_only=True)
    if store is None:
        logger.error("[ERROR] cannot open cache: %s", args.db)
        sys.exit(1)

    stats = store.stats()
    total = int(stats.get("total", 0))
    ok = int(stats.get("ok", 0))
    failed = int(stats.get("failed", 0))
    empty = int(stats.get("empty", 0))
    other = total - ok - failed - empty

    logger.info("=" * 60)
    logger.info("cache db: %s", os.path.abspath(args.db))
    logger.info("-" * 60)
    logger.info("  total URLs in DB : %s", total)
    logger.info("  ok   (usable)    : %s", ok)
    logger.info("  failed           : %s", failed)
    logger.info("  empty            : %s", empty)
    if other:
        logger.info("  other statuses   : %s", other)
    if total:
        logger.info("  success rate     : %.1f%%", ok / total * 100)
    logger.info("=" * 60)

    if args.list_failed and (failed or empty):
        pending = store.list_urls_by_status(("failed", "empty"))
        logger.info("failed/empty URLs (showing up to %s of %s):", args.limit, len(pending))
        for url in pending[: max(1, args.limit)]:
            rec = store.get_record(url) or {}
            logger.info("  [%s] %s", rec.get("status", "?"), url)


if __name__ == "__main__":
    main()
