#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------
import asyncio
import os
import threading


def safe_asyncio_run(coro):
    result_holder = []
    exc_holder = []

    def _run():
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            result_holder.append(result)
        except Exception as e:
            exc_holder.append(e)
        finally:
            if loop is not None:
                loop.close()

    t = threading.Thread(target=_run)
    t.start()
    t.join()

    if exc_holder:
        raise exc_holder[0]

    if not result_holder:
        raise RuntimeError("safe_asyncio_run did not return any result")

    return result_holder[0]


# Use the default asyncio event loop policy because Playwright is sensitive to uvloop under Ray.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Limit in-process crawler concurrency to reduce Crawl4AI/Playwright race failures.
_crawl_concurrency_env = os.getenv("WEBWALKER_CRAWL_CONCURRENCY", "2")
try:
    _crawl_concurrency = max(1, int(_crawl_concurrency_env))
except (TypeError, ValueError):
    _crawl_concurrency = 2
crawl_lock = threading.BoundedSemaphore(_crawl_concurrency)
