# !/usr/bin/env python3
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

import asyncio
import io
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import aiohttp
import torch

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.http_status import HTTP_OK_200
from agentic_rl.controllers.utils.utils import DEFAULT_RETRY_COUNT, MIN_BACKOFF_FACTOR

logger = Loggers(__name__).get_logger()

_executor = ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2))


# light-weight pickler run in a *separate process*
def _dumps(batch_dict: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(batch_dict, buf, pickle_protocol=5, _use_new_zipfile_serialization=False)
    return buf.getvalue()


# async post that streams the bytes
async def client_post(session, url: str, body: bytes, timeout_s: int = 120):
    data = aiohttp.FormData()
    data.add_field("file", body, filename="batch.pt", content_type="application/octet-stream")
    try:
        async with session.post(url, data=data, timeout=timeout_s) as r:
            return {"status": r.status}  # no JSON parsing
    except Exception as e:
        return {"status": 0, "error": repr(e)}  # network/timeout


async def async_send_batch(
        batch,
        url: str,
        retry: int = DEFAULT_RETRY_COUNT,
        backoff: float = MIN_BACKOFF_FACTOR,
        session: aiohttp.ClientSession | None = None
):
    loop = asyncio.get_running_loop()
    owned = False
    if session is None:
        session = aiohttp.ClientSession()
        owned = True
    try:
        logger.info(f"async send batch to {url}")
        for attempt in range(1, retry + 1):
            body = await loop.run_in_executor(_executor, _dumps, batch)
            res = await client_post(session, url, body)
            s = int(res.get("status") or 0)
            if s == HTTP_OK_200:
                logger.info(f"async send batch succeed, res: {res}")
                return res
            elif attempt < retry:
                logger.error(f"async send batch failed, server returned {s}, retrying...")
                await asyncio.sleep(backoff)
                continue
            return res  # permanent failure
    except Exception as e:
        logger.error(f"async send batch failed, {e}")
        return None
    finally:
        if owned:
            await session.close()
