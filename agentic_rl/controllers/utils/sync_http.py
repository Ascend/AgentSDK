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

import time

import requests

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.utils import MIN_RETRY_COUNT, DEFAULT_BACKOFF_FACTOR

logger = Loggers(__name__).get_logger()


def sync_send(address: str, url: str, retry: int = MIN_RETRY_COUNT, backoff: float = DEFAULT_BACKOFF_FACTOR):
    headers = {"Content-Type": "text/plain"}
    for attempt in range(1, retry + 1):
        try:
            response = requests.post(url, data=address, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retry:
                # log and back off
                logger.warning(
                    f"Attempt {attempt}/{retry} failed sending to {url}: {e!r}. "
                    f"Waiting {backoff}s before retry…"
                )
                time.sleep(backoff)
            else:
                # all retries exhausted
                logger.error(f"All {retry} attempts failed for {url}.")
    return None
