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
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import List, Dict

import aiohttp
import ray
import torch

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.async_http import async_send_batch
from agentic_rl.controllers.utils.http_status import HTTP_OK_200
from agentic_rl.controllers.utils.sync_http import sync_send
from agentic_rl.controllers.utils.utils import DEFAULT_URL_METHOD
from agentic_rl.controllers.utils.utils import (
    post_with_url,
    DEFAULT_RETRY_COUNT,
    DEFAULT_BACKOFF_FACTOR,
    MIN_BACKOFF_FACTOR
)


@dataclass
class _ExportTracker:
    iter_idx: int
    start_ts: float  # wall-clock
    expected: int  # how many shards
    seen: int = 0  # shards written so far


@ray.remote
class DispatchActor:
    def __init__(
            self,
            n_samples_per_prompt,
            validate_num_samples,
            global_batch_size,
            train_iters,
            data_loader,
            initialize_train_dataloader,
            consumed_train_samples,
            data_optimized,
            rollout_server_addr,
    ):
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
        self.logger = Loggers(__name__).get_logger()
        self.failed_groups = deque()

        self.rollout_unit_ready = False
        self.session = None
        self.send_lock = Lock()
        self.max_group_retries = DEFAULT_RETRY_COUNT

        self.current_training_iter = 1

        self.n_samples_per_prompt = n_samples_per_prompt
        # self.actor_config = actor_config
        self.global_batch_size = global_batch_size
        self.total_prompts_for_rollout_unit = 0
        self.total_prompts_for_training = self.global_batch_size * train_iters

        self.train_data_iters, _, _ = initialize_train_dataloader(
            data_loader, validate_num_samples, consumed_train_samples)
        self.data_optimized = data_optimized
        self.rollout_server_addr = rollout_server_addr

    async def _unlock_rollout_unit_remote(self, retry: int = DEFAULT_RETRY_COUNT,
                                          backoff: float = DEFAULT_BACKOFF_FACTOR):
        self.logger.info(f">>> enable rollout unit remote")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/unlock"
        # post_with_url is sync; run in thread so we don't block the event loop
        return await asyncio.to_thread(post_with_url, url, retry, backoff)

    # Start the rollout unit to receive the request
    async def unlock_rollout_unit(self, ):
        return await self._unlock_rollout_unit_remote()

    async def _lock_rollout_unit_remote(self, retry: int = DEFAULT_RETRY_COUNT,
                                        backoff: float = DEFAULT_BACKOFF_FACTOR):
        self.logger.info(f">>> blocking rollout Unit remote")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/lock"
        return await asyncio.to_thread(post_with_url, url, retry, backoff)

    # Lock the rollout unit and do not accept requests
    async def lock_rollout_unit(self):
        return await self._lock_rollout_unit_remote()

    async def _shutdown_rollout_unit_remote(self, retry: int = DEFAULT_RETRY_COUNT,
                                            backoff: float = DEFAULT_BACKOFF_FACTOR):
        self.logger.info(f">>> shutdown rollout Unit remote")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/shutdown"
        return await asyncio.to_thread(post_with_url, url, retry, backoff)

    # stop rollout unit
    async def shutdown_rollout_unit(self):
        return await self._shutdown_rollout_unit_remote()

    async def _create_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    def _get_from_failed_groups(self):
        return self.failed_groups.pop()

    def _get_from_data_iters(self):
        batch = None
        try:
            batch = next(self.train_data_iters)
        except StopIteration:
            self.logger.warning(f">>> rollout data iter end")
        return batch

    def _get_batch_groups_remote(self, n_groups):
        groups: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_groups):
            failed_num = len(self.failed_groups)
            batch = self._get_from_failed_groups() if failed_num > 0 else self._get_from_data_iters()
            if batch is None:
                break
            groups.append(batch)
        return groups

    async def _do_batch_groups_send_remote(self, groups):
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/send_batch"
        tasks = [async_send_batch(g, url, session=self.session) for g in groups]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _stat_succeed(self, groups, results):
        success = []
        for g, res in zip(groups, results):
            # Send successfully - placed in the success queue
            status = int((res or {}).get("status") or 0)
            if status == HTTP_OK_200:
                success.append(g)
                continue
            # Failed items are placed back into the queue.
            self.failed_groups.append(g)
        return len(success)

    async def _send_batch_groups_remote(self, n_groups):
        groups = self._get_batch_groups_remote(n_groups)
        self.logger.info(f"|perf-stat|train| get batch groups: {len(groups)}")
        if len(groups) <= 0:
            return 0

        await self._create_session()
        results = await self._do_batch_groups_send_remote(groups)
        succeed = self._stat_succeed(groups, results)
        self.logger.info(f"|perf-stat|train| send batch succeed: {succeed}, failed: {len(self.failed_groups)}")
        return succeed

    async def send_batch_groups(self, n_groups):
        return await self._send_batch_groups_remote(n_groups)

    def check_stop_batch(self) -> bool:
        return self.total_prompts_for_rollout_unit >= self.total_prompts_for_training

    async def shutdown(self):
        await self.shutdown_rollout_unit()

    def set_rollout_unit_ready(self):
        self.rollout_unit_ready = True

    def check_rollout_unit_ready(self):
        return self.rollout_unit_ready

    def init_done(self):
        pass

    def set_current_training_iter(self, iteration: int):
        self.current_training_iter = iteration

    async def _notify_weights_update_remote(self, weight_save_dir: str):
        self.logger.info(f"all shards for {weight_save_dir} are written, notifying rollout unit remote")
        tasks = []
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/notify_weights_update"
        tasks.append(asyncio.to_thread(sync_send, weight_save_dir, url,
                                       DEFAULT_RETRY_COUNT, MIN_BACKOFF_FACTOR))
        await asyncio.gather(*tasks)

    async def notify_weights_update(self, weight_save_dir: str):
        await self._notify_weights_update_remote(weight_save_dir)
