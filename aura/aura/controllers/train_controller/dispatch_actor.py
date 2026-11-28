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
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import List, Dict

import aiohttp
import ray
import torch

from aura.base.log.loggers import Loggers
from aura.controllers.utils.async_http import async_send_batch
from aura.controllers.utils.http_status import HTTP_OK_200
from aura.controllers.utils.sync_http import sync_send
from aura.controllers.utils.utils import DEFAULT_URL_METHOD
from aura.controllers.utils.utils import post_with_url, DEFAULT_RETRY_COUNT, DEFAULT_BACKOFF_FACTOR, MIN_BACKOFF_FACTOR


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
            data_loader, validate_num_samples, consumed_train_samples
        )
        self.data_optimized = data_optimized
        self.rollout_server_addr = rollout_server_addr

        self.data_iter_end = False
        self._recovery_paused = False
        self._active_sends = 0
        self._prompt_cursor = None
        self._cursor_factory = lambda: initialize_train_dataloader(data_loader, validate_num_samples, consumed_train_samples)[0]
        self._cursor_config = (data_loader, validate_num_samples, consumed_train_samples)

    def enable_replay_cursor(self):
        if self._prompt_cursor is not None:
            return
        from aura.memory.replay.source import PromptCursor
        from aura.memory.replay.codec import digest
        from omegaconf import OmegaConf
        data, validate, consumed = self._cursor_config
        identity_config = OmegaConf.to_container(data, resolve=True) if OmegaConf.is_config(data) else vars(data)
        self._prompt_cursor = PromptCursor(self._cursor_factory,
            digest({'data': identity_config, 'validate': validate, 'initial_consumed': consumed}), self.train_data_iters)

    async def pause_replay_source(self):
        if self._prompt_cursor is None:
            raise RuntimeError('Explicit recovery cursor not enabled')
        self._recovery_paused = True
        while self._active_sends:
            await asyncio.sleep(0.01)
        if self.failed_groups:
            raise RuntimeError('Ambiguous failed generation requests cannot be checkpointed')
        return self._prompt_cursor.state()

    def restore_replay_cursor(self, state):
        if not self._recovery_paused or self._active_sends or self.failed_groups:
            raise RuntimeError('Prompt restore requires a paused empty dispatch')
        self._prompt_cursor.restore(state)
        self.train_data_iters = self._prompt_cursor.iterator
        self.data_iter_end = self._prompt_cursor.exhausted
        return self._prompt_cursor.state()

    def resume_replay_source(self):
        if not self._recovery_paused:
            raise RuntimeError('Dispatch was not paused')
        self._recovery_paused = False

    def data_iter_finished(self):
        return self.data_iter_end

    async def _unlock_rollout_unit_remote(
        self, retry: int = DEFAULT_RETRY_COUNT, backoff: float = DEFAULT_BACKOFF_FACTOR
    ):
        self.logger.info(">>> enable rollout unit remote")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/unlock"
        # post_with_url is sync; run in thread so we don't block the event loop
        return await asyncio.to_thread(post_with_url, url, retry, backoff)

    # Start the rollout unit to receive the request
    async def unlock_rollout_unit(
        self,
    ):
        return await self._unlock_rollout_unit_remote()

    async def _lock_rollout_unit_remote(
        self, retry: int = DEFAULT_RETRY_COUNT, backoff: float = DEFAULT_BACKOFF_FACTOR
    ):
        self.logger.info(">>> blocking rollout Unit remote")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/lock"
        return await asyncio.to_thread(post_with_url, url, retry, backoff)

    # Lock the rollout unit and do not accept requests
    async def lock_rollout_unit(self):
        return await self._lock_rollout_unit_remote()

    async def _shutdown_rollout_unit_remote(
        self, retry: int = DEFAULT_RETRY_COUNT, backoff: float = DEFAULT_BACKOFF_FACTOR
    ):
        self.logger.info(">>> shutdown rollout Unit remote")
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
            batch = next(self._prompt_cursor if self._prompt_cursor is not None else self.train_data_iters)
        except StopIteration:
            self.logger.warning(">>> rollout data iter end")
            self.data_iter_end = True
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
        if self._recovery_paused:
            raise RuntimeError('Generation is paused for a paired checkpoint')
        self._active_sends += 1
        try:
            return await self._send_batch_groups_remote(n_groups)
        finally:
            self._active_sends -= 1

    def check_stop_batch(self) -> bool:
        return self.total_prompts_for_rollout_unit >= self.total_prompts_for_training

    async def shutdown(self):
        await self.shutdown_rollout_unit()

    async def rollout_already_quit(self):
        self.logger.info(">>> judge rollout unit is already quit")
        url = f"{DEFAULT_URL_METHOD}://{self.rollout_server_addr}/rollout/is_quit"
        return await asyncio.to_thread(post_with_url, url, DEFAULT_RETRY_COUNT, DEFAULT_BACKOFF_FACTOR)

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
        tasks.append(asyncio.to_thread(sync_send, weight_save_dir, url, DEFAULT_RETRY_COUNT, MIN_BACKOFF_FACTOR))
        await asyncio.gather(*tasks)

    async def notify_weights_update(self, weight_save_dir: str):
        await self._notify_weights_update_remote(weight_save_dir)
