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
from typing import Any

import ray

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


SAMPLE_QUEUE_NUM_CPUS = 1
SAMPLE_QUEUE_MAX_CONCURRENCY = 20


@ray.remote(num_cpus=SAMPLE_QUEUE_NUM_CPUS, max_concurrency=SAMPLE_QUEUE_MAX_CONCURRENCY)
class SampleQueue:
    """Asynchronous streaming queue for fully_async mode.

    Producer (RolloutWorker) puts one sample (a sub-batch of trajectories) at a
    time via ``put_sample``; consumer (FullyAsyncTrainer) pulls samples one at a
    time via ``get_sample`` until enough are gathered for a mini-batch.

    A ``None`` sample returned by ``get_sample`` signals shutdown so the trainer
    training loop can terminate cleanly.
    """

    def __init__(self, max_queue_size: int = 256):
        self.max_queue_size = int(max_queue_size)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.running = True
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0
        logger.info(f"[async-queue] initialized with max_queue_size={self.max_queue_size}")

    async def put_sample(self, sample: Any) -> bool:
        """Put a single sample into the queue.

        If queue is full, remove the oldest sample (queue head, lowest weight
        version) and append the new one, so the rollouter is never blocked by
        backpressure and the freshest samples are always retained.
        """
        if not self.running:
            logger.warning("[async-queue] put_sample after shutdown, dropping sample")
            return False
        if self.queue.full():
            try:
                dropped = self.queue.get_nowait()
                self.dropped_samples += 1
                logger.warning(
                    f"[async-queue] queue full, dropped oldest sample "
                    f"(weight_version={getattr(dropped, 'weight_version', '?')}), "
                    f"dropped_total={self.dropped_samples}"
                )
            except Exception:
                logger.error("[async-queue] error while dropping oldest sample", exc_info=True)
        await self.queue.put(sample)
        self.total_produced += 1
        if self.total_produced % 32 == 0:
            logger.info(f"[async-queue] milestone: produced={self.total_produced}, consumed={self.total_consumed}, "
                        f"dropped={self.dropped_samples}, queue_size={self.queue.qsize()}")
        return True

    async def get_sample(self) -> Any:
        """Get a single sample from the queue.

        Blocks (await) until a sample is available. Returns ``None`` when the
        queue is shut down and empty, signalling the trainer to stop.
        """
        while self.queue.empty() and self.running:
            await asyncio.sleep(0.05)
        if self.queue.empty() and not self.running:
            return None
        sample = await self.queue.get()
        self.total_consumed += 1
        return sample

    async def get_queue_size(self) -> int:
        return self.queue.qsize()

    async def get_statistics(self) -> dict:
        return {
            "queue_size": self.queue.qsize(),
            "total_produced": self.total_produced,
            "total_consumed": self.total_consumed,
            "dropped_samples": self.dropped_samples,
            "max_queue_size": self.max_queue_size,
            "running": self.running,
        }

    async def shutdown(self):
        self.running = False
        logger.info(f"[async-queue] shutdown: produced={self.total_produced}, consumed={self.total_consumed}, dropped={self.dropped_samples}")


SAMPLE_QUEUE_ACTOR_NAME = "fully_async_sample_queue"
SAMPLE_QUEUE_NAMESPACE = "fully_async_raygroup"


def create_sample_queue(max_queue_size: int = 256):
    """Create or get the singleton SampleQueue actor (named, detached)."""
    from aura.controllers.utils.utils import create_actor
    return create_actor(
        name=SAMPLE_QUEUE_ACTOR_NAME,
        cls=SampleQueue,
        namespace=SAMPLE_QUEUE_NAMESPACE,
        options={"num_cpus": SAMPLE_QUEUE_NUM_CPUS, "max_concurrency": SAMPLE_QUEUE_MAX_CONCURRENCY},
        actor_kwargs={"max_queue_size": max_queue_size},
    )


def get_sample_queue():
    """Get the existing SampleQueue actor handle."""
    return ray.get_actor(SAMPLE_QUEUE_ACTOR_NAME, namespace=SAMPLE_QUEUE_NAMESPACE)
