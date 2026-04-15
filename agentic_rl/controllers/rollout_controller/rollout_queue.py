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
import threading
from collections import deque
from threading import Lock

import ray

from agentic_rl.base.log.loggers import Loggers


@ray.remote
class RolloutQueueActor:
    def __init__(self):
        self.logger = Loggers(__name__).get_logger()
        self.batch_queue = deque()
        self.queue_lock = Lock()

        self.abort_queue = deque()
        self.abort_lock = Lock()

        self.running = False
        self.weight_updating_status = {}
        self.shutdown_event = threading.Event()

    def add_abort_queue(self, request):
        with self.abort_lock:
            self.abort_queue.append(request)
        self.logger.info(f"add request to abort queue, queue size: {len(self.abort_queue)}")

    def pop_abort_queue(self):
        with self.abort_lock:
            request = self.abort_queue.popleft()
            self.logger.info(f"pop request from abort queue, queue size: {len(self.abort_queue)}")
            return request

    def add_queue(self, batch):
        with self.queue_lock:
            self.batch_queue.append(batch)
        self.logger.info(f"add batch to queue, queue size: {len(self.batch_queue)}")

    def pop_queue(self):
        with self.queue_lock:
            batch = self.batch_queue.popleft()
            self.logger.info(f"pop batch from queue, queue size: {len(self.batch_queue)}")
            return batch

    def queue_size(self):
        return len(self.batch_queue)

    def set_block(self):
        self.running = False

    def set_running(self):
        self.running = True

    def is_running(self):
        return self.running

    def shutdown(self):
        self.shutdown_event.set()

    def is_shutdown(self):
        return self.shutdown_event.is_set()

    def init_done(self):
        pass


def get_rollout_queue_actor():
    return ray.get_actor("rollout_queue", namespace="controller_raygroup")
