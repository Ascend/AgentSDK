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
from collections import deque

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME, MAX_TIMEOUT


class TrainQueue:
    def __init__(
            self,
            max_queue_size: int,
            global_batch_size: int,
            n_samples_per_prompt: int
    ):
        self.logger = Loggers(__name__).get_logger()
        self.queue = deque()
        self.metric_queue = deque()

        self.n_samples_per_prompt = n_samples_per_prompt
        self.global_batch_size = global_batch_size
        self.max_queue_len = max_queue_size * global_batch_size * n_samples_per_prompt

        self.timeout = MAX_TIMEOUT
        self.sleep_time = DEFAULT_SLEEP_TIME
        self.logger.info(f">>> queue actor create success")

    def add_minibatch(self, outputs, metric) -> bool:
        """
        Always enqueues the batch.  Returns False if, *after* enqueueing,
        the queue's length has reached or exceeded max_queue_len.
        """
        self.queue.append(outputs)
        self.metric_queue.append(metric)

        if len(self.queue) > self.max_queue_len:
            self.logger.warning(f"Queue size {len(self.queue)} exceeded max_queue_len={self.max_queue_len}")
            return False
        return True

    # Dead loop waiting for data (considering the rollout phase to be particularly long)
    def pop_batch(self):
        """Called by the DataLoader to pull the next sample."""
        out = self.queue.popleft()
        metric = self.metric_queue.popleft()
        return out, metric

    def size(self) -> int:
        return len(self.queue)

    def get_max_len(self):
        return self.max_queue_len

    def __len__(self):
        if not self.queue:
            return 0
        return len(self.queue)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.queue):
            raise IndexError(index)
