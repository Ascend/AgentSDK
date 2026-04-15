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
import time
from dataclasses import dataclass

import ray

from agentic_rl.base.log.loggers import Loggers


@dataclass
class _ExportTracker:
    iteration: int
    start_ts: float  # wall-clock
    expected: int  # how many shards
    seen: int = 0  # shards written so far


@ray.remote
class WeightUpdateActor:
    def __init__(self, dispatch_actor, actor_handlers):
        self.logger = Loggers(__name__).get_logger()

        self.dispatch_actor = dispatch_actor
        self.actor_handlers = actor_handlers

        self.current_training_iter: int | None = 1
        self._exports: dict[str, _ExportTracker] = {}
        self.weight_export_events: list[dict] = []
        self.export_durations: list[float] = []
        self.finish_delays: list[int] = []

        self.update_finished = False

    def _update_weights_async(self, weight_save_dir: str, iteration: int):
        """Called by the trainer; returns immediately."""
        self.logger.info(f">>> updating weights with weights dir: {weight_save_dir} iteration: {iteration}")
        self.current_training_iter = iteration
        self._exports[weight_save_dir] = _ExportTracker(
            iteration=iteration,
            start_ts=time.time(),
            expected=len(self.actor_handlers) if type(self.actor_handlers) is list else len(
                self.actor_handlers._workers)
        )

        if type(self.actor_handlers) is list:
            [h.prepare_infer_params_to_cpu.remote(weight_save_dir) for h in self.actor_handlers]
        else:
            self.actor_handlers.prepare_infer_params_to_cpu(weight_save_dir)
        self.logger.info(
            f">>> start async exporting weights, expected actor count: {self._exports[weight_save_dir].expected}")

    def _update_metrics(self, weight_save_dir, start_ts, iteration):
        end_ts = time.time()
        duration = end_ts - start_ts
        self.export_durations.append(duration)

        # delay wrt training iter
        finish_delay = max(0, int(self.current_training_iter) - int(iteration))
        self.finish_delays.append(finish_delay)

        # event log
        self.weight_export_events.append({
            "weight_save_dir": weight_save_dir,
            "start": start_ts,
            "end": end_ts,
            "duration": duration,
            "status": "ok",
            "iteration": iteration,
            "finish_delay_iters": finish_delay,
        })

        self.logger.info(f"updating weights events: {self.weight_export_events}")

        # cleanup
        self._exports.pop(weight_save_dir, None)

    def _finalise_export(self, weight_save_dir, start_ts, iteration):
        # Notify generation units
        self.logger.info(f"weights are exported to path={weight_save_dir}, notify to update the weights.")
        self.dispatch_actor.notify_weights_update.remote(weight_save_dir)
        self._update_metrics(weight_save_dir, start_ts, iteration)

    def update_weights_to_file(self, weight_save_dir: str, iteration: int):
        self._update_weights_async(weight_save_dir=weight_save_dir, iteration=iteration)

    def weight_saved(self, weight_save_dir: str):
        tracker = self._exports.get(weight_save_dir)
        if tracker is None:
            return
        tracker.seen += 1
        if tracker.seen >= tracker.expected:
            self.update_finished = True
            self._finalise_export(weight_save_dir, tracker.start_ts, tracker.iteration)

    def update_weights_finished(self):
        if self.update_finished:
            self.update_finished = False
            return True
        return False

    def init_done(self):
        pass
