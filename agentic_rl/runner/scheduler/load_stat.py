#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------


# Standard library imports
import os
import statistics
import time
import asyncio
from typing import Optional

# vLLM imports
from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.scheduler.workload import InstanceWorkLoad

logger = Loggers(__name__).get_logger()


async def vllm_log_stats_periodically(self):
    interval = float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10"))  #  10 seconds
    while True:
        try:
            await asyncio.sleep(interval)
            await self.engine.do_log_stats()
        except Exception as e:
            logger.error(f"[ERROR] Failed to log stats: {e}")

class WorkloadStatLogger(StatLoggerBase):

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._reset(time.monotonic())
        self.last_scheduler_stats = SchedulerStats()
        # Prefix cache metrics. This cannot be reset.
        # TODO: Make the interval configurable.
        self.prefix_caching_metrics = PrefixCachingMetrics()
        self.spec_decoding_logging = SpecDecodingLogging()
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0
        self.ins_workload: InstanceWorkLoad = vllm_config.workload
        self.ins_workload.max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0
        self.tpot_list: list[float] = []
        self.ttft_list: list[float] = []
        self.num_finished_requests: int = 0

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens
        if iteration_stats.finished_requests:
            for req in iteration_stats.finished_requests:
                self.ttft_list.append(req.prefill_time)
                self.tpot_list.append(req.decode_time / req.num_generation_tokens)

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        # Compute summary metrics for tracked stats
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)
    
    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(scheduler_stats.spec_decoding_stats)

            self.last_scheduler_stats = scheduler_stats
    
    def log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)
        avg_ttft = statistics.mean(self.ttft_list) if len(self.ttft_list) > 0 else 0
        avg_tpot = statistics.mean(self.tpot_list) if len(self.tpot_list) > 0 else 0

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        log_fn = logger.info
        if not any(
            (prompt_throughput, generation_throughput,
             self.last_prompt_throughput, self.last_generation_throughput)):
            # Avoid log noise on an idle production system
            log_fn = logger.error
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

        self.spec_decoding_logging.log(log_fn=log_fn)
        self.ins_workload.dp_loads[str(self.engine_index)].num_running_reqs = scheduler_stats.num_running_reqs
        self.ins_workload.dp_loads[str(self.engine_index)].num_waiting_reqs = scheduler_stats.num_waiting_reqs
        self.ins_workload.dp_loads[str(self.engine_index)].prompt_throughput = prompt_throughput
        self.ins_workload.dp_loads[str(self.engine_index)].generation_throughput = generation_throughput
        self.ins_workload.dp_loads[str(self.engine_index)].kv_cache_usage = scheduler_stats.kv_cache_usage * 100
        self.ins_workload.dp_loads[str(self.engine_index)].prefixcache_hit_rate = self.prefix_caching_metrics.hit_rate * 100
        self.ins_workload.dp_loads[str(self.engine_index)].tpot = avg_tpot
        self.ins_workload.dp_loads[str(self.engine_index)].ttft = avg_ttft

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d", self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks)
            