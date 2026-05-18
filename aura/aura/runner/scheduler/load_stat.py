#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
import statistics
import time
from collections import defaultdict, deque
import asyncio
from typing import Optional
from dataclasses import dataclass
from prometheus_client.parser import text_string_to_metric_families

from vllm.config import VllmConfig
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingLogging
from vllm.v1.metrics.loggers import StatLoggerBase

from aura.base.log.loggers import Loggers
from aura.runner.scheduler.workload import InstanceWorkLoad

logger = Loggers(__name__).get_logger()


@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""


class PrefixCachingMetrics:
    def __init__(self, max_recent_requests: int = 1000) -> None:
        super().__init__()

        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0

        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        if stats.reset:
            self.reset()

        if stats.requests == 0:
            return

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        while len(self.query_queue) > 1 and self.aggregated_requests > self.max_recent_requests:
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


def parse_prometheus_text_to_json(last_metrics, prometheus_text, max_num_seqs):
    engines_data = defaultdict(dict)

    metric_mapping = {
        'vllm:num_requests_running': 'num_requests_running',
        'vllm:num_requests_waiting': 'num_requests_waiting',
        'vllm:gpu_prefix_cache_hits_total': 'gpu_prefix_cache_hits_total',
        'vllm:prefix_cache_misses_total': 'prefix_cache_misses_total',
        'vllm:prompt_tokens_total': 'prompt_tokens_total',
        'vllm:generation_tokens_total': 'generation_tokens_total',
        'vllm:kv_cache_usage_perc': 'kv_cache_usage_perc',
    }

    for family in text_string_to_metric_families(prometheus_text):
        for sample in family.samples:
            metric_name = sample.name
            labels = sample.labels
            value = sample.value

            if metric_name not in metric_mapping:
                continue

            engine_id = labels.get('engine')
            if not engine_id:
                continue

            engines_data[engine_id][metric_mapping[metric_name]] = value

    dp_loads = {}
    org_dp_loads = {}
    interval = float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10"))

    for engine_id, metrics in engines_data.items():
        num_running_reqs = int(metrics.get('num_requests_running', 0))
        num_waiting_reqs = int(metrics.get('num_requests_waiting', 0))
        kv_cache_usage = metrics.get('kv_cache_usage_perc', 0.0) * 100

        prompt_tokens_total = metrics.get('prompt_tokens_total', 0.0)
        generation_tokens_total = metrics.get('generation_tokens_total', 0.0)

        gpu_cache_hits = metrics.get('gpu_prefix_cache_hits_total', 0.0)
        prefix_cache_misses = metrics.get('prefix_cache_misses_total', 0.0)
        prefix_cache_queries = gpu_cache_hits + prefix_cache_misses

        if prefix_cache_queries > 0:
            prefixcache_hit_rate = (gpu_cache_hits / prefix_cache_queries) * 100  # 转换为百分比
        else:
            prefixcache_hit_rate = 0.0

        if last_metrics is None:
            prompt_throughput = prompt_tokens_total / interval
            generation_throughput = generation_tokens_total / interval
        else:
            last_prompt_total = last_metrics[engine_id]['prompt_throughput']
            last_generation_total = last_metrics[engine_id]['generation_throughput']

            prompt_throughput = max(0, (prompt_tokens_total - last_prompt_total) / interval)
            generation_throughput = max(0, (generation_tokens_total - last_generation_total) / interval)

        ttft = 0.0
        tpot = 0.0

        dp_loads[engine_id] = {
            'num_running_reqs': num_running_reqs,
            'num_waiting_reqs': num_waiting_reqs,
            'num_routing_reqs': 0,
            'prompt_throughput': prompt_throughput,
            'generation_throughput': generation_throughput,
            'ttft': ttft,
            'tpot': tpot,
            'kv_cache_usage': kv_cache_usage,
            'prefixcache_hit_rate': prefixcache_hit_rate,
        }

        org_dp_loads[engine_id] = {
            'num_running_reqs': num_running_reqs,
            'num_waiting_reqs': num_waiting_reqs,
            'num_routing_reqs': 0,
            'prompt_throughput': prompt_tokens_total,
            'generation_throughput': generation_tokens_total,
            'ttft': ttft,
            'tpot': tpot,
            'kv_cache_usage': kv_cache_usage,
            'prefixcache_hit_rate': prefixcache_hit_rate,
        }

    return {'dp_loads': dp_loads, 'max_num_seqs': max_num_seqs}, org_dp_loads


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

    def record(
        self, scheduler_stats: Optional[SchedulerStats], iteration_stats: Optional[IterationStats], engine_idx: int = 0
    ):
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
            (prompt_throughput, generation_throughput, self.last_prompt_throughput, self.last_generation_throughput)
        ):
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
        self.ins_workload.dp_loads[str(self.engine_index)].prefixcache_hit_rate = (
            self.prefix_caching_metrics.hit_rate * 100
        )
        self.ins_workload.dp_loads[str(self.engine_index)].tpot = avg_tpot
        self.ins_workload.dp_loads[str(self.engine_index)].ttft = avg_ttft

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "Engine %03d: vllm cache_config_info with initialization after num_gpu_blocks is: %d",
                self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks,
            )
