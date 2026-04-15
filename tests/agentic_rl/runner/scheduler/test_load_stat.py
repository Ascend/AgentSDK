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
import asyncio
import os
import pytest
import sys
import time
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for load_stat tests."""
    mock_vllm = MagicMock()
    mock_vllm_config = MagicMock()
    mock_vllm_v1_core = MagicMock()
    mock_vllm_v1_metrics = MagicMock()
    mock_vllm_v1_spec_decode = MagicMock()
    mock_vllm_v1_metrics_loggers = MagicMock()

    class MockPrefixCachingMetrics:
        def __init__(self):
            self.hit_rate = 0.0

        def observe(self, stats):
            pass

    class MockSpecDecodingLogging:
        def observe(self, stats):
            pass

        def log(self, log_fn):
            pass

    class MockSpecDecodingProm:
        pass

    class MockIterationStats:
        pass

    class MockSchedulerStats:
        pass

    mock_vllm_config.SupportsMetricsInfo = MagicMock
    mock_vllm_config.VllmConfig = MagicMock
    mock_vllm_v1_core.PrefixCachingMetrics = MockPrefixCachingMetrics
    mock_vllm_v1_metrics.IterationStats = MockIterationStats
    mock_vllm_v1_metrics.SchedulerStats = MockSchedulerStats
    mock_vllm_v1_spec_decode.SpecDecodingLogging = MockSpecDecodingLogging
    mock_vllm_v1_spec_decode.SpecDecodingProm = MockSpecDecodingProm
    mock_vllm_v1_metrics_loggers.StatLoggerBase = object

    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", mock_vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.v1", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.core", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", mock_vllm_v1_core)
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.stats", mock_vllm_v1_metrics)
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode.metrics", mock_vllm_v1_spec_decode)
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.loggers", mock_vllm_v1_metrics_loggers)

    monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.load_stat", raising=False)

    yield {
        "MockPrefixCachingMetrics": MockPrefixCachingMetrics,
        "MockSpecDecodingLogging": MockSpecDecodingLogging,
        "MockIterationStats": MockIterationStats,
        "MockSchedulerStats": MockSchedulerStats,
    }


class TestVllmLogStatsPeriodically:

    def setup_method(self):
        """Setup method to import functions before each test."""
        from agentic_rl.runner.scheduler.load_stat import vllm_log_stats_periodically
        self.vllm_log_stats_periodically = vllm_log_stats_periodically

    @pytest.mark.asyncio
    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    async def test_vllm_log_stats_periodically(self, mock_logger):
        mock_self = MagicMock()
        mock_self.engine.do_log_stats = AsyncMock()

        call_count = [0]

        async def sleep_side_effect(interval):
            call_count[0] += 1
            if call_count[0] == 1:
                return
            raise asyncio.CancelledError()

        with patch('asyncio.sleep', side_effect=sleep_side_effect):
            with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "5"}):
                with pytest.raises(asyncio.CancelledError):
                    await self.vllm_log_stats_periodically(mock_self)

        mock_self.engine.do_log_stats.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    async def test_vllm_log_stats_periodically_with_exception(self, mock_logger):
        mock_self = MagicMock()
        mock_self.engine.do_log_stats = AsyncMock(side_effect=Exception("Test error"))

        call_count = [0]

        async def sleep_side_effect(interval):
            call_count[0] += 1
            if call_count[0] == 1:
                return
            raise asyncio.CancelledError()

        with patch('asyncio.sleep', side_effect=sleep_side_effect):
            with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "5"}):
                with pytest.raises(asyncio.CancelledError):
                    await self.vllm_log_stats_periodically(mock_self)

        mock_logger.error.assert_called()


class TestWorkloadStatLogger:

    def setup_method(self):
        """Setup method to import classes before each test."""
        from agentic_rl.runner.scheduler.load_stat import WorkloadStatLogger
        from agentic_rl.runner.scheduler.workload import InstanceWorkLoad
        self.WorkloadStatLogger = WorkloadStatLogger
        self.InstanceWorkLoad = InstanceWorkLoad

    def _create_mock_vllm_config(self):
        """Helper method to create mock vllm config."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.workload = self.InstanceWorkLoad(dp_size=1)
        mock_vllm_config.scheduler_config.max_num_seqs = 8
        mock_vllm_config.cache_config.num_gpu_blocks = 100
        return mock_vllm_config

    @pytest.fixture
    def logger_instance(self):
        mock_vllm_config = self._create_mock_vllm_config()
        return self.WorkloadStatLogger(mock_vllm_config, engine_index=0)

    def test_init(self):
        mock_vllm_config = self._create_mock_vllm_config()
        logger = self.WorkloadStatLogger(mock_vllm_config, engine_index=0)

        assert logger.engine_index == 0
        assert logger.vllm_config == mock_vllm_config
        assert logger.num_prompt_tokens == 0
        assert logger.num_generation_tokens == 0
        assert logger.tpot_list == []
        assert logger.ttft_list == []
        assert logger.num_finished_requests == 0

    def test_reset(self, logger_instance):
        logger_instance.num_prompt_tokens = 100
        logger_instance.num_generation_tokens = 200
        logger_instance.tpot_list = [0.1, 0.2]
        logger_instance.ttft_list = [1.0, 2.0]

        now = time.monotonic()
        logger_instance._reset(now)

        assert logger_instance.num_prompt_tokens == 0
        assert logger_instance.num_generation_tokens == 0
        assert logger_instance.tpot_list == []
        assert logger_instance.ttft_list == []
        assert logger_instance.num_finished_requests == 0

    def test_track_iteration_stats(self, logger_instance, mock_dependencies):
        MockIterationStats = mock_dependencies["MockIterationStats"]

        mock_iteration_stats = MagicMock(spec=MockIterationStats)
        mock_iteration_stats.num_prompt_tokens = 50
        mock_iteration_stats.num_generation_tokens = 100

        mock_req1 = MagicMock()
        mock_req1.prefill_time = 1.0
        mock_req1.decode_time = 4.0
        mock_req1.num_generation_tokens = 10

        mock_req2 = MagicMock()
        mock_req2.prefill_time = 1.5
        mock_req2.decode_time = 3.0
        mock_req2.num_generation_tokens = 10

        mock_iteration_stats.finished_requests = [mock_req1, mock_req2]

        logger_instance._track_iteration_stats(mock_iteration_stats)

        assert logger_instance.num_prompt_tokens == 50
        assert logger_instance.num_generation_tokens == 100
        assert logger_instance.ttft_list == [1.0, 1.5]
        assert logger_instance.tpot_list == [0.4, 0.3]

    def test_get_throughput(self, logger_instance):
        now = time.monotonic()
        logger_instance.last_log_time = now - 5.0

        throughput = logger_instance._get_throughput(100, now)
        assert throughput == 20.0

        logger_instance.last_log_time = now
        throughput = logger_instance._get_throughput(100, now)
        assert throughput == 0.0

    def test_record_with_iteration_stats(self, logger_instance, mock_dependencies):
        MockSchedulerStats = mock_dependencies["MockSchedulerStats"]

        mock_scheduler_stats = MagicMock(spec=MockSchedulerStats)
        mock_scheduler_stats.prefix_cache_stats = None
        mock_scheduler_stats.spec_decoding_stats = None
        mock_iteration_stats = MagicMock()

        with patch.object(logger_instance, '_track_iteration_stats') as mock_track:
            logger_instance.record(mock_scheduler_stats, mock_iteration_stats)

            mock_track.assert_called_once_with(mock_iteration_stats)
            assert logger_instance.last_scheduler_stats == mock_scheduler_stats

    def test_record_without_iteration_stats(self, logger_instance, mock_dependencies):
        MockSchedulerStats = mock_dependencies["MockSchedulerStats"]

        mock_scheduler_stats = MagicMock(spec=MockSchedulerStats)
        mock_scheduler_stats.prefix_cache_stats = None
        mock_scheduler_stats.spec_decoding_stats = None

        with patch.object(logger_instance, '_track_iteration_stats') as mock_track:
            logger_instance.record(mock_scheduler_stats, None)

            mock_track.assert_not_called()
            assert logger_instance.last_scheduler_stats == mock_scheduler_stats

    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    def test_log(self, mock_logger, logger_instance):
        logger_instance.num_prompt_tokens = 100
        logger_instance.num_generation_tokens = 200
        logger_instance.tpot_list = [0.1, 0.2, 0.3]
        logger_instance.ttft_list = [1.0, 1.5, 2.0]

        logger_instance.last_scheduler_stats = MagicMock()
        logger_instance.last_scheduler_stats.num_running_reqs = 5
        logger_instance.last_scheduler_stats.num_waiting_reqs = 2
        logger_instance.last_scheduler_stats.kv_cache_usage = 0.75

        logger_instance.prefix_caching_metrics.hit_rate = 0.8

        logger_instance.log()

        assert logger_instance.ins_workload.dp_loads["0"].num_running_reqs == 5
        assert logger_instance.ins_workload.dp_loads["0"].num_waiting_reqs == 2
        assert logger_instance.ins_workload.dp_loads["0"].prompt_throughput > 0
        assert logger_instance.ins_workload.dp_loads["0"].generation_throughput > 0
        assert logger_instance.ins_workload.dp_loads["0"].kv_cache_usage == 75.0
        assert logger_instance.ins_workload.dp_loads["0"].prefixcache_hit_rate == 80.0
        assert logger_instance.ins_workload.dp_loads["0"].tpot == 0.2
        assert logger_instance.ins_workload.dp_loads["0"].ttft == 1.5

        assert logger_instance.num_prompt_tokens == 0
        assert logger_instance.num_generation_tokens == 0
        assert logger_instance.tpot_list == []
        assert logger_instance.ttft_list == []

    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    def test_log_with_no_throughput(self, mock_logger, logger_instance):
        logger_instance.last_scheduler_stats = MagicMock()
        logger_instance.last_scheduler_stats.num_running_reqs = 0
        logger_instance.last_scheduler_stats.num_waiting_reqs = 0
        logger_instance.last_scheduler_stats.kv_cache_usage = 0.0

        logger_instance.prefix_caching_metrics.hit_rate = 0.0

        logger_instance.spec_decoding_logging.log = MagicMock()

        logger_instance.log()

        logger_instance.spec_decoding_logging.log.assert_called_once()
        call_kwargs = logger_instance.spec_decoding_logging.log.call_args[1]
        assert call_kwargs['log_fn'] == mock_logger.error

    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    def test_log_engine_initialized(self, mock_logger, logger_instance):
        logger_instance.vllm_config.cache_config.num_gpu_blocks = 100

        logger_instance.log_engine_initialized()

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "vllm cache_config_info" in call_args[0][0]
        assert call_args[0][1] == 0
        assert call_args[0][2] == 100

    @patch('agentic_rl.runner.scheduler.load_stat.logger')
    def test_log_engine_initialized_no_gpu_blocks(self, mock_logger, logger_instance):
        logger_instance.vllm_config.cache_config.num_gpu_blocks = None

        logger_instance.log_engine_initialized()

        mock_logger.info.assert_not_called()
