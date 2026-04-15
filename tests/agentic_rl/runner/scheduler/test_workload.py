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
import json
import os
import pytest
import sys
from collections import deque
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for workload tests."""
    monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.workload", raising=False)

    with patch("agentic_rl.runner.scheduler.workload.logger") as mock_logger:
        yield {
            "logger": mock_logger,
        }


class TestDPWorkLoad:
    """Tests for DPWorkLoad class."""

    def setup_method(self):
        """Setup method to import DPWorkLoad before each test."""
        from agentic_rl.runner.scheduler.workload import DPWorkLoad
        self.DPWorkLoad = DPWorkLoad

    def test_init(self, mock_dependencies):
        """Test DPWorkLoad initialization."""
        dp_workload = self.DPWorkLoad()
        assert dp_workload.num_running_reqs == 0
        assert dp_workload.num_waiting_reqs == 0
        assert dp_workload.prompt_throughput == 0.0
        assert dp_workload.generation_throughput == 0.0
        assert dp_workload.kv_cache_usage == 0.0
        assert dp_workload.prefixcache_hit_rate == 0.0
        assert dp_workload.tpot == 0.0
        assert dp_workload.ttft == 0.0
        assert dp_workload.num_routing_reqs == 0
        assert isinstance(dp_workload.history, deque)
        assert len(dp_workload.history) == 0

    def test_update(self, mock_dependencies):
        """Test DPWorkLoad update method."""
        dp_workload = self.DPWorkLoad()
        data = {
            "num_running_reqs": 5,
            "num_waiting_reqs": 2,
            "prompt_throughput": 100.0,
            "generation_throughput": 200.0,
            "kv_cache_usage": 75.0,
            "prefixcache_hit_rate": 80.0,
            "tpot": 0.5,
            "ttft": 1.0,
        }

        dp_workload.update(data)

        assert dp_workload.num_running_reqs == 5
        assert dp_workload.num_waiting_reqs == 2
        assert dp_workload.prompt_throughput == 100.0
        assert dp_workload.generation_throughput == 200.0
        assert dp_workload.kv_cache_usage == 75.0
        assert dp_workload.prefixcache_hit_rate == 80.0
        assert dp_workload.tpot == 0.5
        assert dp_workload.ttft == 1.0

    def test_to_dict(self, mock_dependencies):
        """Test DPWorkLoad to_dict method."""
        dp_workload = self.DPWorkLoad()
        dp_workload.num_running_reqs = 3
        dp_workload.num_waiting_reqs = 1

        result = dp_workload.to_dict()

        assert result["num_running_reqs"] == 3
        assert result["num_waiting_reqs"] == 1

    def test_add_req(self, mock_dependencies):
        """Test DPWorkLoad add_req method."""
        dp_workload = self.DPWorkLoad()
        dp_workload.add_req()
        assert dp_workload.num_routing_reqs == 1

        dp_workload.add_req()
        assert dp_workload.num_routing_reqs == 2

    def test_del_req(self, mock_dependencies):
        """Test DPWorkLoad del_req method."""
        dp_workload = self.DPWorkLoad()
        dp_workload.num_routing_reqs = 2
        dp_workload.del_req()
        assert dp_workload.num_routing_reqs == 1

        dp_workload.del_req()
        assert dp_workload.num_routing_reqs == 0

        dp_workload.del_req()
        assert dp_workload.num_routing_reqs == 0

    def test_get_load_score(self, mock_dependencies):
        """Test DPWorkLoad get_load_score method."""
        dp_workload = self.DPWorkLoad()
        dp_workload.num_running_reqs = 2
        dp_workload.num_waiting_reqs = 1
        dp_workload.num_routing_reqs = 1

        score = dp_workload.get_load_score()
        assert score == 4

    def test_add_to_history(self, mock_dependencies):
        """Test DPWorkLoad add_to_history method."""
        dp_workload = self.DPWorkLoad()
        dp_workload.num_running_reqs = 5
        dp_workload.add_to_history()

        assert len(dp_workload.history) == 1
        assert dp_workload.history[0]["num_running_reqs"] == 5


class TestInstanceWorkLoad:
    """Tests for InstanceWorkLoad class."""

    def setup_method(self):
        """Setup method to import InstanceWorkLoad before each test."""
        from agentic_rl.runner.scheduler.workload import InstanceWorkLoad
        self.InstanceWorkLoad = InstanceWorkLoad

    def test_init_with_dp_address(self, mock_dependencies):
        """Test InstanceWorkLoad initialization with dp_address."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])

        assert len(workload.dp_loads) == 2
        assert "dp-0" in workload.dp_loads
        assert "dp-1" in workload.dp_loads
        assert workload.max_num_seqs == 8

    def test_init_with_dp_size(self, mock_dependencies):
        """Test InstanceWorkLoad initialization with dp_size."""
        workload = self.InstanceWorkLoad(dp_size=3)

        assert len(workload.dp_loads) == 3
        assert "0" in workload.dp_loads
        assert "1" in workload.dp_loads
        assert "2" in workload.dp_loads

    def test_to_dict(self, mock_dependencies):
        """Test InstanceWorkLoad to_dict method."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])
        workload.dp_loads["dp-0"].num_running_reqs = 5

        result = workload.to_dict()

        assert "dp_loads" in result
        assert "max_num_seqs" in result
        assert result["dp_loads"]["dp-0"]["num_running_reqs"] == 5

    def test_update(self, mock_dependencies):
        """Test InstanceWorkLoad update method."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])
        data = {
            "dp_loads": {
                "dp-0": {"num_running_reqs": 3},
                "dp-1": {"num_running_reqs": 2}
            },
            "max_num_seqs": 16
        }

        workload.update("192.168.1.1:8080", data)

        assert workload.dp_loads["dp-0"].num_running_reqs == 3
        assert workload.dp_loads["dp-1"].num_running_reqs == 2
        assert workload.max_num_seqs == 16

    def test_update_with_invalid_data(self, mock_dependencies):
        """Test InstanceWorkLoad update method with invalid data."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])
        workload.update("192.168.1.1:8080", "invalid")
        assert workload.max_num_seqs == 8

    def test_get_load_score(self, mock_dependencies):
        """Test InstanceWorkLoad get_load_score method."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])
        workload.dp_loads["dp-0"].num_running_reqs = 2
        workload.dp_loads["dp-0"].num_waiting_reqs = 1
        workload.dp_loads["dp-1"].num_running_reqs = 3

        score = workload.get_load_score()
        assert score == 6

    def test_add_to_history(self, mock_dependencies):
        """Test InstanceWorkLoad add_to_history method."""
        workload = self.InstanceWorkLoad(dp_address=["dp-0", "dp-1"])
        workload.dp_loads["dp-0"].num_running_reqs = 5
        workload.add_to_history()

        assert len(workload.dp_loads["dp-0"].history) == 1


class TestWorkLoadManger:
    """Tests for WorkLoadManger class."""

    def setup_method(self):
        """Setup method to import WorkLoadManger before each test."""
        from agentic_rl.runner.scheduler.workload import WorkLoadManger
        self.WorkLoadManger = WorkLoadManger

    def test_init(self, mock_dependencies):
        """Test WorkLoadManger initialization."""
        workload_manager = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=2,
            role="test"
        )
        assert len(workload_manager.ins_loads) == 2
        assert "192.168.1.1:8080" in workload_manager.ins_loads
        assert "192.168.1.2:8080" in workload_manager.ins_loads
        assert workload_manager.role == "test"

    def test_to_dict(self, mock_dependencies):
        """Test WorkLoadManger to_dict method."""
        workload_manager = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=2,
            role="test"
        )
        result = workload_manager.to_dict()

        assert "role" in result
        assert "ins_loads" in result
        assert result["role"] == "test"

    def test_update(self, mock_dependencies):
        """Test WorkLoadManger update method."""
        workload_manager = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=2,
            role="test"
        )
        data = {
            "192.168.1.1:8080": {
                "dp_loads": {"0": {"num_running_reqs": 3}},
                "max_num_seqs": 16
            }
        }

        workload_manager.update(data)

        assert workload_manager.ins_loads["192.168.1.1:8080"].dp_loads["0"].num_running_reqs == 3

    def test_add_to_history(self, mock_dependencies):
        """Test WorkLoadManger add_to_history method."""
        workload_manager = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=2,
            role="test"
        )
        workload_manager.add_to_history({})

        for ins_load in workload_manager.ins_loads.values():
            for dp_load in ins_load.dp_loads.values():
                assert len(dp_load.history) == 1

    def test_save_history_to_json(self, mock_dependencies, tmp_path):
        """Test WorkLoadManger save_history_to_json method."""
        workload_manager = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=2,
            role="test"
        )
        workload_manager.update({
            "192.168.1.1:8080": {
                "dp_loads": {"0": {"num_running_reqs": 5}},
                "max_num_seqs": 8
            }
        })
        workload_manager.add_to_history({})

        filename = str(tmp_path / "history.json")
        workload_manager.save_history_to_json(filename)

        with open(filename, 'r') as f:
            data = json.load(f)

        assert "192.168.1.1:8080" in data


class TestPollWorkloadOpenai:
    """Tests for poll_workload_openai function."""

    def setup_method(self):
        """Setup method to import poll_workload_openai before each test."""
        from agentic_rl.runner.scheduler.workload import poll_workload_openai
        self.poll_workload_openai = poll_workload_openai

    @pytest.mark.asyncio
    async def test_poll_workload_openai_success(self, mock_dependencies):
        """Test poll_workload_openai with successful response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value='{"dp_loads": {"0": {"num_running_reqs": 5}}}')

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        result = await self.poll_workload_openai(mock_session, "192.168.1.1:8080")

        assert result["dp_loads"]["0"]["num_running_reqs"] == 5

    @pytest.mark.asyncio
    async def test_poll_workload_openai_error(self, mock_dependencies):
        """Test poll_workload_openai with error response."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with pytest.raises(Exception):
            await self.poll_workload_openai(mock_session, "192.168.1.1:8080")


class TestPollAllInstances:
    """Tests for poll_all_instances function."""

    def setup_method(self):
        """Setup method to import poll_all_instances and WorkLoadManger before each test."""
        from agentic_rl.runner.scheduler.workload import poll_all_instances, WorkLoadManger
        self.poll_all_instances = poll_all_instances
        self.WorkLoadManger = WorkLoadManger

    @pytest.mark.asyncio
    async def test_poll_all_instances(self, mock_dependencies):
        """Test poll_all_instances with successful polling."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        mock_session = MagicMock()

        async def mock_poll(*args, **kwargs):
            return {"dp_loads": {"0": {"num_running_reqs": 3}}}

        with patch('agentic_rl.runner.scheduler.workload.poll_workload_openai', mock_poll):
            result = await self.poll_all_instances(mock_session, workloads)

        assert "192.168.1.1:8080" in result

    @pytest.mark.asyncio
    async def test_poll_all_instances_with_exception_result(self, mock_dependencies):
        """Test poll_all_instances with exception in result."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            dp_size=1
        )

        mock_session = MagicMock()

        call_count = [0]

        async def mock_poll(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"dp_loads": {"0": {"num_running_reqs": 3}}}
            else:
                raise Exception("Connection refused")

        with patch('agentic_rl.runner.scheduler.workload.poll_workload_openai', mock_poll):
            result = await self.poll_all_instances(mock_session, workloads)

        assert "192.168.1.1:8080" in result
        assert "192.168.1.2:8080" in result
        assert result["192.168.1.1:8080"] == {"dp_loads": {"0": {"num_running_reqs": 3}}}
        assert "ERROR:" in result["192.168.1.2:8080"]

    @pytest.mark.asyncio
    async def test_poll_all_instances_with_gather_exception(self, mock_dependencies):
        """Test poll_all_instances with gather exception."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        mock_session = MagicMock()

        async def mock_gather(*args, **kwargs):
            for coro in args:
                coro.close()
            raise RuntimeError("gather failed")

        with patch('asyncio.gather', mock_gather):
            result = await self.poll_all_instances(mock_session, workloads)

        assert result == {}


class TestWorkloadUpdatePeriodically:
    """Tests for workload_update_periodically function."""

    def setup_method(self):
        """Setup method to import workload_update_periodically and WorkLoadManger before each test."""
        from agentic_rl.runner.scheduler.workload import workload_update_periodically, WorkLoadManger
        self.workload_update_periodically = workload_update_periodically
        self.WorkLoadManger = WorkLoadManger

    @pytest.mark.asyncio
    async def test_workload_update_periodically(self, mock_dependencies):
        """Test workload_update_periodically normal execution."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        call_count = [0]

        async def mock_poll_all(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()
            return {"192.168.1.1:8080": {"dp_loads": {"0": {"num_running_reqs": 3}}}}

        async def mock_sleep(interval):
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "1", "LOG_WORKLOAD_ENABLE": "0"}):
            with patch('agentic_rl.runner.scheduler.workload.poll_all_instances', mock_poll_all):
                with patch('asyncio.sleep', mock_sleep):
                    with patch('aiohttp.ClientSession') as mock_session_cls:
                        mock_session = AsyncMock()
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_session_cls.return_value = mock_session
                        await self.workload_update_periodically(None, workloads)

    @pytest.mark.asyncio
    async def test_workload_update_periodically_invalid_interval(self, mock_dependencies):
        """Test workload_update_periodically with invalid interval."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        call_count = [0]

        async def mock_poll_all(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()
            return {"192.168.1.1:8080": {"dp_loads": {"0": {"num_running_reqs": 3}}}}

        async def mock_sleep(interval):
            assert interval == 10.0
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "invalid", "LOG_WORKLOAD_ENABLE": "0"}):
            with patch('agentic_rl.runner.scheduler.workload.poll_all_instances', mock_poll_all):
                with patch('asyncio.sleep', mock_sleep):
                    with patch('aiohttp.ClientSession') as mock_session_cls:
                        mock_session = AsyncMock()
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_session_cls.return_value = mock_session
                        await self.workload_update_periodically(None, workloads)

    @pytest.mark.asyncio
    async def test_workload_update_periodically_print_stats_enabled(self, mock_dependencies):
        """Test workload_update_periodically with print stats enabled."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        call_count = [0]

        async def mock_poll_all(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()
            return {"192.168.1.1:8080": {"dp_loads": {"0": {"num_running_reqs": 3}}}}

        async def mock_sleep(interval):
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "1", "LOG_WORKLOAD_ENABLE": "1"}):
            with patch('agentic_rl.runner.scheduler.workload.poll_all_instances', mock_poll_all):
                with patch('asyncio.sleep', mock_sleep):
                    with patch('aiohttp.ClientSession') as mock_session_cls:
                        mock_session = AsyncMock()
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_session_cls.return_value = mock_session
                        await self.workload_update_periodically(None, workloads)
                    mock_dependencies["logger"].info.assert_called()

    @pytest.mark.asyncio
    async def test_workload_update_periodically_exception_in_loop(self, mock_dependencies):
        """Test workload_update_periodically with exception in loop."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        call_count = [0]

        async def mock_poll_all(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("poll failed")
            raise asyncio.CancelledError()

        async def mock_sleep(interval):
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch.dict(os.environ, {"VLLM_LOG_STATS_INTERVAL": "1", "LOG_WORKLOAD_ENABLE": "0"}):
            with patch('agentic_rl.runner.scheduler.workload.poll_all_instances', mock_poll_all):
                with patch('asyncio.sleep', mock_sleep):
                    with patch('aiohttp.ClientSession') as mock_session_cls:
                        mock_session = AsyncMock()
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_session_cls.return_value = mock_session
                        await self.workload_update_periodically(None, workloads)
                    mock_dependencies["logger"].error.assert_called()


class TestStartWorkloadUpdate:
    """Tests for start_workload_update function."""

    def setup_method(self):
        """Setup method to import start_workload_update and WorkLoadManger before each test."""
        from agentic_rl.runner.scheduler.workload import start_workload_update, WorkLoadManger
        self.start_workload_update = start_workload_update
        self.WorkLoadManger = WorkLoadManger

    def test_start_workload_update(self, mock_dependencies):
        """Test start_workload_update function."""
        workloads = self.WorkLoadManger(
            addresses=["192.168.1.1:8080"],
            dp_size=1
        )

        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            result = self.start_workload_update(None, workloads)

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            assert result == mock_thread_instance
