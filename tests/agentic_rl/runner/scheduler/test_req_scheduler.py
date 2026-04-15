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
from unittest.mock import patch, MagicMock, AsyncMock


def create_mock_workload_manager():
    """Create a mock WorkLoadManger class with all required methods."""

    class MockWorkLoadManger:
        def __init__(self, addresses, dp_size, role):
            self.addresses = addresses
            self.dp_size = dp_size
            self.role = role
            self.ins_loads = {}
            for addr in addresses:
                self.ins_loads[addr] = MagicMock()
                self.ins_loads[addr].dp_loads = {}
                self.ins_loads[addr].max_num_seqs = 8
                for dp_id in range(dp_size):
                    self.ins_loads[addr].dp_loads[str(dp_id)] = MagicMock()
                    self.ins_loads[addr].dp_loads[str(dp_id)].get_load_score = MagicMock(return_value=0)
                    self.ins_loads[addr].dp_loads[str(dp_id)].add_req = MagicMock()
                    self.ins_loads[addr].dp_loads[str(dp_id)].del_req = MagicMock()
                    self.ins_loads[addr].dp_loads[str(dp_id)].num_routing_reqs = 0
                    self.ins_loads[addr].dp_loads[str(dp_id)].num_running_reqs = 0

    return MockWorkLoadManger


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for req_scheduler tests."""
    mock_utils = MagicMock()
    mock_utils.DEFAULT_URL_METHOD = "http"

    mock_workload = MagicMock()
    mock_workload.WorkLoadManger = create_mock_workload_manager()
    mock_workload.start_workload_update = MagicMock(return_value=MagicMock())

    monkeypatch.setitem(sys.modules, "agentic_rl.controllers.utils.utils", mock_utils)
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.workload", mock_workload)

    monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", raising=False)

    with patch("agentic_rl.runner.scheduler.req_scheduler.logger") as mock_logger:
        yield {
            "logger": mock_logger,
            "utils": mock_utils,
            "workload": mock_workload,
        }


class TestSchedulerBase:
    """Tests for SchedulerBase class."""

    def setup_method(self):
        """Setup method to import scheduler classes before each test."""
        from agentic_rl.runner.scheduler.req_scheduler import (
            SchedulerBase, SimpleTrajScheduler, LBStepScheduler,
            LBTrajScheduler, SchedulerFactory
        )
        self.SchedulerBase = SchedulerBase
        self.SimpleTrajScheduler = SimpleTrajScheduler
        self.LBStepScheduler = LBStepScheduler
        self.LBTrajScheduler = LBTrajScheduler
        self.SchedulerFactory = SchedulerFactory

    @pytest.fixture
    def scheduler(self):
        with patch('time.sleep'):
            return self.SimpleTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

    def test_scheduler_base_get_schedule_result_not_implemented(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.SchedulerBase(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        with pytest.raises(NotImplementedError, match="Subclasses must implement get_schedule_result"):
            scheduler.get_schedule_result("app-1")

    def test_scheduler_base_release_address_not_implemented(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.SchedulerBase(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        with pytest.raises(NotImplementedError, match="Subclasses must implement release_address"):
            scheduler.release_address("192.168.1.1:8080-0", "app-1")

    def test_init(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.SimpleTrajScheduler(
                addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        assert scheduler.dp_size == 2
        assert len(scheduler.ins_address) == 2
        assert scheduler.scheduling is True
        assert "192.168.1.1:8080-0" in scheduler.running_reqs
        assert "192.168.1.1:8080-1" in scheduler.running_reqs
        assert "192.168.1.2:8080-0" in scheduler.running_reqs
        assert "192.168.1.2:8080-1" in scheduler.running_reqs

    def test_reset(self, scheduler):
        scheduler.scheduling = False
        scheduler.reset()
        assert scheduler.scheduling is True

    @pytest.mark.asyncio
    async def test_schedule_success(self, scheduler):
        result = await scheduler.schedule("app-1", "req-1")
        assert result is not None
        assert "app-1" in scheduler.application_id_to_request_id
        assert scheduler.application_id_to_request_id["app-1"] == "req-1"

    @pytest.mark.asyncio
    async def test_schedule_stopped(self, scheduler):
        scheduler.scheduling = False
        result = await scheduler.schedule("app-1", "req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_schedule_wait_then_success(self, scheduler):
        call_count = [0]

        original_get_schedule_result = scheduler.get_schedule_result

        def mock_get_schedule_result(application_id, force=False):
            call_count[0] += 1
            if call_count[0] < 3:
                return None
            return original_get_schedule_result(application_id, force)

        scheduler.get_schedule_result = mock_get_schedule_result

        result = await scheduler.schedule("app-1", "req-1")
        assert result is not None
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_schedule_wait_then_stopped(self, scheduler):
        call_count = [0]

        def mock_get_schedule_result(application_id, force=False):
            call_count[0] += 1
            if call_count[0] >= 2:
                scheduler.scheduling = False
            return None

        scheduler.get_schedule_result = mock_get_schedule_result

        result = await scheduler.schedule("app-1", "req-1")
        assert result is None
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_release(self, scheduler):
        await scheduler.schedule("app-1", "req-1")
        dp_addr = scheduler.application_id_to_dp["app-1"]

        await scheduler.release(dp_addr, "app-1", "req-1")

        assert "app-1" not in scheduler.application_id_to_request_id

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cancel_requests(self, mock_session, scheduler):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        scheduler.running_reqs["192.168.1.1:8080-0"] = ["req-1", "req-2"]

        await scheduler.cancel_requests()

        assert scheduler.scheduling is False
        assert scheduler.running_reqs["192.168.1.1:8080-0"] == []

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cancel_request(self, mock_session, scheduler):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        scheduler.application_id_to_ins["app-1"] = "192.168.1.1:8080"
        scheduler.application_id_to_request_id["app-1"] = "req-1"

        await scheduler.cancel_request("app-1")

        mock_session_instance.post.assert_called_once()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cancel_request_exception(self, mock_session, scheduler, mock_dependencies):
        mock_session_instance = MagicMock()
        mock_session_instance.post = AsyncMock(side_effect=Exception("Network error"))
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        scheduler.application_id_to_ins["app-1"] = "192.168.1.1:8080"
        scheduler.application_id_to_request_id["app-1"] = "req-1"

        await scheduler.cancel_request("app-1")

        mock_dependencies["logger"].error.assert_called_once()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cancel_request_dp_addr_is_none(self, mock_session, scheduler):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        scheduler.application_id_to_ins["app-1"] = None
        scheduler.application_id_to_dp["app-1"] = "192.168.1.1:8080-0"
        scheduler.application_id_to_request_id["app-1"] = "req-1"

        await scheduler.cancel_request("app-1")

        mock_session_instance.post.assert_called_once()
        call_args = mock_session_instance.post.call_args
        assert "192.168.1.1:8080/v1/cancel" in call_args[0][0]


class TestSimpleTrajScheduler:
    """Tests for SimpleTrajScheduler class."""

    def setup_method(self):
        """Setup method to import scheduler classes before each test."""
        from agentic_rl.runner.scheduler.req_scheduler import SimpleTrajScheduler
        self.SimpleTrajScheduler = SimpleTrajScheduler

    @pytest.fixture
    def scheduler(self):
        with patch('time.sleep'):
            return self.SimpleTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

    def test_get_best_instance(self, scheduler):
        result = scheduler.get_best_instance("app-1")
        assert result == ""

    def test_get_best_dp(self, scheduler):
        result = scheduler.get_best_dp("192.168.1.1:8080", "app-1")
        assert result == 0

    def test_get_schedule_result_first_time(self, scheduler):
        result = scheduler.get_schedule_result("app-1")
        assert result is not None
        assert "app-1" in scheduler.application_id_to_dp
        assert scheduler._usage[result] == 1

    def test_get_schedule_result_same_application(self, scheduler):
        first_result = scheduler.get_schedule_result("app-1")
        second_result = scheduler.get_schedule_result("app-1")
        assert first_result is not None
        assert second_result is not None
        total_usage = sum(scheduler._usage.values())
        assert total_usage == 2

    def test_get_schedule_result_load_balance(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.SimpleTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        scheduler._usage["192.168.1.1:8080-0"] = 10
        scheduler._usage["192.168.1.1:8080-1"] = 0

        result = scheduler.get_schedule_result("app-2")
        assert result == "192.168.1.1:8080-1"

    def test_release_address(self, scheduler):
        scheduler.get_schedule_result("app-1")
        dp_addr = scheduler.application_id_to_dp["app-1"]
        initial_usage = scheduler._usage[dp_addr]

        scheduler.release_address(dp_addr, "app-1")

        assert scheduler._usage[dp_addr] == max(0, initial_usage - 1)

    def test_get_schedule_result_no_load_balance(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.SimpleTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        scheduler._usage["192.168.1.1:8080-0"] = 5
        scheduler._usage["192.168.1.1:8080-1"] = 4

        scheduler.application_id_to_dp["app-1"] = "192.168.1.1:8080-0"

        result = scheduler.get_schedule_result("app-1")

        assert result == "192.168.1.1:8080-0"
        assert scheduler._usage["192.168.1.1:8080-0"] == 6


class TestLBStepScheduler:
    """Tests for LBStepScheduler class."""

    def setup_method(self):
        """Setup method to import scheduler classes before each test."""
        from agentic_rl.runner.scheduler.req_scheduler import LBStepScheduler
        self.LBStepScheduler = LBStepScheduler

    @pytest.fixture
    def scheduler(self):
        with patch('time.sleep'):
            return self.LBStepScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

    def test_get_best_instance(self, scheduler):
        result = scheduler.get_best_instance("app-1")
        assert result == "192.168.1.1:8080"

    def test_get_best_dp(self, scheduler):
        result = scheduler.get_best_dp("192.168.1.1:8080", "app-1")
        assert result in ["0", "1"]

    def test_get_schedule_result(self, scheduler):
        result = scheduler.get_schedule_result("app-1")
        assert result is not None
        assert "app-1" in scheduler.application_id_to_ins
        assert "app-1" in scheduler.application_id_to_dp

    def test_get_schedule_result_full_load(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBStepScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=1,
                workload_inf=None,
                role="test"
            )

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=10)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        result = scheduler.get_schedule_result("app-1")
        assert result is None

    def test_get_schedule_result_force(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBStepScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=1,
                workload_inf=None,
                role="test"
            )

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=10)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        result = scheduler.get_schedule_result("app-1", force=True)
        assert result is not None

    def test_release_address(self, scheduler):
        result = scheduler.get_schedule_result("app-1")
        scheduler.release_address(result, "app-1")
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].del_req.assert_called()


class TestLBTrajScheduler:
    """Tests for LBTrajScheduler class."""

    def setup_method(self):
        """Setup method to import scheduler classes before each test."""
        from agentic_rl.runner.scheduler.req_scheduler import LBTrajScheduler
        self.LBTrajScheduler = LBTrajScheduler

    @pytest.fixture
    def scheduler(self):
        with patch('time.sleep'):
            return self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

    def test_get_best_instance(self, scheduler):
        result = scheduler.get_best_instance("prompt-1")
        assert result == "192.168.1.1:8080"

    def test_get_best_dp(self, scheduler):
        result = scheduler.get_best_dp("192.168.1.1:8080", "prompt-1")
        assert result in ["0", "1"]

    def test_get_schedule_result_first_time(self, scheduler):
        result = scheduler.get_schedule_result("prompt-1-app-1")
        assert result is not None
        assert "prompt-1-app-1" in scheduler.application_id_to_dp
        assert "prompt" in scheduler.prompt_id_to_dps

    def test_get_schedule_result_same_prompt(self, scheduler):
        first_result = scheduler.get_schedule_result("prompt-1-app-1")
        second_result = scheduler.get_schedule_result("prompt-1-app-2")
        assert first_result is not None
        assert second_result is not None

    def test_get_schedule_result_full_load(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=1,
                workload_inf=None,
                role="test"
            )

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=10)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        result = scheduler.get_schedule_result("prompt-1-app-1")
        assert result is None

    def test_get_schedule_result_application_already_scheduled(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        first_result = scheduler.get_schedule_result("prompt-1-app-1")
        assert first_result is not None

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=5)
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["1"].get_load_score = MagicMock(return_value=2)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        second_result = scheduler.get_schedule_result("prompt-1-app-1")
        assert second_result == first_result

    def test_get_schedule_result_application_already_scheduled_high_load(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        first_result = scheduler.get_schedule_result("prompt-1-app-1")
        assert first_result is not None

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=10)
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["1"].get_load_score = MagicMock(return_value=2)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        second_result = scheduler.get_schedule_result("prompt-1-app-1")
        assert second_result is not None

    def test_get_schedule_result_cached_dp_available(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        scheduler.prompt_id_to_dps["prompt"] = {"192.168.1.1:8080-0"}
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=5)
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].num_routing_reqs = 5
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].num_running_reqs = 3
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["1"].get_load_score = MagicMock(return_value=2)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8
        result = scheduler.get_schedule_result("prompt-1-app-1")
        assert result == "192.168.1.1:8080-0"

    def test_get_schedule_result_cached_dp_not_available(self, mock_dependencies):
        with patch('time.sleep'):
            scheduler = self.LBTrajScheduler(
                addresses=["192.168.1.1:8080"],
                dp_size=2,
                workload_inf=None,
                role="test"
            )

        scheduler.prompt_id_to_dps["prompt-1"] = {"192.168.1.1:8080-0"}

        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].get_load_score = MagicMock(return_value=10)
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].num_routing_reqs = 3
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].num_running_reqs = 5
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["1"].get_load_score = MagicMock(return_value=2)
        scheduler.workload.ins_loads["192.168.1.1:8080"].max_num_seqs = 8

        result = scheduler.get_schedule_result("prompt-1-app-2")
        assert result == "192.168.1.1:8080-1"

    def test_release_address(self, scheduler):
        result = scheduler.get_schedule_result("prompt-1-app-1")
        scheduler.release_address(result, "prompt-1-app-1")
        scheduler.workload.ins_loads["192.168.1.1:8080"].dp_loads["0"].del_req.assert_called()

    def test_reset(self, scheduler):
        scheduler.get_schedule_result("prompt-1-app-1")
        scheduler.reset()
        assert len(scheduler.application_id_to_dp) == 0
        assert len(scheduler.prompt_id_to_dps) == 0


class TestSchedulerFactory:
    """Tests for SchedulerFactory class."""

    def setup_method(self):
        """Setup method to import scheduler classes before each test."""
        from agentic_rl.runner.scheduler.req_scheduler import (
            SchedulerFactory, SimpleTrajScheduler, LBStepScheduler, LBTrajScheduler
        )
        self.SchedulerFactory = SchedulerFactory
        self.SimpleTrajScheduler = SimpleTrajScheduler
        self.LBStepScheduler = LBStepScheduler
        self.LBTrajScheduler = LBTrajScheduler

    def test_register(self, mock_dependencies):
        class CustomScheduler:
            def __init__(self, addresses, dp_size, workload_inf, role):
                pass

        self.SchedulerFactory.register('custom', CustomScheduler)
        assert 'custom' in self.SchedulerFactory._registry

    def test_get_scheduler_simple_traj(self, mock_dependencies):
        with patch.dict(os.environ, {"SCHEDULER_ALGO": "simple-traj"}):
            with patch('time.sleep'):
                scheduler = self.SchedulerFactory.get_scheduler(
                    addresses=["192.168.1.1:8080"],
                    dp_size=2,
                    workload_inf=None,
                    role="test"
                )

        assert isinstance(scheduler, self.SimpleTrajScheduler)

    def test_get_scheduler_lb_step(self, mock_dependencies):
        with patch.dict(os.environ, {"SCHEDULER_ALGO": "lb-step"}):
            with patch('time.sleep'):
                scheduler = self.SchedulerFactory.get_scheduler(
                    addresses=["192.168.1.1:8080"],
                    dp_size=2,
                    workload_inf=None,
                    role="test"
                )

        assert isinstance(scheduler, self.LBStepScheduler)

    def test_get_scheduler_lb_traj(self, mock_dependencies):
        with patch.dict(os.environ, {"SCHEDULER_ALGO": "lb-traj"}):
            with patch('time.sleep'):
                scheduler = self.SchedulerFactory.get_scheduler(
                    addresses=["192.168.1.1:8080"],
                    dp_size=2,
                    workload_inf=None,
                    role="test"
                )

        assert isinstance(scheduler, self.LBTrajScheduler)

    def test_get_scheduler_unknown(self, mock_dependencies):
        with patch.dict(os.environ, {"SCHEDULER_ALGO": "unknown"}):
            with pytest.raises(ValueError) as excinfo:
                with patch('time.sleep'):
                    self.SchedulerFactory.get_scheduler(
                        addresses=["192.168.1.1:8080"],
                        dp_size=2,
                        workload_inf=None,
                        role="test"
                    )

        assert "Unknown scheduler" in str(excinfo.value)
