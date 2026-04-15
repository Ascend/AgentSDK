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


# Standard library imports
import os
import time
import asyncio
from typing import Optional

# Third-party library imports
import aiohttp

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.utils import DEFAULT_URL_METHOD
from agentic_rl.runner.scheduler.workload import WorkLoadManger, start_workload_update

logger = Loggers(__name__).get_logger()

class SchedulerBase:
    """
    scheduler chooses a infer dp-instance
    """

    def __init__(self, addresses: list[str], dp_size: int = 1, workload_inf=None, role=""):
        # instance address, eg: tp4dp8 on 4 nodes is an instance
        self.ins_address = addresses
        self.dp_size = dp_size

        self.application_id_to_request_id: dict[str, str] = {}
        self.application_id_to_ins: dict[str, str] = {}
        self.application_id_to_dp: dict[str, str] = {}
        self.prompt_id_to_dps: dict[str, set[str]] = {}
        self.scheduling = True
        self.role = role
        # dp_addr: vllm-request_id
        self.running_reqs: dict[str, list[str]] = {}
        for ins in addresses:
            for dp_id in range(self.dp_size):
                self.running_reqs[f"{ins}-{dp_id}"] = []
        self.workload = WorkLoadManger(addresses, self.dp_size, self.role)
        self.stat_thread = start_workload_update(workload_inf, self.workload)
        self._lock = asyncio.Lock()
        interval = int(os.getenv("VLLM_LOG_STATS_INTERVAL", "10"))
        # Ensure receiving a workload report
        time.sleep(interval * 2)

    def get_schedule_result(self, application_id: str, force: bool = False) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement get_schedule_result")

    def release_address(self, addr: str, application_id: str) -> None:
        raise NotImplementedError("Subclasses must implement release_address")

    def reset(self):
        self.scheduling = True

    async def schedule(self, application_id: str, request_id: str) -> Optional[str]:
        """
        Pick the server address with the smallest usage count and increment its counter.
        """
        while self.scheduling:
            async with self._lock:
                dp_addr = self.get_schedule_result(application_id, False)
            if dp_addr is not None:
                self.application_id_to_request_id[application_id] = request_id
                return dp_addr
            await asyncio.sleep(1)
        # Terminate inference
        return None

    async def release(self, addr: str, application_id: str, request_id: str) -> None:
        """
        Decrement the usage count for a server address when done.
        """
        self.application_id_to_request_id.pop(application_id)
        async with self._lock:
            self.release_address(addr, application_id)

    async def cancel_requests(self):
        self.scheduling = False
        for dp_addr, req_list in self.running_reqs.items():
            if len(req_list) > 0:
                address, _ = dp_addr.split('-')
                req_address = f"{DEFAULT_URL_METHOD}://{address}/v1"
                async with aiohttp.ClientSession() as session:
                    await session.post(f"{req_address}/cancel", json={"requests": req_list},
                                       timeout=aiohttp.ClientTimeout(total=60))
                logger.info(f"cancel request {req_list}")
                async with self._lock:
                    self.running_reqs[dp_addr] = []

    async def cancel_request(self, application_id):
        dp_addr = self.application_id_to_ins[application_id]
        if dp_addr is None:
            dp_addr_rank = self.application_id_to_dp[application_id]
            dp_addr = dp_addr_rank.split('-')[0]
        request_id = self.application_id_to_request_id[application_id]

        logger.info(f"scheduler cancel request for application_id: {application_id}, "
                    f"request_id: {request_id}, dp_addr: {dp_addr}")

        req_address = f"{DEFAULT_URL_METHOD}://{dp_addr}/v1"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{req_address}/cancel", json={"requests": request_id},
                                   timeout=aiohttp.ClientTimeout(total=60))
        except Exception as e:
            logger.error(f"scheduler cancel request failed: {e}")


class SimpleTrajScheduler(SchedulerBase):
    """
    Scheduling work:
    1. When a trajectory is assigned to a DP, all steps in subsequent epochs are bound to that DP, ensuring affinity.
    """
    def __init__(self, addresses: list[str], dp_size: int = 1, workload_inf=None, role=""):
        super().__init__(addresses, dp_size, workload_inf, role)
        self._usage: dict[str, int] = {}
        # Initialize usage counts for any new addresses
        for addr in self.ins_address:
            for dp_rank in range(self.dp_size):
                dp_addr = addr + '-' + str(dp_rank)
                if dp_addr not in self._usage:
                    self._usage[dp_addr] = 0

    def get_best_instance(self, application_id: str) -> str:
        return ""

    def get_best_dp(self, ins_addr: str, application_id: str) -> int:
        return 0

    def get_schedule_result(self, application_id: str, force: bool = False) -> Optional[str]:
        """
        Pick the server address with the smallest usage count and increment its counter.
        """
        min_address, min_usage = min(self._usage.items(), key=lambda x: x[1])
        if application_id not in self.application_id_to_dp:
            self.application_id_to_dp[application_id] = min_address
            self._usage[min_address] += 1
        else:
            # Data locality
            cur_address = self.application_id_to_dp[application_id]
            cur_usage = self._usage[cur_address]
            # Load balance if there is skew
            if (min_usage == 0 or cur_usage - min_usage >= 4) and cur_usage > 0:
                self.application_id_to_dp[application_id] = min_address
                self._usage[min_address] += 1
            else:
                self._usage[cur_address] += 1
        dp_addr = self.application_id_to_dp[application_id]
        ins_addr, dp_rank = dp_addr.split('-')
        self.workload.ins_loads[ins_addr].dp_loads[dp_rank].add_req()

        return dp_addr

    def release_address(self, addr: str, application_id: str) -> None:
        """
        Decrement the usage count for a server address when done.
        """
        ins_addr, dp_rank = addr.split('-')
        self._usage[addr] = max(0, self._usage.get(addr, 0) - 1)
        self.workload.ins_loads[ins_addr].dp_loads[dp_rank].del_req()


class LBStepScheduler(SchedulerBase):
    """
    Scheduling work:
    Load balancing scheduling at trajectory step granularity
    """

    def __init__(self, addresses: list[str], dp_size: int = 1, workload_inf=None, role=""):
        super().__init__(addresses, dp_size, workload_inf, role)

    def get_best_instance(self, application_id: str) -> str:
        # Get the instance previously used by the current application_id
        cached_ins = self.application_id_to_ins.get(application_id, "")

        # Select the instance with the smallest load, preferring cached instances if loads are equal
        ins_items = self.workload.ins_loads.items()
        min_ins, ins_workload = min(
            ins_items,
            key=lambda x: (
                x[1].get_load_score(),           # Primary sort key: load score
                0 if x[0] == cached_ins else 1   # Secondary sort key: prefer cached instance
            )
        )
        return min_ins

    def get_best_dp(self, ins_addr: str, application_id: str) -> str:
        # Get the DP previously used by the current application_id (if exists)
        cached_dp = self.application_id_to_dp.get(application_id, "")

        # Select the DP with the smallest load, preferring cached DPs if loads are equal
        dp_items = self.workload.ins_loads[ins_addr].dp_loads.items()
        min_dp, dp_workload = min(
            dp_items,
            key=lambda x: (
                x[1].get_load_score(),          # Primary sort key: load score
                0 if x[0] == cached_dp else 1   # Secondary sort key: prefer cached DP
            )
        )
        return min_dp

    def get_schedule_result(self, application_id: str, force: bool = False) -> Optional[str]:
        ins_addr = self.get_best_instance(application_id)
        dp_rank = self.get_best_dp(ins_addr, application_id)
        dp_workload = self.workload.ins_loads[ins_addr].dp_loads[dp_rank]
        if dp_workload.get_load_score() >= self.workload.ins_loads[ins_addr].max_num_seqs and (not force):
            # System is fully loaded, waiting!!!
            return None
        dp_workload.add_req()
        self.application_id_to_ins[application_id] = ins_addr
        self.application_id_to_dp[application_id] = dp_rank
        return f"{ins_addr}-{dp_rank}"

    def release_address(self, dp_addr: str, application_id: str) -> None:
        ins_addr, dp_rank = dp_addr.split('-')
        self.workload.ins_loads[ins_addr].dp_loads[dp_rank].del_req()

class LBTrajScheduler(SchedulerBase):
    """
    Scheduling work:
    Load balancing scheduling at trajectory granularity. Each step of a trajectory is bound to a DP. If the trajectory is executing a tool causing the inference request to be idle, this position is still reserved.
    """

    def __init__(self, addresses: list[str], dp_size: int = 1, workload_inf=None, role=""):
        super().__init__(addresses, dp_size, workload_inf, role)

    def get_best_instance(self, prompt_id: str) -> str:
        # Get the list of instances previously used by the current prompt_id
        cached_addr = self.prompt_id_to_dps.get(prompt_id, [])
        cache_ins = [item.split('-')[0] for item in cached_addr]
        # Select the instance with the smallest load, preferring cached instances if loads are equal
        ins_items = self.workload.ins_loads.items()
        min_ins, ins_workload = min(
            ins_items,
            key=lambda x: (
                x[1].get_load_score(),           # Primary sort key: load score
                0 if x[0] in cache_ins else 1   # Secondary sort key: prefer cached instance
            )
        )
        return min_ins

    def get_best_dp(self, ins_addr: str, prompt_id: str) -> str:
        # Get the list of DPs previously used by the current prompt_id
        cached_addr = self.prompt_id_to_dps.get(prompt_id, [])
        cache_dp = [item.split('-')[1] for item in cached_addr]

        # Select the DP with the smallest load, preferring cached DPs if loads are equal
        dp_items = self.workload.ins_loads[ins_addr].dp_loads.items()
        min_dp, dp_workload = min(
            dp_items,
            key=lambda x: (
                x[1].get_load_score(),          # Primary sort key: load score
                0 if x[0] in cache_dp else 1    # Secondary sort key: prefer cached DP
            )
        )
        return min_dp

    def get_schedule_result(self, application_id: str, force: bool = False) -> Optional[str]:
        # Get the sample group corresponding to each trajectory (one of n_sample)
        prompt_id = application_id.split('-', 1)[0]
        if prompt_id not in self.prompt_id_to_dps.keys():
            self.prompt_id_to_dps[prompt_id] = set()

        if application_id not in self.application_id_to_dp:
            # Hit the DP instance where the same prompt group is located
            dp_adders = self.prompt_id_to_dps[prompt_id]
            for dp_addr in dp_adders:
                ins_addr, dp_rank = dp_addr.split('-')
                max_num_seqs = self.workload.ins_loads[ins_addr].max_num_seqs
                dp_workload = self.workload.ins_loads[ins_addr].dp_loads[dp_rank]
                # When batch issuing tasks at the initial iteration, prioritize ensuring that all trajectories in the same sample group run on the same DP
                if ((dp_workload.get_load_score() < max_num_seqs)
                    and (dp_workload.num_routing_reqs > dp_workload.num_running_reqs)):
                    dp_workload.add_req()
                    self.application_id_to_dp[application_id] = dp_addr
                    self.prompt_id_to_dps[prompt_id].add(dp_addr)
                    return dp_addr
            min_ins_addr = self.get_best_instance(prompt_id)
            min_dp_rank = self.get_best_dp(min_ins_addr, prompt_id)
            min_dp_workload = self.workload.ins_loads[min_ins_addr].dp_loads[min_dp_rank]
            max_num_seqs = self.workload.ins_loads[min_ins_addr].max_num_seqs
            dp_addr = min_ins_addr + '-' + min_dp_rank
            # If the number of requests being executed or trajectories being run by the DP with the smallest load >= maximum DP concurrency
            if (min_dp_workload.get_load_score() >= max_num_seqs) and (not force):
                # System is fully loaded, waiting!!!
                return None
            min_dp_workload.add_req()
            self.application_id_to_dp[application_id] = dp_addr
            self.prompt_id_to_dps[prompt_id].add(dp_addr)
            return dp_addr
        else:
            # Trajectory hits the same DP
            ins_addr, dp_rank = self.application_id_to_dp[application_id].split('-')
            dp_workload = self.workload.ins_loads[ins_addr].dp_loads[dp_rank]
            max_num_seqs = self.workload.ins_loads[ins_addr].max_num_seqs
            # If the hit DP is busy, select a DP with lighter load
            if dp_workload.get_load_score() >= (max_num_seqs * 1.1):
                min_ins_addr = self.get_best_instance(prompt_id)
                min_dp_rank = self.get_best_dp(min_ins_addr, prompt_id)
                min_dp_workload = self.workload.ins_loads[min_ins_addr].dp_loads[min_dp_rank]
                if min_dp_workload.get_load_score() < (max_num_seqs * 0.5):
                    min_dp_workload.add_req()
                    dp_addr = min_ins_addr + '-' + min_dp_rank
                    self.prompt_id_to_dps[prompt_id].add(dp_addr)
                    return dp_addr
                elif not force:
                    # System is heavily loaded, waiting!!!
                    return None
            dp_workload.add_req()
            dp_addr = self.application_id_to_dp[application_id]
            self.prompt_id_to_dps[prompt_id].add(dp_addr)
            return dp_addr

    def release_address(self, dp_addr: str, application_id: str) -> None:
        ins_addr, dp_rank = dp_addr.split('-')
        self.workload.ins_loads[ins_addr].dp_loads[dp_rank].del_req()

    def reset(self):
        super().reset()
        self.application_id_to_dp.clear()
        self.prompt_id_to_dps.clear()


class SchedulerFactory:
    _registry = {}

    @classmethod
    def register(cls, name, scheduler_class):
        cls._registry[name] = scheduler_class

    @classmethod
    def get_scheduler(cls, addresses: list[str], dp_size, workload_inf=None, role=""):
        scheduler_algo = os.getenv("SCHEDULER_ALGO", "lb-step")
        """Get scheduler instance based on configuration"""
        scheduler_class = cls._registry.get(scheduler_algo)
        if not scheduler_class:
            raise ValueError(f"Unknown scheduler: {scheduler_algo}")
        logger.info(f"{role} scheduler_algo is {scheduler_algo}")
        return scheduler_class(addresses, dp_size, workload_inf, role)


# Register schedulers
SchedulerFactory.register('simple-traj', SimpleTrajScheduler)
SchedulerFactory.register('lb-step', LBStepScheduler)
SchedulerFactory.register('lb-traj', LBTrajScheduler)