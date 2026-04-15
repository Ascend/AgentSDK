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
import asyncio
import json
import traceback
import threading
from collections import deque
from typing import Dict

# Third-party library imports
import aiohttp

# Internal imports
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

DEFAULT_WORKLOAD_PRINT_CNT = 3

class DPWorkLoad:
    def __init__(self):
        self.num_routing_reqs: int = 0  # Requests being scheduled in router, not yet in inference engine
        self.num_running_reqs: int = 0
        self.num_waiting_reqs: int = 0
        self.prompt_throughput: float = 0
        self.generation_throughput: float = 0
        self.ttft: int = 0
        self.tpot: int = 0
        self.prefill_tokens: int = 0
        self.decode_tokens: int = 0
        self.kv_cache_usage: float = 0
        self.prefixcache_hit_rate: float = 0
        self.history = deque(maxlen=1000)

    def to_dict(self) -> dict:
        return {
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
            "num_routing_reqs": self.num_routing_reqs,
            "prompt_throughput": self.prompt_throughput,
            "generation_throughput": self.generation_throughput,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "kv_cache_usage": self.kv_cache_usage,
            "prefixcache_hit_rate": self.prefixcache_hit_rate,
        }

    def update(self, data: dict):
        self.num_running_reqs = data.get("num_running_reqs", 0)
        self.num_waiting_reqs = data.get("num_waiting_reqs", 0)
        self.prompt_throughput = data.get("prompt_throughput", 0.0)
        self.generation_throughput = data.get("generation_throughput", 0.0)
        self.tpot = data.get("tpot", 0)
        self.ttft = data.get("ttft", 0)
        self.kv_cache_usage = data.get("kv_cache_usage", 0)
        self.prefixcache_hit_rate = data.get("prefixcache_hit_rate", 0)
        self.prefill_tokens = data.get("prefill_tokens", 0)
        self.decode_tokens = data.get("decode_tokens", 0)
        # Routing requests need to be cleared to avoid duplication with num_running_reqs + num_waiting_reqs, which would cause scheduling failures
        self.num_routing_reqs = data.get("num_routing_reqs", 0)

    def add_req(self):
        self.num_routing_reqs += 1

    def del_req(self):
        self.num_routing_reqs = max(0, self.num_routing_reqs - 1)

    def get_load_score(self) -> float:
        return self.num_running_reqs + self.num_waiting_reqs + self.num_routing_reqs

    def add_to_history(self):
        self.history.append(self.to_dict().copy())


class InstanceWorkLoad:
    def __init__(self, dp_address: list[str] = None, dp_size: int = 0):
        self.dp_loads: dict[str, DPWorkLoad] = {}
        self.num_free_dp: int = 0
        self.max_num_seqs: int = 8
        if dp_address is not None:
            for dp_id in dp_address:
                self.dp_loads[dp_id] = DPWorkLoad()
        else:
            for dp_id in range(dp_size):
                self.dp_loads[str(dp_id)] = DPWorkLoad()

    def to_dict(self) -> dict:
        return {
            "dp_loads": {dp_id: dp_load.to_dict() for dp_id, dp_load in self.dp_loads.items()},
            "max_num_seqs": self.max_num_seqs,
        }

    def update(self, ins_addr: str, data: dict):
        # logger.info(f"update {ins_addr}, {data=}")
        data = data if isinstance(data, dict) else {}
        dp_loads_data = data.get("dp_loads", {})
        for dp_id, dp_load_data in dp_loads_data.items():
            if dp_id not in self.dp_loads:
                self.dp_loads[dp_id] = DPWorkLoad()
            self.dp_loads[dp_id].update(dp_load_data)
        self.num_free_dp = sum(1 if load.get_load_score() == 0 else 0 for load in self.dp_loads.values())
        self.max_num_seqs = data.get("max_num_seqs", 8)

    def get_load_score(self) -> float:
        return sum(dp_load.get_load_score() for dp_load in self.dp_loads.values())

    def add_to_history(self):
        # Add history for each DP
        for dp in self.dp_loads.values():
            dp.add_to_history()

class WorkLoadManger:
    def __init__(self, addresses: list[str], dp_size: int, role=""):
        self.ins_loads: dict[str, InstanceWorkLoad] = {addr: InstanceWorkLoad([], dp_size) for addr in addresses}
        self.num_free_ins: int = len(addresses)
        self.role = role

    def to_dict(self) -> dict:
        """Convert WorkLoadManger instance to dictionary"""
        return {
            "role": self.role,
            "ins_loads": {addr: ins_load.to_dict() for addr, ins_load in self.ins_loads.items()},
            "num_free_ins": self.num_free_ins,
        }

    def update(self, data: dict):
        for ins_addr, ins_data in data.items():
            if ins_data is not None:
                self.ins_loads[ins_addr].update(ins_addr, ins_data)
        self.num_free_ins = sum(1 if load.get_load_score() == 0 else 0 for load in self.ins_loads.values())

    def add_to_history(self, data):
        for ins_load in self.ins_loads.values():
            ins_load.add_to_history()

    def save_history_to_json(self, filename: str):
        history_data = {}
        for ins_addr, ins_load in self.ins_loads.items():
            history_data[ins_addr] = {}
            for dp_id, dp_load in ins_load.dp_loads.items():
                history_data[ins_addr][dp_id] = list(dp_load.history)  # 转换为普通 list 以支持 JSON 序列化

        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=4)

async def poll_workload_openai(session: aiohttp.ClientSession, address: str) -> str:
    """
    Asynchronously polls the workload endpoint.
    Uses the passed session to make the request.
    """
    base_url = f"http://{address}/v1/workload"
    headers = {
        "Content-Type": "application/json",
    }
    try:
        async with session.get(base_url, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")

            # Correctly read JSON from aiohttp response
            result = json.loads(await response.json())
            return result
    except Exception as e:
        logger.error(f"Error polling {address}: {e}")
        raise


async def poll_all_instances(session: aiohttp.ClientSession, workloads: WorkLoadManger) -> Dict[str, str]:
    """
    Gathers workload data from all instances concurrently.
    """
    tasks = []
    for instance_address, _ in workloads.ins_loads.items():
        task = poll_workload_openai(session, instance_address)
        tasks.append(task)

    # Run all tasks concurrently and wait for them to complete
    # results will be a list of strings (the results from each poll)
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Map results back to instance addresses
        res_dict = {}
        for i, instance_address in enumerate(workloads.ins_loads.keys()):
            result = results[i]
            if isinstance(result, Exception):
                print(f"[ERROR] Exception for {instance_address}: {result}")
                res_dict[instance_address] = f"ERROR: {result}"
            else:
                res_dict[instance_address] = result
        return res_dict
    except Exception as e:
        print(f"[ERROR] Exception in poll_all_instances: {e}")
        return {}

async def workload_update_periodically(self, workloads: WorkLoadManger):
    """
    The main asynchronous loop that updates workload periodically.
    """
    interval_str = os.getenv("VLLM_LOG_STATS_INTERVAL", "10")
    print_stats = bool(int(os.getenv("LOG_WORKLOAD_ENABLE", "1")))
    try:
        interval = float(interval_str)
    except ValueError:
        print(f"[WARNING] Invalid VLLM_LOG_STATS_INTERVAL '{interval_str}', using default 10.0 seconds.")
        interval = 10.0
    print_cnt = DEFAULT_WORKLOAD_PRINT_CNT

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                res_dict = await poll_all_instances(session, workloads) if self is None else await self.get_workload()
                workloads.update(res_dict)
                if print_stats:
                    if workloads.num_free_ins < len(workloads.ins_loads):
                        print_cnt = DEFAULT_WORKLOAD_PRINT_CNT
                    if print_cnt > 0:
                        logger.info(f"workload_update_periodically res={workloads.to_dict()}")
                        print_cnt = print_cnt - 1
            except asyncio.CancelledError:
                logger.warning("[INFO] Workload update loop cancelled.")
                break # Exit loop if task is cancelled
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logger.error(f"[ERROR] Workload update loop exception: {exp}")

            await asyncio.sleep(interval)


def start_workload_update(self, workloads: WorkLoadManger):
    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(workload_update_periodically(self, workloads))
        except Exception as e:
            logger.error(f"workload_update_periodically task thread crashed: {e}")
            traceback.print_exc()
        finally:
            loop.close()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread