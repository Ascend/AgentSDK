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

import json
import os
import random
import time
import warnings
from datetime import datetime
from typing import Dict, List

import click
from PIL import Image

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = CURRENT_PATH[:CURRENT_PATH.find("/agentic_rl/base/misc")]
LOGS_PATH = ROOT_PATH + "/logs/app_stats"


def colorful_print(string: str, *args, **kwargs) -> None:
    end = kwargs.pop("end", "\n")
    print(click.style(string, *args, **kwargs), end=end, flush=True)


def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs), stacklevel=2)


def get_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")


def pad_from_left(input_id_list, pad_token_id):
    max_len = max([len(input_id) for input_id in input_id_list])
    if len(input_id_list) == 1:
        # add some randomness to the padding if the batch size is one, for better batch inference
        max_len += random.randint(1, 100)
    padded_input_ids = [[pad_token_id] * (max_len - len(input_id)) + input_id for input_id in input_id_list]
    return padded_input_ids


def merge_dicts(dict_list):
    merged_dict: dict[str, list] = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict


class RequestIdStat:
    def __init__(self, req_id: str = ""):
        self.req_id: str = req_id
        self.address: str = ""
        self.step_idx: int = 0
        self.prompt_len: int = 0
        self.terminal_reason: str = ""
        self.route_tick: float = time.time()
        self.vllm_start_tick: float = 0
        self.vllm_end_tick: float = 0
        self.env_start_tick: float = 0
        self.env_end_tick: float = 0

    def to_dict(self):
        return {
            "request_id": self.req_id,
            "address": self.address,
            "step_idx": self.step_idx,
            "prompt_len": self.prompt_len,
            "terminal_reason": self.terminal_reason,
            "route_tick": self.route_tick,
            "vllm_start_tick": self.vllm_start_tick,
            "vllm_end_tick": self.vllm_end_tick,
            "env_start_tick": self.env_start_tick,
            "env_end_tick": self.env_end_tick,
            "vllm_delay": self.vllm_end_tick - self.vllm_start_tick,
            "env_delay": self.env_end_tick - self.env_start_tick
        }


class AppIdStat:
    def __init__(self, app_id: str = ""):
        self.req_stats: dict[str, RequestIdStat] = {}
        self.app_id = app_id
        self.trajectory_id = None
        self.total_vllm_delay: float = 0
        self.total_env_delay: float = 0
        self.total_delay: float = 0

    def stat_route(self, req_id: str, address: str, prompt_len: int):
        if self.req_stats.get(req_id) is None:
            self.req_stats[req_id] = RequestIdStat(req_id=req_id)
        self.req_stats[req_id].address = address
        self.req_stats[req_id].prompt_len = prompt_len

    def stat_vllm_step(self, req_id: str, step_idx: int, start: float, end: float):
        if self.req_stats.get(req_id) is None:
            self.req_stats[req_id] = RequestIdStat(req_id=req_id)
        self.req_stats[req_id].step_idx = step_idx
        self.req_stats[req_id].vllm_start_tick = start
        self.req_stats[req_id].vllm_end_tick = end
        self.total_vllm_delay += (end - start)

    def stat_env_step(self, req_id: str, step_idx: int, start: float, end: float, terminal_reason: str):
        if self.req_stats.get(req_id) is None:
            self.req_stats[req_id] = RequestIdStat(req_id=req_id)
        self.req_stats[req_id].step_idx = step_idx
        self.req_stats[req_id].env_start_tick = start
        self.req_stats[req_id].env_end_tick = end
        self.req_stats[req_id].terminal_reason = terminal_reason
        self.total_env_delay += (end - start)

    def stat_env_state(self, req_id: str, terminal_reason: str):
        if self.req_stats.get(req_id) is None:
            self.req_stats[req_id] = RequestIdStat(req_id=req_id)
        self.req_stats[req_id].terminal_reason = terminal_reason

    def stat_trajectory(self, trajectory_id):
        self.trajectory_id = trajectory_id

    def to_dict(self):
        self.total_delay = self.total_vllm_delay + self.total_env_delay
        return {
            "application_id": self.app_id,
            "total_vllm_delay": self.total_vllm_delay,
            "total_env_delay": self.total_env_delay,
            "total_delay": self.total_delay,
            "request_count": len(self.req_stats),
            "trajectory_id": self.trajectory_id,
            "requests": [req.to_dict() for req in self.req_stats.values()]
        }


class ScheduleStat:
    def __init__(self, address: str = ""):
        self.address: str = address
        self.reqs: List[str] = []
        self.total_prompt_len: int = 0

    def stat_add(self, req_id: str, prompt_len: int):
        self.reqs.append(req_id)
        self.total_prompt_len += prompt_len

    def to_dict(self):
        return {
            "address": self.address,
            "processed_tokens": self.total_prompt_len,
            "request_count": len(self.reqs),
            "requests": self.reqs
        }


class AppStats:
    _instance = None
    enabled: bool = os.environ.get('GTS_STATS_ENABLE') is None or os.environ.get('GTS_STATS_ENABLE') == "1"

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.appid_stats = {}
            cls._instance.dp_stats = {}
            cls._instance._initialized = True
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Additional initialization logic can be added here
        self._initialized = True
        self.appid_stats: Dict[str, AppIdStat] = {}
        self.dp_stats: Dict[str, ScheduleStat] = {}

    @staticmethod
    def get_request_id(app_id: str, step_idx: int):
        # cmpl-${application_id}--${step_idx}-0
        return f"cmpl-{app_id}--{step_idx}-0"

    def stat_route(self, app_id: str, req_id: str, address: str, prompt_len: int):
        if not AppStats.enabled:
            return
        # The input "req_id" is the original ID that was manually concatenated.
        # Inside the vllm system, it will be further encapsulated.
        req_id = f"cmpl-{req_id}-0"
        if self.appid_stats.get(app_id) is None:
            self.appid_stats[app_id] = AppIdStat(app_id=app_id)
        self.appid_stats[app_id].stat_route(req_id, address, prompt_len)
        if self.dp_stats.get(address) is None:
            self.dp_stats[address] = ScheduleStat(address)
        self.dp_stats[address].stat_add(req_id, prompt_len)

    def stat_vllm_step(self, app_id: str, step_idx: int, start: float, end: float):
        if not AppStats.enabled:
            return
        if self.appid_stats.get(app_id) is None:
            self.appid_stats[app_id] = AppIdStat(app_id=app_id)
        self.appid_stats[app_id].stat_vllm_step(AppStats.get_request_id(app_id, step_idx), step_idx, start, end)

    def stat_env_step(self, app_id: str, step_idx: int, start: float, end: float, terminal_reason: str):
        if not AppStats.enabled:
            return
        if self.appid_stats.get(app_id) is None:
            self.appid_stats[app_id] = AppIdStat(app_id=app_id)
        self.appid_stats[app_id].stat_env_step(
            self.get_request_id(app_id, step_idx), step_idx, start, end, terminal_reason)

    def stat_env_state(self, app_id: str, step_idx: int, terminal_reason: str):
        if not AppStats.enabled:
            return
        if self.appid_stats.get(app_id) is None:
            self.appid_stats[app_id] = AppIdStat(app_id=app_id)
        self.appid_stats[app_id].stat_env_state(self.get_request_id(app_id, step_idx), terminal_reason)

    def stat_trajectory(self, app_id, trajectory_id):
        self.appid_stats[app_id].stat_trajectory(trajectory_id)

    def clear(self):
        self.appid_stats = {}
        self.dp_stats = {}

    def print(self, inner_iter: int = 0):
        if not AppStats.enabled:
            return
        stats_data = {
            "timestamp": time.time(),
            "iteration": inner_iter,
            "applications": {
                app_id: stat.to_dict()
                for app_id, stat in self.appid_stats.items()
            },
            "schedulers": {
                dp_addr: stat.to_dict()
                for dp_addr, stat in self.dp_stats.items()
            }
        }

        current_time = datetime.now().strftime('%Y%m%d')
        cur_dir_path = os.path.join(LOGS_PATH, current_time)
        os.makedirs(cur_dir_path, exist_ok=True)
        filename = f"{cur_dir_path}/app_stats_iter_{inner_iter}_{int(time.time())}.json"
        try:
            json_str = json.dumps(stats_data, indent=2, ensure_ascii=False)
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Statistics saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {str(e)}")
        self.clear()


app_stats = AppStats()
