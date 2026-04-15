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

import json
import os
import time
from datetime import datetime
from typing import Dict


class RequestStat:
    def __init__(
        self,
        request_id: str = "",
        add_tick: float = 0.0,
        schedule_tick: float = 0.0,
        prefill_done_tick: float = 0.0,
        finish_tick: float = 0.0
    ):
        self.request_id = request_id
        self.add_tick = add_tick
        self.schedule_tick = schedule_tick
        self.prefill_done_tick = prefill_done_tick
        self.finish_tick = finish_tick
        self.prompt_len: int = 0
        self.output_len: int = 0

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "add_tick": self.add_tick,
            "schedule_tick": self.schedule_tick,
            "prefill_done_tick": self.prefill_done_tick,
            "finish_tick": self.finish_tick,
            "prompt_len": self.prompt_len,
            "output_len": self.output_len
        }


class RequestStats:
    index_: int = 0
    enabled: bool = (
        os.environ.get('GTS_STATS_ENABLE') is None or 
        os.environ.get('GTS_STATS_ENABLE') == "1"
    )

    def __init__(self):
        self.req_stats: Dict[str, RequestStat] = {}

    def stat_add(self, request_id: str):
        if not RequestStats.enabled:
            return
        self.req_stats[request_id] = RequestStat(add_tick=time.time())

    def stat_schedule(self, request_id: str):
        if not RequestStats.enabled:
            return
        self.req_stats[request_id].schedule_tick = time.time()

    def stat_prefill_done(self, request_id: str):
        if not RequestStats.enabled:
            return
        self.req_stats[request_id].prefill_done_tick = time.time()

    def stat_finish(self, request_id: str, prompt_len: int = 0, output_len: int = 0):
        if not RequestStats.enabled:
            return
        self.req_stats[request_id].finish_tick = time.time()
        self.req_stats[request_id].prompt_len = prompt_len
        self.req_stats[request_id].output_len = output_len

    def reset(self):
        self.req_stats = {}

    def print(self):
        if not RequestStats.enabled:
            return
        stats_data = {
            "timestamp": time.time(),
            "request": {
                id: stat.to_dict()
                for id, stat in self.req_stats.items()
            }
        }
        pid = os.getpid()
        today = datetime.now().strftime('%Y%m%d')
        cur_dir_path = os.path.join('logs/vllm_schedule', today)
        try:
            if not os.path.exists(cur_dir_path):
                os.makedirs(cur_dir_path, exist_ok=True)
        except Exception as e:
            print(f"Failed to create dir{cur_dir_path}: {str(e)}")
        filename = os.path.join(cur_dir_path, f"vllm_schedule_{RequestStats.index_}_{pid}_{int(time.time())}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            print(f"Statistics saved to {filename}")
        except Exception as e:
            print(f"Failed to save statistics: {str(e)}")

        self.reset()
        RequestStats.index_ += 1
