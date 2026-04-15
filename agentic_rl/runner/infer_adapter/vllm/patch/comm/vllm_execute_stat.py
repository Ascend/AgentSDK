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
import socket
import time
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List

import pandas as pd
import torch
from vllm.logger import logger


class StatTimeUtil:
    def __init__(self):
        self.last_time = time.time()

    def get_duration(self, is_npu_exist=True):
        if is_npu_exist:
            torch.npu.synchronize()
        
        current_time = time.time()
        duration = current_time - self.last_time
        self.last_time = current_time
        return duration * 1000


class StatPhase(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return 0 + count

    step_start_time = auto()
    step_finished_time = auto()

    prepare_input_time = auto()
    aclgraph_dispatcher_time = auto()
    forward_time = auto()
    kvconnectoroutput_time = auto()
    post_process_time = auto()
    pop_captured_sync_time = auto()

    step_total_time = auto()
    step_inter_time = auto()

    forward_init_metadata_time = auto()
    forward_embedding_time = auto()
    forward_alllayers_time = auto()
    forward_last_norm_time = auto()
    forward_metadata_unpadding_time = auto()

    post_process_compute_logits_time = auto()
    post_process_sampler_time = auto()
    post_process_other_time = auto()

    with_prefill = auto()
    attn_state = auto()
    batch_num = auto()
    num_actual_tokens = auto()
    seq_lens = auto()
    is_dummy_run = auto()
    is_profiling = auto()


class _VllmOutputStatics:
    def __init__(self):
        self.stats: Dict[str, List[float]] = {}
        self.stats["title"] = [phase.name for phase in StatPhase]
        self.last_step_finish_time = 0
        self.step_start_time= 0
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.process_name = (
            f"{self.local_ip} IntegratedWorker pid={os.getpid()}"
        )
        self.cur_requestid_stepid = ""
        self.base_path = "logs/vllm_statistic"

    def set_process_name(self, process_name: str) -> None:
        """Set the process name for statistics identification.
        
        Args:
            process_name: Name to identify the process.
        """
        self.process_name = f"{self.local_ip} {process_name} pid={os.getpid()}"

    def set_cur_requestid_stepid(self, request_step_id: str, start_time: float):
        self.step_start_time = start_time
        self.cur_requestid_stepid = f"{self.process_name}/{request_step_id}"
        
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
            self.stats[self.cur_requestid_stepid][StatPhase.is_profiling.value] = False
            self.stats[self.cur_requestid_stepid][StatPhase.is_dummy_run.value] = False

        if self.last_step_finish_time > 0:
            inter_time = (start_time - self.last_step_finish_time) * 1000
            self.stats[self.cur_requestid_stepid][StatPhase.step_inter_time.value] = inter_time
        
        self.stats[self.cur_requestid_stepid][StatPhase.step_start_time.value] = start_time

    def set_step_finish_time(self, finish_time: float):
        self.last_step_finish_time = finish_time
        
        if self.cur_requestid_stepid in self.stats:
            total_time = (finish_time - self.step_start_time) * 1000
            self.stats[self.cur_requestid_stepid][StatPhase.step_total_time.value] = total_time
            self.stats[self.cur_requestid_stepid][StatPhase.step_finished_time.value] = finish_time

    def add_stat(self, stat_phase: StatPhase, duration_time: float):
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
        self.stats[self.cur_requestid_stepid][stat_phase.value] = duration_time

    def set_stat(self, stat_phase: StatPhase, value: float):
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
        self.stats[self.cur_requestid_stepid][stat_phase.value] = value

    def print_stats(self):
        if not is_vllm_statistic:
            return
        
        print(
            f"print_stats len: {len(self.stats)}, "
            f"process_name: {self.process_name}"
        )
        
        if len(self.stats) > 1:
            print("_VllmOutputStatics:", self.stats)

    def print_one_stats(self):
        if not is_vllm_statistic:
            return
        
        if self.cur_requestid_stepid in self.stats:
            print(
                f"_VllmOutputStatics cur_request-id_step-id: "
                f"{self.cur_requestid_stepid} : "
                f"{self.stats[self.cur_requestid_stepid]}"
            )

    def write_stats_tofile(self):
        if not is_vllm_statistic:
            return

        if len(self.stats) > 1:
            df = pd.DataFrame(self.stats).set_index('title').transpose().reset_index()
            df = df.rename(columns={'index': 'title'})

            today = datetime.now().strftime('%Y%m%d')
            if vllm_stat_save_path_suffix != " ":
                today = f"{today}_{vllm_stat_save_path_suffix}"
            cur_dir_path = os.path.join(self.base_path, today)
            try:
                if not os.path.exists(cur_dir_path):
                    os.makedirs(cur_dir_path, exist_ok=True)
            except Exception as e:
                logger.warn(f"Failed to create dir{cur_dir_path}: {str(e)}")
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_name = f"{self.process_name}-{formatted_time}.csv"
            file_name = os.path.join(cur_dir_path, file_name)
            df.to_csv(file_name, index=False)

    def clear(self):
        if not is_vllm_statistic:
            return
        
        self.stats.clear()
        self.stats["title"] = [phase.name for phase in StatPhase]
        self.last_step_finish_time = 0
        self.step_start_time = 0
        self.cur_requestid_stepid = ""


is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
vllm_stat_save_path_suffix = os.environ.get("VLLM_STAT_SVAE_PATH_SUFFIX", " ")

vllm_output_statics = _VllmOutputStatics()
