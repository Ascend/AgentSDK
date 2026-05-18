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


def get_container_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Here we use an external address that does not need to be actually reachable (such as Google DNS)
        # As long as the address format is correct, the kernel will select the corresponding local IP according to the routing table
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        # 如果断网，兜底回退到 localhost
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class StatPhase(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return 0 + count

    step_start_time = auto()
    step_finished_time = auto()
    step_total_time = auto()
    step_inter_time = auto()

    prepare_input_time = auto()
    aclgraph_dispatcher_time = auto()
    forward_time = auto()
    kvconnectoroutput_time = auto()
    post_process_time = auto()
    pop_captured_sync_time = auto()

    prepare_remove_reqs_time = auto()
    prepare_add_reqs_time = auto()
    prepare_update_states_time = auto()
    prepare_other_states_time = auto()

    prepare_copy_bt_time = auto()
    prepare_get_tokens_time = auto()
    prepare_pad_tokens_time = auto()
    prepare_sync_meta_time = auto()
    prepare_set_lora_time = auto()
    prepare_pos_cpu_time = auto()
    prepare_mrope_time = auto()
    prepare_pos_npu_time = auto()
    prepare_slot_map_time = auto()
    prepare_atten_mask_time = auto()
    prepare_seq_len_time = auto()
    prepare_attn_meta_time = auto()
    prepare_inputids_cpu_time = auto()
    prepare_copy_inputids_time = auto()
    prepare_inputsembeds_time = auto()
    prepare_slice_inputids_time = auto()
    prepare_update_ids_and_pos_time = auto()
    prepare_inter_tensors_time = auto()
    prepare_logits_indice_time = auto()
    prepare_specdeco_meta_time = auto()
    prepare_lmhead_logits_indices_time = auto()

    forward_init_metadata_time = auto()
    forward_embedding_time = auto()
    forward_alllayers_time = auto()
    forward_last_norm_time = auto()
    forward_metadata_unpadding_time = auto()

    post_process_compute_logits_time = auto()
    post_process_sampler_time = auto()
    post_process_other_time = auto()

    post_samper_logits_slice_time = auto()
    post_samper_compute_logprobs_time = auto()
    post_samper_logits_preproc_time = auto()
    post_samper_processor_apply_time = auto()
    post_samper_apply_penalties_time = auto()
    post_samper_sample_next_token_time = auto()
    post_samper_sampled_long_time = auto()
    post_samper_gather_logprobs_time = auto()
    post_samper_sampled_int32_time = auto()

    post_samper_sample_greedy_time = auto()
    post_samper_sample_apply_temperature_time = auto()
    post_samper_sample_processor_apply_again_time = auto()
    post_samper_sample_topk_topp_time = auto()
    post_samper_sample_greedy_where_time = auto()

    post_samper_sample_topk_topp_apply_time = auto()
    post_samper_sample_topk_topp_logits_log_softmax_time = auto()
    post_samper_sample_topk_topp_probs_softmax_time = auto()
    post_samper_sample_topk_topp_random_sample_time = auto()

    with_prefill = auto()
    attn_state = auto()
    batch_num = auto()
    num_actual_tokens = auto()
    max_query_len = auto()
    seq_lens = auto()
    is_dummy_run = auto()
    is_profiling = auto()


class _VllmOutputStatics:
    def __init__(self):
        self.stats: Dict[str, List[float]] = {}
        self.stats["title"] = [phase.name for phase in StatPhase]
        self.last_step_finish_time = 0
        self.step_start_time = 0
        self.local_ip = get_container_ip()
        self.process_name = self.local_ip + " " + "IntegratedWorker" + " pid=" + str(os.getpid())
        self.cur_requestid_stepid = ""
        self.base_path = "logs/vllm_statistic"

    def set_process_name(self, process_name: str) -> None:
        """Set the process name for statistics identification.

        Args:
            process_name: Name to identify the process.
        """
        self.process_name = self.local_ip + " " + process_name + " pid=" + str(os.getpid())

    def set_cur_requestid_stepid(self, cur_requestid_stepid, start_time: float):
        self.step_start_time = start_time
        self.cur_requestid_stepid = self.process_name + "/" + cur_requestid_stepid
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
            self.stats[self.cur_requestid_stepid][StatPhase.is_profiling.value] = False
            self.stats[self.cur_requestid_stepid][StatPhase.is_dummy_run.value] = False

        if self.last_step_finish_time > 0:
            self.stats[self.cur_requestid_stepid][StatPhase.step_inter_time.value] = (
                start_time - self.last_step_finish_time
            ) * 1000  # ms
        self.stats[self.cur_requestid_stepid][StatPhase.step_start_time.value] = start_time

    def set_step_finish_time(self, finish_time: float):
        self.last_step_finish_time = finish_time
        if self.cur_requestid_stepid in self.stats:
            self.stats[self.cur_requestid_stepid][StatPhase.step_total_time.value] = (
                finish_time - self.step_start_time
            ) * 1000  # ms
            self.stats[self.cur_requestid_stepid][StatPhase.step_finished_time.value] = finish_time  # tick

    def add_stat(self, stat_phaseid: StatPhase, duration_time: float):
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
        self.stats[self.cur_requestid_stepid][stat_phaseid.value] = duration_time

    def set_stat(self, stat_phaseid: StatPhase, value):
        if self.cur_requestid_stepid not in self.stats:
            self.stats[self.cur_requestid_stepid] = [0] * len(StatPhase)
        self.stats[self.cur_requestid_stepid][stat_phaseid.value] = value

    def print_stats(self):
        if not is_vllm_statistic:
            return
        print("print_stats len:", len(self.stats), " self.process_name:", self.process_name)
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
        v = os.getenv('ENABLE_VLLM_STAT', "False")
        print(
            f"write_stats_tofile is_vllm_statistic={v} vllm_stat_save_path_suffix={vllm_stat_save_path_suffix} num_data={len(self.stats)}"
        )
        if not is_vllm_statistic:
            return

        if len(self.stats) > 1:
            df = (
                pd.DataFrame(self.stats).set_index('title').transpose().reset_index().rename(columns={'index': 'title'})
            )
            self.clear()

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
vllm_stat_save_path_suffix = os.environ.get("VLLM_STAT_SAVE_PATH_SUFFIX", " ")

vllm_output_statics = _VllmOutputStatics()
