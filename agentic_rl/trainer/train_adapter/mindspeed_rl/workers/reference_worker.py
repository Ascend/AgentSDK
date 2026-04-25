# -*- coding: utf-8 -*-
#
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
# 
import time

import ray
import torch

from mindspeed_rl.trainer.utils.parallel_state import (
    get_context_parallel_rank,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.utils.pad_process import truncate_rows
from mindspeed_rl.utils.utils import is_multimodal, mstx_timer_decorator
from mindspeed_rl.workers.reference_woker import ReferenceWorkerBase

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class ReferenceWorkerBasePatch(ReferenceWorkerBase):
    """Patched reference worker that dispatches experience batches and collects ref log probs."""

    @mstx_timer_decorator
    def compute_ref_log_prob(self) -> None:
        """Consume experience batches and compute reference-model log probabilities."""
        experience_consumer_stage = 'ref_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length']
        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids', 'input_ids_length'])
        experience_count = self.rl_config.ref_dispatch_size
        sorted_indexes = self.get_dp_range_indexes(
            experience_count, use_vllm=False) if self.rl_config.guarantee_order else None

        start_time_defined = False
        first_dispatch_data_defined = False
        first_collect_data_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            if not first_dispatch_data_defined:
                first_dispatch_start_time = time.time()
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                cp_size=self.megatron_config.context_parallel_size,
                cp_algo=self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                get_n_samples=self.rl_config.partial_rollout_max_split > 1
            )

            if batch_data and index:
                if not first_dispatch_data_defined:
                    ray.get(self.td.update_metrics.remote(
                        "dispatch_timing(first)/reference_model",
                        value=[round(time.time(), 4), round(first_dispatch_start_time, 4)],
                        cumulate=True
                    ))
                    first_dispatch_data_defined = True

                if not start_time_defined:
                    start_time = time.time()
                    start_time_defined = True
                    ray.get(
                        self.td.update_metrics.remote(
                            "start_time/reference_model",
                            value=[round(start_time, 4)],
                            cumulate=True
                        )
                    )
                output, batch = self.reference.compute_log_prob(batch_data)

                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'ref_log_prob': log_probs}

                    if not first_collect_data_defined:
                        first_collect_start_time = time.time()

                    self.collect_transfer_dock_data(output, index)

                    if not first_collect_data_defined:
                        ray.get(self.td.update_metrics.remote(
                            "collect_timing(first)/reference_model",
                            value=[time.time() - first_collect_start_time],
                            cumulate=True
                        ))
                        first_collect_data_defined = True

                    end_time = time.time()
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/reference_model",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        parallel_state = get_parallel_state()
        use_vllm = False
        if (is_pipeline_last_stage(parallel_state, use_vllm)
                and get_tensor_model_parallel_rank(parallel_state, use_vllm) == 0
                and self.parallel_state.get_context_parallel_rank() == 0):
            ref_end_time = time.time()
            ray.get(
                self.td.update_metrics.remote(
                    "end_time/reference",
                    value=[round(ref_end_time, 4)]
                )
            )
        logger.info("finish compute ref log prob")


@ray.remote(resources={"NPU": 0.3})
class ReferenceWorker(ReferenceWorkerBasePatch):
    """Ray remote reference worker bound to an NPU resource."""