#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import torch
import ray
from mindspeed_rl.trainer.grpo_trainer_hybrid import GRPOTransferDock as _GRPOTransferDock

from agentic_rl.base.log.loggers import Loggers


logger = Loggers(__name__)


# GRPOTransferDock is wrapped by ray.remote, so we need to get the real class
try:
    raw_grpo_transferred_dock_cls = _GRPOTransferDock.__ray_metadata__.modified_class
except AttributeError as attribute_error:
    logger.error(f"Unable to get GRPOTransferDock from mindspeed_rl: {attribute_error}")
    raise AttributeError("Unable to get GRPOTransferDock from mindspeed_rl.") from attribute_error


@ray.remote(max_concurrency=100, num_cpus=10)
class GRPOTransferDock(raw_grpo_transferred_dock_cls):

    def reset_experience_len(self, max_len):
        """Reset experience len and experience data"""
        if not isinstance(max_len, int):
            raise ValueError(f"max_len should be int, got {type(max_len)}.")
        self.max_len = max_len
        self.experience_data = {
            key: [None for _ in range(self.max_len)]
            for key in self.experience_columns
        }
        self.experience_data_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_columns
        }
        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_consumers
        }
        self.n_samples_per_prompt = self.max_len // self.prompts_num