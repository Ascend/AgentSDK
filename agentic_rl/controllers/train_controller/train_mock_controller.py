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
import random

import torch

from agentic_rl.controllers.train_controller.train_controller import TrainController, logger


class TrainMockController(TrainController):

    def __init__(
            self,
            actor_worker,
            actor_config,
            rl_config,
            generate_config,
            initialize_rollout_dataloader,
            consumed_train_samples,
            data_optimized
    ) -> None:
        super(TrainMockController, self).__init__(
            actor_worker,
            actor_config,
            rl_config,
            generate_config,
            initialize_rollout_dataloader,
            consumed_train_samples,
            data_optimized)
        self.rl_config = rl_config

    def wait_for_rollout_unit_ready(self):
        logger.info("wait for all rollout units ready ...")
        logger.info("all rollout units are ready!")

    def update_rollout_weights(self, iteration: int):
        logger.info(f"mock >>> update_rollout_weights success")

    def send_initial_batch_groups_to_rollout(self):
        logger.info(f"mock send_initial_batch_groups_to_rollout")

    def unlock_rollout_unit(self):
        logger.info(f"mock unlock_rollout_unit")

    def get_next_training_batch(self, last_iteration: bool = False):
        logger.info(f"waiting next training batch ...")
        prompt_bias = self.rl_config.mock_prompt_mean - self.rl_config.mock_prompt_gap // 2
        raw_prompt_length = [int(random.random() * self.rl_config.mock_prompt_gap) + prompt_bias
                             for _ in range(self.global_batch_size)]
        response_bias = self.rl_config.mock_response_mean - self.rl_config.mock_response_gap // 2
        response_length = [
            int(random.random() * self.rl_config.mock_response_gap) + response_bias
            for _ in raw_prompt_length
            for _ in range(self.n_samples_per_prompt)
        ]
        prompt_ids = [
            torch.randint(1000, 2000, (x,)) 
            for x in raw_prompt_length
        ]
        response_ids = [
            torch.cat([torch.randint(1000, 2000, (x - 1,)), torch.tensor([self.rl_config.mock_eos_token_id])])
            for x in response_length
        ]

        input_ids = [
            torch.cat([prompt_ids[i], response_ids[i * self.n_samples_per_prompt + j]])
            for i in range(self.global_batch_size)
            for j in range(self.n_samples_per_prompt)
        ]
        prompt_length = [
            torch.tensor([x])
            for x in raw_prompt_length
            for _ in range(self.n_samples_per_prompt)
        ]
        responses_length = [
            torch.tensor([x])
            for x in response_length
        ]

        response_batch = torch.nn.utils.rnn.pad_sequence(
            response_ids,
            batch_first=True,
            padding_value=0,
        )

        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        response_mask = torch.ones_like(response_batch, dtype=torch.float32)

        outputs = {
            # List, of varying lengths, represents the actual lengths (for all subsequent rounds)
            'responses': response_ids,
            # No pad, varying in length, including prompt (initial) and response (all subsequent rounds)
            'input_ids': input_ids,
            'response_length': responses_length,
            'prompt_length': prompt_length,
            'rm_scores': score_batch,
            'token_level_rewards': score_batch,
            # The output of the tool has been masked.
            'response_mask': response_mask
        }
        metric = {
            'rollout_cost': 1,
            'resharding_to_infer': 1,
        }
        return outputs, metric
