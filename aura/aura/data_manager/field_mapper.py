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

from typing import Dict, List

import torch


class FieldMapper:
    """
    Field mapper between Verl and Mindspeed_RL
    """

    @classmethod
    def convert_batch(cls, raw_data: Dict, pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
        """
        Convert the mindspeed_rl format to the verl format

        Args:
            raw_data: The original data in the mindspeed_rl format
                - prompts: List[Tensor]
                - responses: List[Tensor]
                - response_mask: Optional[List[Tensor]]
                - rm_scores: Optional[List[Tensor]]
                - token_level_rewards: Optional[List[Tensor]]
            pad_token_id: padding token ID

        Returns:
            dict: Data in the Verl format
                - prompts: Tensor (batch, prompt_length) Left padding
                - responses: Tensor (batch, response_length) Right padding
                - input_ids: Tensor (batch, total_length)
                - attention_mask: Tensor (batch, total_length)
                - response_mask: Tensor (batch, response_length)
                - position_ids: Tensor (batch, total_length)
                - rm_scores: Tensor (batch, response_length)
                - token_level_rewards: Tensor (batch, response_length)
        """
        responses_list = raw_data['responses']
        prompts_list = raw_data['prompts']
        bsz = len(responses_list)

        # 1. Dynamically determine the maximum length within the Batch
        prompt_length = max(p.size(-1) for p in prompts_list)
        response_length = max(r.size(-1) for r in responses_list)
        total_length = prompt_length + response_length

        # 2. Initialize the full-scale Tensor
        device = responses_list[0].device
        batch = {
            "prompts": torch.full((bsz, prompt_length), pad_token_id, device=device, dtype=torch.long),
            "responses": torch.full((bsz, response_length), pad_token_id, device=device, dtype=torch.long),
            "input_ids": torch.full((bsz, total_length), pad_token_id, device=device, dtype=torch.long),
            "attention_mask": torch.zeros((bsz, total_length), device=device, dtype=torch.long),
            "response_mask": torch.zeros((bsz, response_length), device=device, dtype=torch.long),
            "position_ids": torch.zeros((bsz, total_length), device=device, dtype=torch.long),
            "rm_scores": torch.zeros((bsz, response_length), device=device, dtype=torch.float32),
            "token_level_rewards": torch.zeros((bsz, response_length), device=device, dtype=torch.float32),
            "rollout_log_probs": torch.zeros((bsz, response_length), device=device, dtype=torch.float32),
        }

        for i in range(bsz):
            cls._process_single_sample(
                batch=batch,
                idx=i,
                raw_data=raw_data,
                prompts_list=prompts_list,
                responses_list=responses_list,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        if 'prompt_ids' in raw_data.keys():
            # Retain the original prompt_id for GRPO grouping
            batch['_prompt_id'] = raw_data['prompt_ids']

        return batch

    @classmethod
    def _process_single_sample(
            cls,
            batch: Dict,
            idx: int,
            raw_data: Dict,
            prompts_list: List,
            responses_list: List,
            prompt_length: int,
            response_length: int,
    ):
        """Handling the filling and conversion of individual samples"""
        # A. Prompt Left Padding
        p_data = prompts_list[idx].squeeze()
        p_len = p_data.size(0)
        batch["prompts"][idx, -p_len:] = p_data

        # B. Response Right Padding
        r_data = responses_list[idx].squeeze()
        r_len = r_data.size(0)
        batch["responses"][idx, :r_len] = r_data

        # C. merge input_ids [Padded Prompt | Valid Response]
        batch["input_ids"][idx, :prompt_length] = batch["prompts"][idx]
        batch["input_ids"][idx, prompt_length:prompt_length + r_len] = r_data

        # D. Attention Mask (The actual data positions are marked as 1)
        batch["attention_mask"][idx, prompt_length - p_len:prompt_length + r_len] = 1

        # E. Response Mask
        if 'response_mask' in raw_data and raw_data['response_mask'] is not None:
            rm_data = raw_data['response_mask'][idx].squeeze()
            batch["response_mask"][idx, :rm_data.size(0)] = rm_data
        else:
            # Default: All valid response tokens are 1.
            batch["response_mask"][idx, :r_len] = 1

        # F. Handle Reward
        if 'rm_scores' in raw_data:
            rew_data = raw_data['rm_scores'][idx].flatten()
            limit = min(rew_data.size(0), response_length)
            batch["rm_scores"][idx, :limit] = rew_data[:limit]

        if 'token_level_rewards' in raw_data:
            rew_data = raw_data['token_level_rewards'][idx].flatten()
            limit = min(rew_data.size(0), response_length)
            batch["token_level_rewards"][idx, :limit] = rew_data[:limit]

        if 'rollout_log_probs' in raw_data:
            logprob_data = raw_data['rollout_log_probs'][idx].flatten()
            limit = min(logprob_data.size(0), response_length)
            batch["rollout_log_probs"][idx, :limit] = logprob_data[:limit]

        # G. Generate Position IDs
        valid_mask = batch["attention_mask"][idx]
        seq_pos = torch.cumsum(valid_mask, dim=0)
        batch["position_ids"][idx] = seq_pos * valid_mask

    @classmethod
    def convert_dataproto_to_msrl(cls, data_proto) -> Dict:
        """
        Convert the verl DataProto back to the mindspeed_rl format (if necessary)

        For result write-back or compatibility scenarios
        """
        result = {}

        if data_proto.batch is not None:
            for key, tensor in data_proto.batch.items():
                result[key] = tensor

        for key, value in data_proto.non_tensor_batch.items():
            result[key] = value

        return result
