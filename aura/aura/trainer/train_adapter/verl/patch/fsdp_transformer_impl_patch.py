#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad
import verl.utils.torch_functional as verl_F

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def prepare_model_outputs_patch(self, output, output_args, micro_batch: TensorDict):
    logger.debug("prepare_model_outputs_patch")

    use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
    pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
    calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

    model_output = {}

    input_ids = micro_batch["input_ids"]

    if use_remove_padding:
        input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
        temperature_rmpad = output_args["temperature_rmpad"]

        if use_fused_kernels:
            # temperature is singleton
            log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
            entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
        else:
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            ## [Aura feature sp] patch begin
            if logits_rmpad.shape[0] != temperature_rmpad.shape[0]:
                repeat_factor = logits_rmpad.shape[0] // temperature_rmpad.shape[0]
                temperature_rmpad = temperature_rmpad.repeat(repeat_factor)
            ## [Aura feature sp] patch end
            logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

            # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=input_ids_rmpad_rolled,
                inplace_backward=inplace_backward,
            )

            # compute entropy
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

        # gather log_prob if sp > 1
        if self.use_ulysses_sp:
            pad_size = output_args["pad_size"]

            # gather and unpad for the ulysses sp
            log_probs = gather_outputs_and_unpad(
                log_probs,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )
            if calculate_entropy:
                entropy_rmpad = gather_outputs_and_unpad(
                    entropy_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )

        if pad_mode == DatasetPadMode.NO_PADDING:
            cu_seqlens = input_ids.offsets()
            # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

    else:  # not using rmpad and no ulysses sp
        response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
        if use_fused_kernels:
            log_probs = output.log_probs[:, -response_length - 1 : -1]
            entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

        else:
            logits = output.logits  # (bsz, response_length, vocab_size)
            temperature = output_args["temperature"]  # (bsz,)
            temperature = temperature.unsqueeze(-1).unsqueeze(-1)
            logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                logits_rmpad = torch.cat([t for t in logits.unbind()])
                input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                    entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

    model_output["log_probs"] = log_probs
    if calculate_entropy:
        model_output["entropy"] = entropy

    return model_output
