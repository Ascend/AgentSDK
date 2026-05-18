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

import copy
from typing import Dict, List, Union
from torch import Tensor
from tensordict import TensorDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


def padding_dict_to_tensor_dict(experience_data: Dict[str, Union[Tensor, List[Tensor]]]):
    experience_batch = {}
    experience_data_length = []
    for experience_column, value in experience_data.items():
        max_length = max(len(exp) for exp in value)
        padded_tensors = [
            torch.nn.functional.pad(exp, (0, max_length - len(exp)), mode='constant', value=0) for exp in value
        ]
        experience_batch[experience_column] = torch.stack(padded_tensors, dim=0)
        experience_data_length.extend([torch.tensor(len(exp)) for exp in value])
    experience_batch['original_length'] = torch.stack(experience_data_length)
    experience_batch = TensorDict.from_dict(experience_batch)
    return experience_batch


def remove_padding_tensor_dict_to_dict(data_dict: TensorDict[str, Union[Tensor, List[Tensor]]]):
    remove_padding_tensors = {}
    if data_dict is None:
        return remove_padding_tensors
    if 'original_length' not in data_dict.keys():
        return data_dict
    data_lengths = data_dict['original_length']
    for idx, (key, dict_value) in enumerate(data_dict.items()):
        if key == 'original_length':
            continue
        remove_padding_tensors[key] = truncate_rows(
            dict_value, data_lengths[idx * len(dict_value) : (idx + 1) * len(dict_value)]
        )
    return remove_padding_tensors


def remove_padding_and_split_to_list(
    responses: torch.Tensor, eos_token_id: int, pad_token_id: int, to_list: bool = False
) -> List[torch.Tensor]:
    output = []
    for i in range(responses.shape[0]):
        response = responses[i]
        non_zeros = torch.nonzero(response == pad_token_id, as_tuple=False)
        if len(non_zeros) != 0:
            first_pad_index = non_zeros[0][0]
        else:
            first_pad_index = len(response)
        if pad_token_id == eos_token_id:
            response = response[: first_pad_index + 1]
        else:
            response = response[:first_pad_index]
        if to_list:
            response = response[:-1].cpu().numpy().tolist()
        output.append(response)
    return output


def pad_multiple(data_list: List[Tensor], pad_id: Union[float, int], multiple: int = 1) -> Tensor:
    padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
    max_len = padded.size(1)
    target_len = ((max_len + multiple - 1) // multiple) * multiple
    padded = F.pad(padded, (0, target_len - max_len), value=pad_id)

    return padded


def truncate_middle_and_pad(responses, input_tensor, truncate_lengths, pad_value=0.0):
    """
    input_tensor: Tensor of shape (mbs, seq_len, vocab_size)
    truncate_lengths: Tensor of shape (mbs, 2), where truncate_lengths[i, 0] is the start index to keep,
                      and truncate_lengths[i, 1] is the end index to keep (exclusive).
    pad_value: Value to use for padding (default is 0.0)
    """

    mbs, seq_len, vocab_size = input_tensor.shape

    # Ensure truncate_lengths is within valid range
    truncate_lengths = torch.clamp(truncate_lengths, 0, seq_len)

    # Calculate the new lengths after truncation
    new_lengths = truncate_lengths[:, 1] - truncate_lengths[:, 0]  # (mbs,)

    # Find the maximum length after truncation
    max_new_len = responses.shape[-1]

    # Initialize the output tensor with padding values
    output_tensor = torch.full(
        (mbs, max_new_len, vocab_size), pad_value, dtype=input_tensor.dtype, device=input_tensor.device
    )

    # Fill the output tensor with truncated values
    for i in range(mbs):
        start_idx = truncate_lengths[i, 0].item()  # Start index to keep
        end_idx = truncate_lengths[i, 1].item()  # End index to keep (exclusive)
        new_len = new_lengths[i].item()  # New length after truncation

        # Copy the middle part of the row to the output tensor
        output_tensor[i, :new_len] = input_tensor[i, start_idx:end_idx]

    return output_tensor


def truncate_prompt_and_pad(responses, input_tensor, truncate_lengths, pad_value=0.0):
    """
    input_tensor: Tensor of shape (mbs, seq_len)
    truncate_lengths: Tensor of shape (mbs, 2), where truncate_lengths[i, 0] is the start index to keep,
                      and truncate_lengths[i, 1] is the end index to keep (exclusive).
    pad_value: Value to use for padding (default is 0.0)
    """

    mbs, seq_len = input_tensor.shape

    # Ensure truncate_lengths is within valid range
    truncate_lengths = torch.clamp(truncate_lengths, 0, seq_len)

    # Calculate the new lengths after truncation
    new_lengths = truncate_lengths[:, 1] - truncate_lengths[:, 0]  # (mbs,)

    # Find the maximum length after truncation
    max_new_len = responses.shape[-1]

    # Initialize the output tensor with padding values
    output_tensor = torch.full((mbs, max_new_len), pad_value, dtype=input_tensor.dtype, device=input_tensor.device)

    # Fill the output tensor with truncated values
    for i in range(mbs):
        start_idx = truncate_lengths[i, 0].item()  # Start index to keep
        end_idx = truncate_lengths[i, 1].item()  # End index to keep (exclusive)
        new_len = new_lengths[i].item()  # New length after truncation

        # Copy the middle part of the row to the output tensor
        output_tensor[i, :new_len] = input_tensor[i, start_idx:end_idx]

    return output_tensor


def truncate_rows(tensor: torch.Tensor, index_tensor: torch.Tensor, left_pad: bool = False) -> list[torch.Tensor]:
    """
    tensor: 2D Tensor with shape (mbs, seq_len)
    index_tensor: 2D Tensor with shape (mbs, 1), indicating the truncation position for each row
    """
    mbs = tensor.shape[0]
    truncated_tensors = []

    for i in range(mbs):
        # Handle the case of an empty row (e.g., padding tokens)
        if index_tensor[i].item() == 0 and tensor[i, 0].item() == -1:
            truncated_row = torch.tensor([], dtype=torch.int32).cpu()
        else:
            # Get the truncation index for the current row
            trunc_idx = index_tensor[i].item()
            # Truncate the current row
            if left_pad:
                truncated_row = tensor[i, -trunc_idx:].cpu()
            else:
                truncated_row = tensor[i, :trunc_idx].cpu()

        # Append the truncated row to the list
        truncated_tensors.append(truncated_row)

    return truncated_tensors


def put_prompts_experience(
    batch: Dict[str, torch.Tensor],
    n_samples_per_prompt,
    dataset_additional_keys: List[str] = None,
    indexes=None,
    add_another_batch=False,
):
    """Put data into specified columns and rows.

    Args:
        batch: Batch data from original dataloader.
        n_samples_per_prompt: n_samples_per_prompt
        dataset_additional_keys: The additional experience types from the dataset.
        indexes: Batch data indexes.
        add_another_batch: False
    Returns: TensorDict

    """

    prompts = batch["prompts"]
    prompt_length = []
    for prompt in prompts:
        for _ in range(n_samples_per_prompt):
            prompt_length.append(torch.tensor([len(prompt)]))

    prompts_data = prompts
    prompts = []
    for prompt in prompts_data:
        for _ in range(n_samples_per_prompt):
            prompts.append(copy.deepcopy(prompt))

    add_vals = {}
    for add_keys in dataset_additional_keys:
        if add_keys in batch.keys():
            values = []
            for value in batch[add_keys]:
                for _ in range(n_samples_per_prompt):
                    values.append(value)
            add_vals[add_keys] = values
    prompt_nums = len(prompt_length)
    if add_another_batch:
        indexes = [prompt_nums + i for i in range(prompt_nums)]
    elif indexes is None:
        indexes = [i for i in range(len(prompt_length))]

    data_dict = dict({"prompt_length": prompt_length, "prompts": prompts}, **add_vals)
    return padding_dict_to_tensor_dict(data_dict), indexes
