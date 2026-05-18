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

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import restore_dynamic_batch

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def postprocess_batch_func_patch(output_lst, indices, data: TensorDict):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """
    logger.debug("postprocess_batch_func_patch")
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    if pad_mode != DatasetPadMode.NO_PADDING:
        raise ValueError("postprocess_batch_func only support NO_PADDING pad_mode")

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        if pad_mode == DatasetPadMode.NO_PADDING:
            ## [Aura feature sp] patch begin
            import os

            sp_size = int(os.getenv("ULYSSES_SEQUENCE_PARALLEL_SIZE", 1))
            logger.info(f"post process batch, sp_size: {sp_size}")
            tensors = []
            for nt in model_output[key]:
                values = nt.values()
                if nt._lengths is not None:
                    orig_lengths = nt._lengths.tolist()
                else:
                    offsets = nt.offsets()
                    orig_lengths = (offsets[1:] - offsets[:-1]).tolist()

                local_lengths = [l // sp_size for l in orig_lengths]
                diff = values.size(0) - sum(local_lengths)
                if diff != 0:
                    local_lengths[-1] += diff
                all_seqs = torch.split(values, local_lengths, dim=0)
                tensors.extend(all_seqs)
            # tensors = [tensor for nt in model_output[key] for tensor in nt.unbind()]
            ## [Aura feature sp] patch end
            model_output[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # reverse with dynamic bsz
        if use_dynamic_bsz:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output
