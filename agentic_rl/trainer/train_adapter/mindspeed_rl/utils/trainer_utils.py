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
import ray
import torch
from torch import Tensor
from tensordict import TensorDict
from typing import Any, List, Dict, Tuple, Sequence

from mindspeed_rl import RayGRPOTrainer, Metric
from mindspeed_rl.trainer.utils.transfer_dock import put_prompts_experience
from mindspeed_rl.utils.pad_process import remove_padding_tensor_dict_to_dict

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_adapter.mindspeed_rl import patch
from agentic_rl.data_manager.data_transform import padding_dict_to_tensor_dict_fast

log = Loggers(__name__)
logger = log.get_logger()


class CommonGRPOTrainer(RayGRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_validation(self, batch: dict) -> tuple:
        """Run a single validation generation step and return rewards and IDs."""
        batch, index = put_prompts_experience(batch, self.validate_n_samples,
                                              self.dataset_additional_keys)
        batch = remove_padding_tensor_dict_to_dict(batch)
        return ray.get(self.rollout_worker.generate_validation.remote(batch, index))

    def _validate_agent(self, dataloader, blocking: bool = True) -> dict:
        """
        Evaluate the agent on a validation dataloader and return metric dict.

        Args:
            dataloader: Validation dataloader providing batch data.
            blocking: Whether to block until generation completes.

        Returns:
            A dict mapping metric names to values, e.g. ``{"val/test_score": float}``.
        """
        logger.info(f"DataLoader length: {len(dataloader.dataset)}")

        val_data_iters = iter(dataloader)
        rewards_lst = []
        uid_lst = []
        step = 1

        for batch_data in val_data_iters:
            logger.info(f"validate iteration: {step}")
            reward, id_list = self._generate_validation(batch_data)
            step = step + 1
            rewards_lst.append(reward)
            uid_lst.append(id_list)

        reward_tensor = torch.cat(rewards_lst, dim=0)
        mean_reward = reward_tensor.mean().item()
        logger.info(f"Validation mean reward: {mean_reward}")
        metric_dict = {"val/test_score": mean_reward}
        return metric_dict


def _maybe_repeat(v: Any, batch_size: int, repeats: int) -> Any:
    """
    If the column still has *batch_size* elements (one per distinct prompt)
    fan it out to ``batch_size x repeats``.
    If it is already length ``batch_size x repeats`` just pass through.
    Works for list, tuple, or Tensor.
    """
    if torch.is_tensor(v) and v.dim() > 0:
        if v.size(0) == batch_size:
            return v.repeat_interleave(repeats, dim=0)
        return v
    if isinstance(v, (list, tuple)):
        if len(v) == batch_size:
            return [x for x in v for _ in range(repeats)]
        return list(v)
    return [v] * (batch_size * repeats)


def put_prompts_answers_experience_fast(
    batch: Dict[str, Tensor],
    n_samples_per_prompt: int,
    dataset_additional_keys: List[str] | None = None,
    indexes: Sequence[int] | None = None,
) -> Tuple[TensorDict, List[int]]:
    """
    • No deepcopy/clone - we only read tensors.
    • Uses list-comprehension + `repeat_interleave` where possible.
    • Builds `input_ids` with one comprehension instead of N loops.
    """
    if dataset_additional_keys is None:
        dataset_additional_keys = []

    # 1. Get raw lists coming from the ingestor
    prompts   = batch["prompts"]
    responses = batch["responses"]
    N = len(prompts)
    R = max(int(n_samples_per_prompt), 1)
    if N % R != 0:
        raise ValueError(
            f"Batch already expanded ({N} samples) but "
            f"n_samples_per_prompt={R} → non-integer grouping."
        )
    B = N // R                      # distinct prompts in the minibatch

    # 2. Pre-compute lengths (as int32 tensors – keeps memory small)
    prompt_length   = [torch.tensor([len(p)], dtype=torch.int32) for p in prompts]
    response_length = [torch.tensor([len(r)], dtype=torch.int32) for r in responses]

    # 3. Concatenate prompt+response once per sample
    input_ids_list = [torch.cat((p, r), dim=0) for p, r in zip(prompts, responses)]

    # 4. Repeat any additional columns
    add_vals = {
        k: _maybe_repeat(batch[k], B, R)
        for k in dataset_additional_keys
        if k in batch
    }
    # 5. Build dict
    data_dict = {
        "prompt_length":  prompt_length,
        "prompts":        prompts,
        "response_length": response_length,
        "responses":      responses,
        "input_ids":      input_ids_list,
        **add_vals,
    }

    # Default indexes: lightweight range() wrap – avoids heavy list()
    if indexes is None:
        indexes = list(range(N))

    return padding_dict_to_tensor_dict_fast(data_dict), indexes