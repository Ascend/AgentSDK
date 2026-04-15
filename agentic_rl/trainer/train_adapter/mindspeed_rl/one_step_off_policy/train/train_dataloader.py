#!/usr/bin/env python3
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
from typing import List, Dict, Any
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

from mindspeed_rl.datasets.build_dataset import build_train_valid_test_datasets

from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_dataset import TrainDataset
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

class TrainDataLoader(DataLoader):
    """
    Shuffles data at the *group* level via RandomSampler/SequentialSampler.
    Each dataset item is a *group*: list[dict].
    Collator flattens or re-packs as you wish.
    """

    def __init__(
        self,
        dataset: TrainDataset,
        num_workers: int,
        seed: int,
        dataset_additional_keys: List[str],
        no_shuffle: bool = False,
        groups_per_step: int = 1,    # how many groups does one step consume?
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.dataset_additional_keys = dataset_additional_keys
        self.groups_per_step = groups_per_step

        def collator(groups_batch: List[List[Dict[str, Any]]]):
            """
            groups_batch: length = groups_per_step
            Each element is a list of sample dicts (length group_size * global_batch_size).
            We flatten across groups_for_step, then build dict-of-lists-of-tensors.
            """

            flat_samples = [
                s
                for group in groups_batch
                for s in group
            ]

            out: Dict[str, List[torch.Tensor]] = {}
            out["input_ids"]      = [torch.tensor(s["input_ids"]) for s in flat_samples]
            out["attention_mask"] = [torch.tensor(s["attention_mask"]) for s in flat_samples]

            # labels may be missing (in inference or certain datasets)
            if "labels" in flat_samples[0]:
                out["labels"] = [torch.tensor(s["labels"]) for s in flat_samples]

            # Add prompt_id and mini_batch_id (keep them as python objects or convert to tensors as ints)
            out["prompt_id"]     = [s["prompt_id"] for s in flat_samples]
            out["mini_batch_id"] = [s["mini_batch_id"] for s in flat_samples]

            for add_key in self.dataset_additional_keys:
                if add_key in flat_samples[0]:
                    out[add_key] = [torch.tensor(s[add_key]) for s in flat_samples]
            return out

        # --- Choose sampler ---
        if not no_shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=g, num_samples=len(dataset))
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(
            dataset,
            batch_size=self.groups_per_step,
            sampler=sampler,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=collator,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

# Optimized data loader
def optimize_train_dataloader(actor_config, validate_num_samples, consumed_train_samples):
    logger.info(">>> optimize initializing train DataLoader")
    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=[actor_config.data_path, ],
        splits_string=actor_config.split,
        seq_length=actor_config.seq_length,
        train_valid_test_num_samples=[
            validate_num_samples, 0, 0
        ],
        seed=actor_config.seed,
        dataset_cls=TrainDataset,
        extra_param=actor_config
    )

    rollout_dataloader = TrainDataLoader(
        dataset = train_ds,
        num_workers = actor_config.num_workers,
        seed = actor_config.seed,
        dataset_additional_keys = actor_config.dataset_additional_keys,
        no_shuffle = actor_config.no_shuffle,
        groups_per_step = 1,
    )
    data_iters = iter(rollout_dataloader)

    # Skip already processed data samples
    [next(data_iters) for _ in range(consumed_train_samples // actor_config.global_batch_size)]

    return data_iters