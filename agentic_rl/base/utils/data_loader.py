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
import numpy as np
import torch
from datasets import Dataset
from omegaconf import ListConfig
from torch.utils.data import Sampler


class GRPODataLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 dataset: Dataset,
                 dataset_additional_keys,
                 global_batch_size,
                 num_samples=None,
                 num_workers=4,
                 seed=42,
                 no_shuffle=False):

        if dataset is None or len(dataset) == 0:
            raise ValueError("Provided dataset is None or contains no data!")

        if not num_samples or num_samples <= 0:
            num_samples = len(dataset)

        if not isinstance(dataset_additional_keys, (list, ListConfig)) or len(dataset_additional_keys) == 0:
            raise ValueError("dataset_additional_keys is not list or is empty list.")

        for key in dataset_additional_keys:
            if key not in dataset.column_names:
                raise ValueError(f"Datasets must contain key '{key}' (from required keys: {dataset_additional_keys}), "
                                 f"but it was not found.")

        for key in dataset.column_names:
            if key not in dataset_additional_keys:
                dataset = dataset.remove_columns(key)

        sampler = RepeatSampler(len(dataset), num_samples, seed, no_shuffle)
        
        def collate_take_first(batch):
            collated = {}

            for key in batch[0].keys():
                values = [sample[key] for sample in batch]

                if isinstance(values[0], list):
                    values = [v[0] if len(v) > 0 else None for v in values]

                if isinstance(values[0], (int, float)):
                    collated[key] = values

                elif isinstance(values[0], str):
                    collated[key] = values

                else:
                    collated[key] = values

            return collated
        
        super().__init__(dataset,
                         num_workers=num_workers,
                         generator=torch.Generator().manual_seed(seed),
                         pin_memory=True,
                         sampler=sampler,
                         batch_size=global_batch_size,
                         drop_last=True,
                         collate_fn=collate_take_first)


class RepeatSampler(Sampler):
    def __init__(self, dataset_len, num_samples, seed, no_shuffle):
        super().__init__()

        if dataset_len <= 0:
            raise ValueError("RepeatSampler require dataset_len greater than 0.")

        self.dataset_len = dataset_len
        self.num_samples = num_samples
        self.seed = seed
        self.no_shuffle = no_shuffle

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        repeat = (self.num_samples + self.dataset_len - 1) // self.dataset_len
        indices = np.tile(np.arange(self.dataset_len), repeat)

        if not self.no_shuffle:
            rng.shuffle(indices)
        indices = indices[:self.num_samples]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples