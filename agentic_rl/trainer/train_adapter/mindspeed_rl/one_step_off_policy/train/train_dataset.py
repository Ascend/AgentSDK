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
from typing import Dict, Callable, Optional, Any, List
import json
import os
import numpy as np

from mindspeed_rl.datasets.base_dataset import BaseDataset
from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()
IGNORE_INDEX = -100

class TrainDataset(BaseDataset):
    """
    Returns *groups* of samples instead of single samples.
    Each group has length = group_size * global_batch_size.
    Every element carries 'prompt_id' and 'mini_batch_id'.

    Assumes a packed indexed dataset, mirroring PromptDataset behavior.
    """
    def __init__(
        self,
        data_prefix: str,
        is_packed_data: bool,
        tokenizer: Callable,
        seq_length: int,
        num_samples: Optional[int] = None,
        name: str = "",
        documents: Any = None,
        seed: int = 42,
        full_shuffle_instruction_dataset: bool = False,
        token_param: Optional[Dict] = None,
        preprocess_template: Optional[str] = None,
        pad_token: int = 0,
        eos_token: int = 1,
        extra_param: Any = None,
        **kwargs,
    ):
        self.data_prefix = data_prefix
        self.is_packed_data = is_packed_data
        self.tokenizer = tokenizer
        self.token_param = token_param
        self.seq_length = seq_length
        self.preprocess_template = preprocess_template
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.num_samples = num_samples
        self.args = extra_param

        if not self.is_packed_data:
            raise NotImplementedError("one step off Dataset currently supports only packed data.")

        # ---- Load packed dataset ----
        self.res_dataset = get_packed_indexed_dataset(
            data_prefix=self.data_prefix,
            filter_length=getattr(extra_param, "max_prompt_length", None),
        )

        # ---- Load side meta (ints) from JSON ----
        self.side_meta: Dict[str, np.ndarray] = {}
        meta_path = f"{self.data_prefix}_packed_meta.json"
        logger.info(f"********* Meta path: {meta_path}")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                raw_meta = json.load(f)
            logger.info(f"********* Meta Columns: {str(raw_meta.keys())}")
            # Cast to numpy arrays (aligned length already ensured during preprocessing)
            for k, v in raw_meta.items():
                raw_arr = np.asarray(v, dtype=np.int64)
                mask = getattr(self.res_dataset, "applied_filter", None)
                if mask is not None:
                    if len(mask) != len(v):
                        raise ValueError(f"Mask len {len(mask)} != meta len {len(v)}")
                    raw_arr = raw_arr[mask]
                self.side_meta[k] = raw_arr

        # dataset_type only used by BaseDataset for logging/identity
        dataset_type = "Astp_DS_Packed"
        super().__init__(self.res_dataset, dataset_type)

        if self.side_meta:
            n_meta = len(next(iter(self.side_meta.values())))
            if n_meta != len(self.res_dataset):
                raise ValueError(f"meta {n_meta} vs packed {len(self.res_dataset)}")

        # ---- Build an index mapping for groups ----
        # Prefer packed field if present; otherwise use side_meta
        batch_group_arr = self._get_field_array("batch_group")
        prev_group = None
        self.ordinal_to_index: Dict[int, int] = {}
        for idx, group in enumerate(batch_group_arr):
            if group != prev_group:
                self.ordinal_to_index[len(self.ordinal_to_index)] = idx
                prev_group = group


    # ------------- Standard Dataset methods ------------------
    def __len__(self):
        return len(self.ordinal_to_index)

    def __getitem__(self, ordinal_idx: int) -> List[Dict[str, Any]]:
        # Start of the group in packed doc indices
        start_idx = self.ordinal_to_index[ordinal_idx]
        batch_group_arr = self._get_field_array("batch_group")
        group_id = batch_group_arr[start_idx]

        group_lst: List[Dict[str, Any]] = []
        idx = start_idx
        total_len = len(batch_group_arr)
        while idx < total_len and batch_group_arr[idx] == group_id:
            packed_item = self.res_dataset[idx]  # dict with whatever was serialized
            item = self._cut_instruction_token(packed_item, dtype=np.int64, sample_idx=idx)
            group_lst.append(item)
            idx += 1

        return group_lst
    # ------------- Internal helpers ------------------
    def _get_field_array(self, key: str) -> np.ndarray:
        if key in self.side_meta:
            return self.side_meta[key]
        # try to pull from packed datasets
        if hasattr(self.res_dataset, "datasets") and key in self.res_dataset.datasets:
            # build a big array once
            return np.array([self.res_dataset[i][key] for i in range(len(self.res_dataset))])

        raise KeyError(f"Field '{key}' not found in packed dataset or side_meta.")

    def _merge_side_meta_into_item(self, item: Dict[str, Any], sample_idx: int) -> None:
        """Attach integer side meta fields for this sample_idx (if any)."""
        for k, arr in self.side_meta.items():
            if k not in item:  # avoid overwriting packed version
                item[k] = np.int64(arr[sample_idx])

    # ------------- Internal token trimming logic --------------
    def _cut_instruction_token(self, item, dtype, sample_idx: int):
        """
        Trim / pad as needed, and add meta fields.
        `item` is the dict from CombinedDataset for one sample.
        """
        # If you still have labels in packed data and don't use args.dataset_additional_keys
        if "labels" in item.keys() and not getattr(self.args, "dataset_additional_keys", []):
            token_length = len(item["input_ids"])
            if token_length <= self.seq_length:
                res = {
                    "input_ids": item["input_ids"].astype(dtype),
                    "attention_mask": np.ones_like(item["input_ids"]).astype(dtype),
                    "labels": item["labels"].astype(dtype),
                }
            else:
                input_ids = item["input_ids"][:self.seq_length]
                res = {
                    "input_ids": input_ids.astype(dtype),
                    "attention_mask": np.ones_like(input_ids).astype(dtype),
                    "labels": item["labels"][:self.seq_length].astype(dtype),
                }
        else:
            # General path for one step off (no labels or custom keys)
            prompt_ids = item["input_ids"]
            input_ids = prompt_ids[:self.seq_length]
            add_vals = {}
            for add_key in getattr(self.args, "dataset_additional_keys", []):
                if add_key in item:
                    # If packed serializer tokenized them, they are numpy arrays already.
                    add_vals[add_key] = item[add_key]
            res = {
                "input_ids": input_ids.astype(dtype),
                "attention_mask": np.ones_like(input_ids).astype(dtype),
            }
            res.update(add_vals)

        # Merge side meta ints (batch_group, mini_batch_id, prompt_id, etc.)
        self._merge_side_meta_into_item(res, sample_idx)

        return res