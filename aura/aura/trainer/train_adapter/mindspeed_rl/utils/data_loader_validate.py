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
from typing import Any, List, Optional, Tuple
import numpy as np

import torch

from torch.utils.data import RandomSampler, SequentialSampler

from mindspeed_rl.datasets.prompt_dataset import PromptDataset
from mindspeed_rl.datasets.indexed_dataset import get_packed_indexed_dataset


class ValidatePromptDataLoader(torch.utils.data.DataLoader):
    """DataLoader for validation / test prompt datasets with optional shuffling."""

    def __init__(
        self,
        dataset: Any,
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: List[str],
        no_shuffle: bool,
        is_pairwise_dataset: bool = False,
        tokenizer: Any = None,
    ) -> None:
        def collator(features, return_tensors=None):
            features_dict = {}
            features_dict["prompts"] = [torch.tensor(value['input_ids']) for value in features]

            for add_key in dataset_additional_keys:
                features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

            return features_dict

        if not no_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(dataset,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collator,
                        pin_memory=True,
                        sampler=sampler,
                        batch_size=global_batch_size,
                        drop_last=False)


def build_validate_test_dataset(
    data_prefix: list,
    seq_length: int,
    validate_num_samples: int,
    dataset_cls: Any,
    tokenizer: Any = None,
    parallel_state: Any = None,
    full_shuffle_instruction_dataset: bool = False,
    no_shuffle: bool = False,
    reset_position_ids: bool = False,
    prompt_type: Optional[str] = None,
    prompt_type_path: Optional[str] = None,
    seed: int = 42,
    extra_param: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Build validation and test datasets from indexed data.

    Args:
        data_prefix: Paths to the data files.
        seq_length: Maximum sequence length.
        validate_num_samples: Number of validation samples to use.
        dataset_cls: Dataset class constructor.
        tokenizer: Tokenizer instance.
        parallel_state: Megatron parallel state.
        full_shuffle_instruction_dataset: Whether to fully shuffle instruction data.
        no_shuffle: Disable shuffling when True.
        reset_position_ids: Reset position IDs when True.
        prompt_type: Prompt template type identifier.
        prompt_type_path: Path to prompt template file.
        seed: Random seed for reproducibility.
        extra_param: Extra parameters forwarded to the dataset constructor.

    Returns:
        A tuple of (valid_dataset, test_dataset), either of which may be None.

    Raises:
        ValueError: If ``dataset_cls`` is None.
    """
    if dataset_cls is None:
        raise ValueError("dataset_cls must be provided.")

    if isinstance(data_prefix, list):
        data_prefix = data_prefix[0]

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix,
                                filter_length=getattr(extra_param, 'max_prompt_length', None)
                            )
    total_num_of_documents = len(list(packed_indexed_dataset.datasets.values())[0])

    all_documents = np.arange(start=0, stop=total_num_of_documents, dtype=np.int32)
    validate_num_samples = min(validate_num_samples, total_num_of_documents)

    valid_documents = all_documents[:validate_num_samples]
    test_documents = all_documents.copy()

    def build_dataset(documents, name):
        if len(documents) == 0:
            return None
        dataset = None

        # documents is a contiguous range [start, end]; shuffling operates at the document level
        dataset = dataset_cls(
            parallel_state=parallel_state,
            dataset_type='LLM',
            data_prefix=data_prefix,
            is_packed_data=True,
            tokenizer=tokenizer,
            seq_length=seq_length,
            num_samples=len(documents),
            name=name,
            documents=documents,
            seed=seed,
            full_shuffle_instruction_dataset=full_shuffle_instruction_dataset,
            no_shuffle=no_shuffle,
            reset_position_ids=reset_position_ids,
            prompt_type=prompt_type,
            prompt_type_path=prompt_type_path,
            extra_param=extra_param
        )
        return dataset

    valid_dataset = build_dataset(valid_documents, 'valid')
    test_dataset = build_dataset(test_documents, 'test')

    return valid_dataset, test_dataset


def build_validate_test_dataloader(
    actor_config: Any,
    validate_num_samples: int,
) -> Tuple[ValidatePromptDataLoader, ValidatePromptDataLoader]:
    """
    Build validation and test dataloaders from actor configuration.

    Args:
        actor_config: Actor configuration with dataset paths and hyperparameters.
        validate_num_samples: Number of validation samples.

    Returns:
        A tuple of (val_dataloader, test_dataloader).
    """
    val_dataset, test_dataset = build_validate_test_dataset(
        data_prefix=[actor_config.test_data_path, ],
        seq_length=actor_config.seq_length,
        validate_num_samples=validate_num_samples,
        seed=actor_config.seed,
        dataset_cls=PromptDataset,
        extra_param=actor_config
    )
    val_dataloader = ValidatePromptDataLoader(
        val_dataset,
        actor_config.global_batch_size,
        actor_config.num_workers,
        actor_config.seed,
        actor_config.dataset_additional_keys,
        actor_config.no_shuffle
    )

    test_dataloader = ValidatePromptDataLoader(
        test_dataset,
        actor_config.global_batch_size,
        actor_config.num_workers,
        actor_config.seed,
        actor_config.dataset_additional_keys,
        actor_config.no_shuffle
    )
    return val_dataloader, test_dataloader