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
from typing import Any, Optional, Tuple

import ray

import third_party.rl
from mindspeed_rl.datasets.build_dataset import build_train_valid_test_datasets
from mindspeed_rl.datasets.prompt_dataset import PromptDataset
from mindspeed_rl.datasets.dataloader import PromptDataLoader

from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.data_loader_validate import build_validate_test_dataloader
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def default_train_dataloader(
    data_loader_config: Any,
    validate_num_samples: int,
    consumed_train_samples: int,
):
    """
    Build the default training dataloader together with optional validation / test dataloaders.

    Args:
        data_loader_config: Configuration object carrying dataset paths, batch size, etc.
        validate_num_samples: Number of samples reserved for validation.
        consumed_train_samples: Number of training samples already consumed (for resumption).

    Returns:
        A tuple of (data_iters, val_dataloader, test_dataloader).
    """
    train_data_path = (
        data_loader_config.train_data_path
        if data_loader_config.train_data_path
        else data_loader_config.data_path
    )
    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=[train_data_path, ],
        splits_string=data_loader_config.split,
        seq_length=data_loader_config.seq_length,
        train_valid_test_num_samples=[
            data_loader_config.train_iters * data_loader_config.global_batch_size, 0, 0
        ],
        seed=data_loader_config.seed,
        dataset_cls=PromptDataset,
        extra_param=data_loader_config
    )
    logger.info('after dataset is built')

    data_loader = PromptDataLoader(
        train_ds, data_loader_config.global_batch_size,
        data_loader_config.num_workers, data_loader_config.seed, data_loader_config.dataset_additional_keys,
        data_loader_config.no_shuffle
    )
    data_iters = iter(data_loader)

    # Skip already-consumed training samples for resumption
    for _ in range(consumed_train_samples // data_loader_config.global_batch_size):
        next(data_iters)

    val_dataloader = None
    test_dataloader = None
    if data_loader_config.test_data_path is not None and len(data_loader_config.test_data_path) > 0:
        val_dataloader, test_dataloader = build_validate_test_dataloader(data_loader_config, validate_num_samples)

    logger.info('after dataloader is built')
    return data_iters, val_dataloader, test_dataloader