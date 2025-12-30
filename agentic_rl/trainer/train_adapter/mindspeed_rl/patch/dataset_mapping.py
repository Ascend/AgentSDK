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
import random

import numpy as np
from mindspeed_rl.datasets.utils import _build_shuffle_idx

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def _check_params(start_index, nb_documents, num_samples, full_shuffle_instruction_dataset, no_shuffle):
    if not isinstance(start_index, np.int32):
        logger.error("start_index should be np.int32")
        raise ValueError("start_index should be np.int32")
    if start_index < 0:
        logger.error(f"start_index should be no negative, but got {start_index}")
        raise ValueError(f"start_index should be no negative, but got {start_index}")
    if not isinstance(nb_documents, int):
        logger.error("nb_documents should be int")
        raise ValueError("nb_documents should be int")
    if nb_documents <= 0:
        logger.error(f"nb_documents should be positive, but got {nb_documents}")
        raise ValueError(f"nb_documents should be positive, but got {nb_documents}")
    if not isinstance(num_samples, int):
        logger.error("num_samples should be int")
        raise ValueError("num_samples should be int")
    if num_samples <= 0:
        logger.error(f"num_samples should be positive, but got {num_samples}")
        raise ValueError(f"num_samples should be positive, but got {num_samples}")
    if not isinstance(full_shuffle_instruction_dataset, bool):
        logger.error("full_shuffle_instruction_dataset should be bool")
        raise ValueError("full_shuffle_instruction_dataset should be bool")
    if not isinstance(no_shuffle, bool):
        logger.error("no_shuffle should be bool")
        raise ValueError("no_shuffle should be bool")


def safe_build_index_mappings(
        name,
        data_prefix,
        start_index,
        nb_documents,
        num_samples: int,
        seed,
        full_shuffle_instruction_dataset,
        parallel_state,
        no_shuffle=False
):
    """
    Patch for build index mapping of mindspeed-rl
    """
    if not isinstance(seed, int):
        logger.error("seed should be int")
        raise ValueError("seed should be int")
    if seed < 0:
        logger.error(f"seed should be no negative, but got {seed}")
        raise ValueError(f"seed should be no negative, but got {seed}")
    np_rng = np.random.RandomState(seed=seed)

    _check_params(start_index, nb_documents, num_samples, full_shuffle_instruction_dataset, no_shuffle)

    shuffle_idx = []
    while len(shuffle_idx) < num_samples:
        try:
            new_document_ids = _build_shuffle_idx(
                nb_documents=nb_documents,
                start_index=start_index,
                np_rng=np_rng,
                no_shuffle=no_shuffle
            )
        except RuntimeError as e:
            logger.error(f"build shuffle idx failed, error:{e}")
            raise RuntimeError("build shuffle idx failed") from e
        except Exception as e:
            logger.error(f"Unexpected error occurred when build shuffle idx, error:{e}")
            raise RuntimeError("Unexpected error occurred when build shuffle idx") from e

        if not isinstance(new_document_ids, np.ndarray):
            logger.error("new_document_ids should be np.ndarray")
            raise ValueError("new_document_ids should be np.ndarray")

        shuffle_idx.extend(new_document_ids.tolist())

    if full_shuffle_instruction_dataset:
        random.shuffle(shuffle_idx)

    return shuffle_idx
