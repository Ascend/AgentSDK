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
import multiprocessing
import os
import sys
from pathlib import Path

import hydra

from third_party.rl.mindspeed_rl.mindspeed_rl.config_cls.data_handler_config import DataHandlerConfig
from third_party.rl.mindspeed_rl.mindspeed_rl.config_cls.validate_config import validate_data_handler_config
from third_party.rl.mindspeed_rl.mindspeed_rl.datasets.indexed_dataset import IndexedDatasetBuilder
from third_party.rl.mindspeed_rl.mindspeed_rl.datasets.preprocess_data import merge_datasets, build_splitter, cut_range_to_subs, \
    handle_subset
from third_party.rl.mindspeed_rl.mindspeed_rl.utils.tokenizer import get_tokenizer
from third_party.rl.mindspeed_rl.mindspeed_rl.datasets.data_handler import build_dataset, get_dataset_handler
from third_party.rl.mindspeed_rl.mindspeed_rl.utils.loggers import Loggers

logger = Loggers(name="process_data")
cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir, "./configs/model/templates.json")

config_name = sys.argv.pop(1)


def preprocess(config):
    args = DataHandlerConfig(config)
    validate_data_handler_config(args)

    if args.merge_group_keys is not None:
        merge_datasets(args)
        return

    tokenizer = get_tokenizer(args.tokenizer_name_or_path,
                              prompt_type=args.prompt_type,
                              prompt_type_path=args.prompt_type_path)
    splitter = build_splitter(args)

    logger.info(f"building dataset: {args.input}")
    raw_data = build_dataset(args)

    if args.n_subs == 1:
        handler = get_dataset_handler(args, raw_data, tokenizer, splitter)
        # serialize to bin&idx
        handler.serialize_to_disk()
    else:
        target_prefix = args.output_prefix
        target_prefixname = os.path.basename(target_prefix)

        num_samples = len(raw_data)
        start_ends = cut_range_to_subs(num_samples, num_samples // args.n_subs)
        subsets = [raw_data.select(range(x[0], x[1])) for x in start_ends]

        # multiprocessing
        params_list = []
        for k, subset in enumerate(subsets):
            args_ = copy.deepcopy(args)
            args_.output_prefix = target_prefix.replace(target_prefixname,
                                                        f'{str(k).zfill(3)}_of_{str(len(subsets) - 1).zfill(3)}_{target_prefixname}')
            params = [args_, subset, tokenizer, splitter]
            params_list.append(params)
        pool = multiprocessing.Pool()
        sub_idx_files = pool.map(handle_subset, params_list)
        pool.close()
        pool.join()

        for key in sub_idx_files[0].keys():
            idx_files = [x[key] for x in sub_idx_files]
            idx_files.sort()
            target_idx = idx_files[0].replace(f'000_of_{str(len(subsets) - 1).zfill(3)}_{target_prefixname}',
                                              target_prefixname)
            target_bin = target_idx.replace('.idx', '.bin')
            idx = IndexedDatasetBuilder(target_bin)
            for idx_file in idx_files:
                idx.add_index(idx_file.replace('.idx', ''))
            idx.finalize(target_idx)

            for idx_file in idx_files:
                os.remove(idx_file)
                os.remove(idx_file.replace('.idx', '.bin'))


@hydra.main(config_path="../configs/datasets", config_name=config_name)
def main(config):
    preprocess(config)


if __name__ == '__main__':
    main()
