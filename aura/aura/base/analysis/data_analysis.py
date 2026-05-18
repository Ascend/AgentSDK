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

import torch
import os
import time
import json

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def enable_data_debug():
    data_debug = os.getenv("ENABLE_DATA_DEBUG", 'false')
    if data_debug.lower() == 'true':
        return True
    return False


def get_path():
    path = "./data_analysis/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_current_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def torch_save_data(data, prefix, iteration=0):
    if not enable_data_debug():
        return

    path = get_path()
    file_timestamp = get_current_timestamp()
    filename = f"{prefix}_{iteration}_{file_timestamp}.pt"
    full_file = os.path.join(path, filename)
    torch.save(data, full_file)
    logger.info(f"torch save data to {full_file} done")


def convert_to_string(value):
    if isinstance(value, torch.Tensor):
        return str(value.tolist())
    elif isinstance(value, list):
        return [convert_to_string(v) for v in value]
    elif isinstance(value, dict):
        return {key: convert_to_string(v) for key, v in value.items()}
    return str(value)


def json_save_data(data, prefix, iteration=0):
    if not enable_data_debug():
        return

    add_iter = {"iteration": iteration, f"{prefix}": data}
    data_str = convert_to_string(add_iter)
    path = get_path()
    file_timestamp = get_current_timestamp()
    full_file = os.path.join(path, f'rollout_{prefix}_{file_timestamp}.json')
    with open(full_file, 'a') as f:
        # noinspection PyTypeChecker
        json.dump(data_str, f, indent=4, ensure_ascii=False)
        f.write('\n')
        logger.info(f"dump data to {full_file} done")
