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

import os
import json

HAS_HACO = True

try:
    from haco import Sentinel, parse_config
    import portalocker
except ImportError:
    HAS_HACO = False


def enable_haco(logger):
    if not HAS_HACO:
        logger.info("No module haco or module portalocker is found, haco monitor will be disabled")
    return HAS_HACO


def update_haco_rollout_master_ip(new_ip):
    # rollout sampler json
    file_path = os.environ.get("ROLLOUT_HACO_JSON", "./configs/rollout_raw_data_sampler.json")

    with open(file_path, 'r+', encoding='utf8') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        content = f.read()
        data = json.loads(content) if content else {}

        data["sampler"]["host_name"] = new_ip

        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4, ensure_ascii=False)


def actor_worker_update_haco(addr, model, optimizer):
    sentinel_config_path = os.environ.get("ACTOR_HACO_JSON", "./configs/accuracy/rollout_raw_data_sampler.json")
    sentinel_config = parse_config(sentinel_config_path)
    sentinel_config["sampler"]["host_name"] = addr
    update_haco_rollout_master_ip(addr)
    return Sentinel(sentinel_config, model=model, optimizer=optimizer)


def vllm_model_runner_update_haco(model):
    rollout_haco_json = os.environ.get("ROLLOUT_HACO_JSON", "../../configs/accuracy/rollout_raw_data_sampler.json")
    sentinel_configs = parse_config(rollout_haco_json)
    return Sentinel(sentinel_configs, model)
