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

import yaml

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls import ExtendedGenerateConfig

logger = Loggers(__name__).get_logger()

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = CURRENT_PATH[:CURRENT_PATH.find("/agentic_rl/base/utils")]
THIRD_PARTY_PATH = ROOT_PATH + "/third_party"
MINDSPEED_RL_PATH = THIRD_PARTY_PATH + "/rl/mindspeed_rl"
MINDSPEED_PATH = THIRD_PARTY_PATH + "/rl/mindspeed"
MINDSPEED_LLM_PATH = THIRD_PARTY_PATH + "/rl/mindspeed_llm"
MEGATRON_PATH = THIRD_PARTY_PATH + "/rl/megatron"
VLLM_PATH = THIRD_PARTY_PATH + "/infer/vllm"
VLLM_ASCEND_PATH = THIRD_PARTY_PATH + "/infer/vllm_ascend"
RLLM__PATH = THIRD_PARTY_PATH + "/agent_engine/rllm"
CONFIGS_PATH = ROOT_PATH + "/configs/msrl_conf"


def get_thirty_path():
    run_env = {
        "env_vars": {
            "PYTHONPATH":
                f"{MINDSPEED_RL_PATH}:{MINDSPEED_PATH}:{MINDSPEED_LLM_PATH}:{MEGATRON_PATH}:"
                f"{VLLM_PATH}:{VLLM_ASCEND_PATH}:"
                f"{RLLM__PATH}:"
                f"{THIRD_PARTY_PATH}:"
                f"$PYTHONPATH"
        }
    }
    return run_env


def load_runtime_env():
    with open(CONFIGS_PATH + "/envs/runtime_env.yaml") as file:
        runtime_env = yaml.safe_load(file)
    return runtime_env


def get_runtime_env(config):
    runtime_env = load_runtime_env()
    logger.info(f"ray init with runtime_env: {runtime_env}")

    generate_config = ExtendedGenerateConfig(config.get("generate_config"))
    if runtime_env["env_vars"]["TASK_QUEUE_ENABLE"] == '2' and (not generate_config.enforce_eager):
        runtime_env["env_vars"]["TASK_QUEUE_ENABLE"] = '1'
        logger.info("change TASK_QUEUE_ENABLE to 1 because enforce_eager is False")
    if (not generate_config.enforce_eager) and (not generate_config.enable_sleep_mode):
        raise "enable_sleep_mode need to be true when enforce_eager is false!!!"
    return runtime_env


def get_vllm_version():
    runtime_env = load_runtime_env()
    return runtime_env['env_vars']['VLLM_VERSION']
