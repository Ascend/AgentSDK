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
import argparse
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict

import yaml

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.ray_env_config import RayEnvVarsConfig

logger = Loggers(__name__)

RAY_GROUP_NAMESPACE = "agentic_raygroup"
_MAX_FILE_SIZE = 1024 * 1024  # 1MB


@contextmanager
def whitelist_environ():
    """
    Filter current environment variables by a whitelist.
    """
    whitelist = {
        "ASCEND_WORK_PATH",
        "ASCEND_AICPU_PATH",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASDOPS_LOG_LEVEL",
        "ASDOPS_LOG_PATH",
        "ASDOPS_LOG_TO_BOOST_TYPE",
        "ASDOPS_LOG_TO_FILE",
        "ASDOPS_LOG_TO_FILE_FLUSH",
        "ASDOPS_LOG_TO_STDOUT",
        "ATB_COMPARE_TILING_EVERY_KERNEL",
        "ATB_DEVICE_TILING_BUFFER_BLOCK_NUM",
        "ATB_HOME_PATH",
        "ATB_HOST_TILING_BUFFER_BLOCK_NUM",
        "ATB_MATMUL_SHUFFLE_K_ENABLE",
        "ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT",
        "ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT",
        "ATB_OPSRUNNER_SETUP_CACHE_ENABLE",
        "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE",
        "ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE",
        "ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE",
        "ATB_SHARE_MEMORY_NAME_SUFFIX",
        "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE",
        "ATB_WORKSPACE_MEM_ALLOC_GLOBAL",
        "HOME",
        "LCCL_DETERMINISTIC",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "TOOLCHAIN_HOME",
    }

    old_env = os.environ.copy()

    os.environ.clear()

    for k in whitelist:
        if k in old_env:
            os.environ[k] = old_env[k]

    yield


def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        FileCheck.check_data_path_is_valid(config_path)
        FileCheck.check_file_size(config_path, _MAX_FILE_SIZE)
    except ValueError as e:
        logger.error(f"Checking config_path failed with value not correct, error: {e}")
        sys.exit(1)
    except TypeError as e:
        logger.error(f"Checking config_path failed with type not correct, error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred when checking config_path, error: {e}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                logger.error(f"config file content is not a dict, but {type(cfg)}")
                sys.exit(1)
            return cfg
    except yaml.YAMLError as e:
        logger.error(f"load config failed of yaml content, err: {e}")
        sys.exit(1)
    except RecursionError as e:
        logger.error(f"failed to parse yaml file, nesting depth is too deep or a circular alias was detected, err: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred when load config, error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", type=str, required=True, help="path of config yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config_path)

    with whitelist_environ():
        try:
            from agentic_rl.base.utils.logger_patch import patch

            patch()

            import ray
            from ray.exceptions import RayError
        except ImportError:
            logger.error("ray is not installed, please install ray first.")
            sys.exit(1)

        from agentic_rl.base.utils.ray_secure_init import ray_secure_init
        ray_envs = RayEnvVarsConfig().to_env_dict()
        if ray.is_initialized():
            logger.error(f"ray should be initialized by agentic_rl, but has already been initialized.")
            sys.exit(1)
        else:
            logger.info('start initializing local ray cluster, when the ray cluster is not initialized')
            ray_secure_init(extra_init_kwargs={'runtime_env': ray_envs, 'namespace': RAY_GROUP_NAMESPACE})
            ray_initialized_by_us = True

        try:
            from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train
            ray.get(train.remote(cfg))
        except RayError as e:
            logger.error(f"Training using mindspeed-rl failed with ray, error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error occurred when training using mindspeed-rl, error: {e}")
            sys.exit(1)
        finally:
            if ray_initialized_by_us and ray.is_initialized():
                logger.info("Shutting down ray cluster.")
                ray.shutdown()


if __name__ == "__main__":
    main()
