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

import os
from typing import Optional
import ray
from agentic_rl.base.log.loggers import Loggers


def get_local_rank(logger_name: Optional[str] = None) -> int:
    """
    Determine the local rank based on the runtime context.
    - If launched via `torchrun`, the `LOCAL_RANK` environment variable is used.
    - If launched via `ray`, the rank is obtained from the ray runtime context.
    - If neither is available, defaults to 0 (for testing or single-process scenarios).

    Args:
        logger_name: Optional logger name for logging. Defaults to module name.

    Returns:
        int: The local rank of the current process.

    Raises:
        ValueError: If logger_name is not a string.
    """
    # Validate logger_name parameter
    if logger_name is not None and not isinstance(logger_name, str):
        raise ValueError(f"logger_name must be a str, got {type(logger_name)}")

    # Lazy logger initialization - only create when needed for logging
    logger = None

    def _get_logger():
        nonlocal logger
        if logger is None:
            logger = Loggers(logger_name) if logger_name else Loggers(__name__)
        return logger

    # Priority 1: Check LOCAL_RANK environment variable
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        try:
            local_rank = int(local_rank_env)
            if local_rank < 0 or local_rank >= 8:
                raise ValueError("LOCAL_RANK must be in [0, 8)")

            _get_logger().info(f"Local rank determined from LOCAL_RANK env var: {local_rank}")
            return local_rank
        except ValueError as e:
            _get_logger().warning(f"Invalid LOCAL_RANK value: {e}")

    # Priority 2: Try Ray runtime context
    try:
        runtime_context = ray.get_runtime_context()
        accelerator_ids = runtime_context.get_accelerator_ids()
        
        if "NPU" in accelerator_ids and accelerator_ids["NPU"]:
            local_rank = int(accelerator_ids["NPU"][0])
            os.environ["LOCAL_RANK"] = str(local_rank)
            _get_logger().info(f"Local rank determined from Ray runtime context: {local_rank}")
            return local_rank
        else:
            _get_logger().warning("Ray runtime context available but no NPU accelerator IDs found")
            
    except (RuntimeError, KeyError, IndexError, ValueError) as e:
        _get_logger().warning(f"Could not get local rank from Ray runtime context: {e}")
    except Exception as e:
        _get_logger().warning(f"Unexpected error accessing Ray runtime context: {e}")

    # Priority 3: Default to 0
    _get_logger().info("Unable to determine local rank. Defaulting to 0.")
    return 0
