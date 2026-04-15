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
from typing import Any

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class MessageQueueDMClient:
    """DataManager adapter for message queue operations."""

    def __init__(self, queue_actor: Any = None) -> None:
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """Put a sample into the queue."""
        logger.debug(f"put sample {param_version}: {sample}")

    async def put_validate(self, data: Any) -> bool:
        """Put validation data into the queue."""
        logger.debug(f"put validate: {data}")

    def get_validate_sync(self) -> Any | None:
        """Get validation data synchronously."""
        logger.debug("get_validate_sync")

    async def get_sample(self) -> Any | None:
        """Get a single sample from the queue."""
        logger.debug("get_sample")

    async def get_queue_size(self) -> int:
        """Get current queue size."""
        logger.debug("get_queue_size")

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)."""
        logger.debug("get_statistics")

    async def clear_queue(self) -> None:
        """Clear queue (async)."""
        logger.debug("clear_queue")

    async def shutdown(self) -> None:
        """Shutdown queue (async)."""
        logger.debug("shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)."""
        logger.debug("get_memory_usage")

    def put_sample_sync(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue (sync - deprecated, use put_sample instead)."""
        logger.debug("put_sample_sync")

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)."""
        logger.debug("get_sample_sync")

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)."""
        logger.debug("get_statistics_sync")

    def update_param_version_sync(self, version: int) -> None:
        """Update parameter version (sync - deprecated)."""
        logger.debug("update_param_version_sync")
