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


from typing import Dict, Union, Callable, Optional, Tuple, Any

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class BaseInferServer:
    async def completions(self, request_data: Dict):
        pass

    async def chat_completions(self, request_data: Dict):
        pass

    async def stream_chat_completions(self, request_data: Dict):
        pass

    async def launch_server(self, *args, **kwargs):
        logger.info(f"BaseInferServer launch")

    async def wake_up(self, *args, **kwargs):
        logger.info(f"BaseInferServer wake_up")

    async def sleep(self, *args, **kwargs):
        logger.info(f"BaseInferServer sleep")

    async def collective_rpc(
            self,
            method: Union[str, Callable],
            timeout: Optional[float] = None,
            args: Tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
    ):
        logger.info(f"BaseInferServer collective_rpc")
