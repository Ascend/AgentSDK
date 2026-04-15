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


# Standard library imports
import json
import random
from typing import Dict, List, Union, Callable, Optional, Tuple, Any

# Third-party library imports
import requests
from openai import AsyncOpenAI

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.base_infer_server import BaseInferServer

logger = Loggers(__name__).get_logger()


class VLLMProxyInferServer(BaseInferServer):
    # Support external launch of vLLM HTTP deployment instances by passing the HTTP address
    def __init__(self, *args, **kwargs):
        self._init_server(*args, **kwargs)
        logger.info(f"VLLMProxyInferServer init done: {self.model=}.")

    def _init_server(
            self,
            model_name,
            chat_server: Union[str, List],
            prefill_server_list: List = None,
            decode_server_list: List = None,
            *args, **kwargs
    ):
        logger.info(f"init server: {chat_server=}, {prefill_server_list=}, {decode_server_list=}.")
        self.model_name = model_name
        self.chat_server_list = chat_server if isinstance(chat_server, List) else [chat_server]
        self.prefill_server_list = prefill_server_list
        self.decode_server_list = decode_server_list

        self.client_list = [
            AsyncOpenAI(base_url=chat_server + '/v1/', api_key='EMPTY')
            for chat_server in self.chat_server_list
        ]
        self.model = model_name

    def _choose_server(self):
        return random.choice(self.client_list)

    async def launch_server(self, *args, **kwargs):
        self._init_server(*args, **kwargs)
        logger.info(f"VLLMProxyInferServer launch done: {self.model=}.")

    async def completions(self, request_data: Dict):
        request_data['logprobs'] = 1
        request_data['extra_body'] = {"return_token_ids": True}

        request_data.pop("stream", None)
        request_data['model'] = self.model

        if 'extra_headers' in request_data:
            request_data.pop('extra_headers')

        completion = await self._choose_server().completions.create(**request_data)

        return completion.model_dump()

    async def chat_completions(self, request_data: Dict):
        request_data['logprobs'] = 1
        request_data['extra_body'] = {"return_token_ids": True}

        request_data.pop("stream", None)
        request_data['model'] = self.model

        if 'extra_headers' in request_data:
            request_data.pop('extra_headers')
        completion = await self._choose_server().chat.completions.create(**request_data)

        return completion.model_dump()

    async def stream_chat_completions(self, request_data: Dict):

        request_data["stream"] = True
        request_data['model'] = self.model

        stream = await self._choose_server().chat.completions.create(**request_data)

        async for chunk in stream:
            json_string = json.dumps(chunk.model_dump(), ensure_ascii=False)
            yield json_string

    async def collective_rpc(
            self,
            method: Union[str, Callable],
            timeout: Optional[float] = 5,
            args: Tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> List:
        payload = {
            "method": method,
            "args": args if args is not None else [],
            "kwargs": kwargs if kwargs is not None else {},
        }

        server_list = []
        if self.prefill_server_list is not None or self.decode_server_list is not None:
            server_list.extend(self.prefill_server_list if self.prefill_server_list else [])
            server_list.extend(self.decode_server_list if self.prefill_server_list else [])
        else:
            server_list.extend(self.chat_server_list)

        result_list = []
        for server in server_list:
            collective_rpc_url = server + '/collective_rpc'
            try:
                logger.info(f"collective_rpc_url={collective_rpc_url}, payload={payload}")
                response = requests.post(
                    collective_rpc_url,
                    json=payload,
                    timeout=timeout + 5
                )

                response.raise_for_status()
                if response.status_code == 200:

                    result = response.json()
                    logger.info(f"collective_rpc_url={collective_rpc_url}, result={result}")
                    result_list.append(result)
                else:
                    raise RuntimeError(f"collective_rpc_url={collective_rpc_url}, response={response.status_code}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
        return []

    async def get_workload(self):
        pass

    async def cancel_requests(self, *args, **kwargs):
        pass
