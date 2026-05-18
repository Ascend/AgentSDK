#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import aiohttp
import asyncio

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class AgentProxyClient:
    def __init__(
        self,
        model_name: str,
        agent_addr: str,
        traj_addr: str,
        run_id: str,
        params: dict,
        timeout: int = 3600,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.agent_addr = agent_addr
        self.traj_addr = traj_addr
        self.run_id = run_id
        self.params = params
        self.traj_timeout = timeout  # TODO: 也可以通过params参数读取
        self.max_retries = max_retries

    async def get_agent_response(self, prompt_messages, application_id, sample_id) -> tuple[int, str]:
        payload = {
            "model": self.model_name,
            "messages": prompt_messages,
            "infer_params": self.params.get("infer_params", {}),
            "extra_params": self.params.get("extra_params", {}),
        }

        url = f"{self.agent_addr}/v1/chat/completions"
        logger.info(
            f"get_agent_response: url={url}, application_id={application_id}, sample_id={sample_id}, body={payload}"
        )

        for attempt in range(1, self.max_retries + 1):
            session_id = f"{application_id},{sample_id},{attempt}"
            if self.run_id:
                payload["infer_url"] = f"{self.traj_addr}/s/{self.run_id}/{session_id}/v1/"
            else:
                payload["infer_url"] = f"{self.traj_addr}/s/{session_id}/v1/"

            try:
                timeout_cfg = aiohttp.ClientTimeout(total=self.traj_timeout, connect=10)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(url, json=payload, ssl=False) as resp:
                        if resp.status == 200:
                            logger.info(f"get_agent_response success: session_id={session_id}")
                            return 0, session_id
                        else:
                            text = await resp.text()
                            raise RuntimeError(f"Task failed: {resp.status}, {text}")
            except Exception as e:
                logger.error(f"session_id:{session_id} [Retry {attempt}/{self.max_retries}] Error: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                else:
                    return 1, session_id


async def main():
    worker = AgentProxyClient("Qwen3-235B-thinking-2507", "http://100.105.162.81:28123", "https://127.0.0.1:8000")
    messages = [
        {"role": "system", "content": "你是一个严谨的助手"},
        {"role": "user", "content": "你好，请介绍一下自己"},
    ]
    params = {
        "extra_params": {"max_steps": 10, "max_model_len": 16384},
        "infer_params": {"temperature": 0.5, "top_p": 0.95, "top_k": 1, "max_tokens": 20},
    }
    result = await worker.get_agent_response(messages, "session123", params)
    logger.info(result)


asyncio.run
