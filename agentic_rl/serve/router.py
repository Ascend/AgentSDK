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


from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse

from agentic_rl.base.exceptions.exceptions import async_raise_http_exception
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask

logger = Loggers(__name__).get_logger()

router = APIRouter(prefix="")


@router.post("/agent/invoke")
@async_raise_http_exception
async def agent_invoke(request: Request):
    """
    Agent Invoke API: Returns in SSE streaming mode (chunked push)
    """

    from agentic_rl.runner.agent_router import AgentRouter
    agent_router: AgentRouter = await AgentRouter.create()

    request_data = await request.json()
    task = AgentTask(**request_data)
    return EventSourceResponse(agent_router.stream_generate_trajectory(task))


@router.post("/v1/chat/completions")
@async_raise_http_exception
async def chat_completions(request: Request):
    """
    Standard Chat Completions API:
    - stream=False -> Returns complete result at once (JSON)
    - stream=True -> Returns in SSE streaming mode (chunked push)
    """
    from agentic_rl.runner.infer_router import InferRouter
    infer_router: InferRouter = await InferRouter.create()

    request_data = await request.json()
    if request_data["stream"]:
        return EventSourceResponse(infer_router.stream_chat_completions(request_data))
    return await infer_router.chat_completions(request_data)
