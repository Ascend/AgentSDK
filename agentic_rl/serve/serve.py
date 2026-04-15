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


import asyncio

from fastapi import FastAPI
from ray import serve

from agentic_rl.serve.router import router

app = FastAPI()
app.include_router(router)


@serve.deployment
@serve.ingress(app)
class AgenticAIDeployment:
    @app.get("/")
    async def root(self):
        return "welcome to agentic ai!"

    @app.get("/delay")
    async def root(self):
        await asyncio.sleep(2)
        return "welcome to agentic ai!"


deployment = AgenticAIDeployment.bind()
