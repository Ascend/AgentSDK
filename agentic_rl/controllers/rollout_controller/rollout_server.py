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
import io
import time

import torch
from fastapi import Request, UploadFile, File, HTTPException, APIRouter

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.http_status import HTTP_ERROR_400

logger = Loggers(__name__).get_logger()


class RolloutServer:
    def __init__(self, rollout_queue, rollout_weight_manager):
        self.running = False
        self.rollout_queue = rollout_queue
        self.rollout_weight_manager = rollout_weight_manager
        self.is_shutdown = False

        self.logger = Loggers(__name__).get_logger()

        self.router = APIRouter(prefix="/rollout", tags=["Rollout Control"])
        self.router.post("/unlock")(self.unlock)
        self.router.post("/lock")(self.lock)
        self.router.post("/send_batch")(self.receive_batch)
        self.router.post("/notify_weights_update")(self.handle_weights_update)
        self.router.post("/shutdown")(self.shutdown)

    async def unlock(self):
        self.running = True
        await self.rollout_queue.set_running.remote()
        return {"Status": "Running"}

    async def lock(self):
        self.running = False
        await self.rollout_queue.set_block.remote()
        return {"Status": "Blocked"}

    async def receive_batch(self, file: UploadFile = File(...)):
        self.logger.info("receive batch from trainer...")
        start_time = time.time()
        contents = await file.read()
        buf = io.BytesIO(contents)
        try:
            batch = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as e:
            self.logger.error(f"torch load error: {e}")
            raise HTTPException(status_code=HTTP_ERROR_400, detail=f"Invalid torch file: {e}")

        await self.rollout_queue.add_queue.remote(batch)
        self.logger.info(f"receive batch end, cost: {time.time() - start_time:.2f} s")
        return {"Status": "ok"}

    async def handle_weights_update(self, request: Request):
        data = await request.body()
        path = data.decode("utf-8")

        self.logger.info(f"start update weights from {path}")
        self.rollout_weight_manager.sync_weights_update.remote(path)
        return {"received": path}

    async def shutdown(self):
        await self.rollout_queue.shutdown.remote()
        self.is_shutdown = True
        logger.info(f"Rollout Server shutdown")
        return {"Status": "ok"}
