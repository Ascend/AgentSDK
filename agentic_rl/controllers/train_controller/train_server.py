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
import io
import json
import time

import ray
import torch
from fastapi import UploadFile, File, Form, APIRouter
from starlette.responses import Response

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.train_controller.train_queue import TrainQueue
from agentic_rl.controllers.utils.msg_handler import deserialize_and_split

DEFAULT_CHUNK = 1 << 19  # 512 kb
DEFAULT_BATCH_SIZE = 1


class TrainServer:
    def __init__(self, max_queue_size, global_batch_size, n_samples_per_prompt):
        # nothing async here; no get_running_loop()
        self.logger = Loggers(__name__).get_logger()
        try:
            self.dispatch_actor = ray.get_actor("dispatch", namespace="controller_raygroup")
            self.queue_instance = TrainQueue(
                max_queue_size=max_queue_size,
                global_batch_size=global_batch_size,
                n_samples_per_prompt=n_samples_per_prompt
            )
        except Exception as e:
            self.logger.error(f"failed to get ray actor in Train Server: {e}")

        self.router = APIRouter(prefix="/train", tags=["Train Control"])
        self.router.post("/send_minibatch")(self.receive_minibatch)
        self.router.post("/is_batch_ready")(self.is_batch_ready)
        self.router.get("/get_minibatch")(self.pop_minibatch)
        self.router.post("/is_ready")(self.is_ready)

    def put_minibatch_to_queue(self, outputs, metric):
        self.logger.info("putting minibatch to queue...")
        not_full = self.queue_instance.add_minibatch(outputs, metric)
        self.logger.info(f"put minibatch to queue complete  {self.queue_instance.size()=} ...")
        if not not_full:
            # The training end cannot handle the consumption, so suspend the inference for now.
            self.logger.info("queue is full, locking rollout unit...")
            self.dispatch_actor.lock_rollout_unit()
            return
        # Continue to send batches to the rollout units
        self.logger.info("queue is not full, sending batch to rollout unit...")
        self.dispatch_actor.send_batch_groups.remote(DEFAULT_BATCH_SIZE)

    async def receive_minibatch(
            self,
            file: UploadFile = File(...),
            rollout_cost: float = Form(...),
            resharding_to_infer: float = Form(...),
            toolcall_reward_mean: float = Form(...),
            toolcall_reward_min: float = Form(...),
            toolcall_reward_max: float = Form(...),
            res_reward_mean: float = Form(...),
            res_reward_min: float = Form(...),
            res_reward_max: float = Form(...)
    ):
        print(f"receive_minibatch ...")
        self.logger.info("receive minibatch...")
        start_time = time.time()
        buf = io.BytesIO()
        chunk = await file.read(DEFAULT_CHUNK)
        while chunk:
            buf.write(chunk)
            chunk = await file.read(DEFAULT_CHUNK)
        buf.seek(0)
        # heavy torch.load in a process pool
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, deserialize_and_split, buf)

        metric = {
            "rollout_cost": rollout_cost,
            "resharding_to_infer": resharding_to_infer,
            "toolcall_reward_mean": toolcall_reward_mean,
            "toolcall_reward_min": toolcall_reward_min,
            "toolcall_reward_max": toolcall_reward_max,
            "res_reward_mean": res_reward_mean,
            "res_reward_min": res_reward_min,
            "res_reward_max": res_reward_max,
        }
        self.put_minibatch_to_queue(outputs, metric)
        await file.close()
        buf.close()
        self.logger.info(f"|perf-stat|train| receive minibatch end, cost={time.time() - start_time:.2f} s")
        return {"status": "ok"}

    async def is_batch_ready(self):
        is_ready = False
        if self.queue_instance.size() > 0:
            is_ready = True
        return {"is_ready": is_ready}

    async def pop_minibatch(self):
        self.logger.info(f"get a training batch ...")
        start_time = time.time()
        outputs, metric = self.queue_instance.pop_batch()
        buf = io.BytesIO()
        torch.save(outputs, buf)
        buf.seek(0)
        file_bytes = buf.getvalue()
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": "attachment; filename=data.bin",
            "X-Metrics-Metadata": json.dumps(metric)
        }
        self.logger.info(
            f"|perf-stat|train| get a training batch end, {self.queue_instance.size()=} cost={time.time() - start_time:.2f} s")
        return Response(content=file_bytes, headers=headers)

    async def is_ready(self):
        self.logger.info(f"rollout unit is ready")
        await self.dispatch_actor.set_rollout_unit_ready.remote()
        return {"status": "ok"}
