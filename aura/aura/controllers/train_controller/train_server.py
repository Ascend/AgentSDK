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
from fastapi import UploadFile, File, Form, APIRouter, HTTPException, Request
from starlette.responses import Response

from aura.base.log.loggers import Loggers
from aura.controllers.train_controller.train_queue import TrainQueue
from aura.controllers.utils.msg_handler import deserialize_and_split

DEFAULT_CHUNK = 1 << 19  # 512 kb
DEFAULT_BATCH_SIZE = 2


class TrainServer:
    def __init__(self, max_queue_size, global_batch_size, n_samples_per_prompt):
        # nothing async here; no get_running_loop()
        self.logger = Loggers(__name__).get_logger()
        self.replay_demand_mode = False
        self._batch_ready = asyncio.Event()
        self._recovery_frozen = False
        try:
            self.dispatch_actor = ray.get_actor("dispatch", namespace="controller_raygroup")
            self.queue_instance = TrainQueue(
                max_queue_size=max_queue_size,
                global_batch_size=global_batch_size,
                n_samples_per_prompt=n_samples_per_prompt,
            )
        except Exception as e:
            self.logger.error(f"failed to get ray actor in Train Server: {e}")

        self.router = APIRouter(prefix="/train", tags=["Train Control"])
        self.router.post("/send_minibatch")(self.receive_minibatch)
        self.router.post("/is_batch_ready")(self.is_batch_ready)
        self.router.post("/wait_batch_ready")(self.wait_batch_ready)
        self.router.get("/get_minibatch")(self.pop_minibatch)
        self.router.post("/is_ready")(self.is_ready)
        self.router.post("/replay/freeze")(self.freeze_replay_queue)
        self.router.post("/replay/restore")(self.restore_replay_queue)
        self.router.post("/replay/resume")(self.resume_replay_queue)

    async def freeze_replay_queue(self):
        if not self.replay_demand_mode:
            raise HTTPException(status_code=409, detail='Recovery requires replay demand mode')
        self._recovery_frozen = True
        return self._replay_queue_state()

    def _replay_queue_state(self):
        from aura.memory.replay.source import plain
        if len(self.queue_instance.queue) != len(self.queue_instance.metric_queue):
            raise RuntimeError('TrainQueue data/metric FIFO mismatch')
        return {'batches': [{'replay_refs': plain(raw['replay_refs'])} for raw in self.queue_instance.queue],
                'metrics': plain(list(self.queue_instance.metric_queue))}

    async def restore_replay_queue(self, request: Request):
        if not self.replay_demand_mode or self.queue_instance.size() or not self._recovery_frozen:
            raise HTTPException(status_code=409, detail='Restore requires an empty frozen queue')
        from aura.memory.replay.codec import strict_json
        value = strict_json(await request.body(), max_bytes=16*1024*1024)
        if not isinstance(value, dict) or set(value) != {'batches', 'metrics'}:
            raise HTTPException(status_code=400, detail='Recovery queue must contain batches and metrics fields')
        if not isinstance(value['batches'], list) or not isinstance(value['metrics'], list):
            raise HTTPException(status_code=400, detail='Recovery batches and metrics must be lists')
        if len(value['batches']) > 1:
            raise HTTPException(status_code=400, detail='Recovery queue allows at most one batch')
        if len(value['batches']) != len(value['metrics']):
            raise HTTPException(status_code=400, detail='Recovery batches and metrics lengths differ')
        for index, batch in enumerate(value['batches']):
            if not isinstance(batch, dict) or 'replay_refs' not in batch:
                raise HTTPException(status_code=400, detail=f'Recovery batch {index} requires replay_refs')
        for index, metric in enumerate(value['metrics']):
            if not isinstance(metric, dict):
                raise HTTPException(status_code=400, detail=f'Recovery metric {index} must be an object')
        for raw, metric in zip(value['batches'], value['metrics']):
            self.queue_instance.add_minibatch({'replay_refs': raw['replay_refs']}, metric)
        if value['batches']:
            self._batch_ready.set()
        return self._replay_queue_state()

    async def resume_replay_queue(self):
        if not self._recovery_frozen:
            raise HTTPException(status_code=409, detail='Queue was not frozen')
        self._recovery_frozen = False
        return {'resumed': True}

    def put_minibatch_to_queue(self, outputs, metric):
        if getattr(self, '_recovery_frozen', False):
            raise RuntimeError('Unexpected late arrival at frozen TrainQueue')
        self.logger.info("putting minibatch to queue...")
        not_full = self.queue_instance.add_minibatch(outputs, metric)
        self._batch_ready.set()
        if self.replay_demand_mode:
            return  # No automatic credit for replay-only steps or pending online groups.
        self.logger.info("put minibatch to queue complete ...")
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
        res_reward_max: float = Form(...),
    ):
        print("receive_minibatch ...")
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

    async def wait_batch_ready(self, timeout: float = 30.0):
        if not 0 < timeout <= 30:
            raise HTTPException(status_code=400, detail="timeout must be in (0, 30]")
        # Receive and wait handlers use the same event loop. No yield between
        # checking the queue and clearing the event, so arrivals cannot be lost.
        if self.queue_instance.size() == 0:
            self._batch_ready.clear()
            try:
                await asyncio.wait_for(self._batch_ready.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        return {"is_ready": self.queue_instance.size() > 0}

    async def is_batch_ready(self):
        is_ready = False
        if self.queue_instance.size() > 0:
            is_ready = True
        return {"is_ready": is_ready}

    async def pop_minibatch(self):
        if self._recovery_frozen:
            raise HTTPException(status_code=409, detail='TrainQueue is frozen')
        self.logger.info("get a training batch ...")
        start_time = time.time()
        outputs, metric = self.queue_instance.pop_batch()
        buf = io.BytesIO()
        torch.save(outputs, buf)
        buf.seek(0)
        file_bytes = buf.getvalue()
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": "attachment; filename=data.bin",
            "X-Metrics-Metadata": json.dumps(metric),
        }
        self.logger.info(f"|perf-stat|train| get a training batch end, cost={time.time() - start_time:.2f} s")
        return Response(content=file_bytes, headers=headers)

    async def is_ready(self):
        self.logger.info("rollout unit is ready")
        await self.dispatch_actor.set_rollout_unit_ready.remote()
        return {"status": "ok"}
