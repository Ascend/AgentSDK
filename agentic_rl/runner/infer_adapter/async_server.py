#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import json
import os
import socket
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict

import fastapi
import ray
import uvicorn
from starlette.requests import Request

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.globals import is_pd_separate
from agentic_rl.base.utils.work_mode import get_work_mode
from agentic_rl.runner.infer_router import InferRouter

logger = Loggers(__name__).get_logger()


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote
def _write_ranktable_on_node(ranktable_data: Dict[str, Any], file_path: str):
    file_dir = os.path.dirname(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ranktable_data, f, indent=4)
    return ray.get_runtime_context().get_node_id()


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            logger.info("FastAPI startup")
            self.server_ready.set()
            yield

            logger.info("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])
        app.router.add_api_route("/v1/completions", self.completions, methods=["POST"])
        app.router.add_api_route("/v1/workload", self.get_workload, methods=["GET"])
        app.router.add_api_route("/v1/cancel", self.cancel_requests, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    @abstractmethod
    async def chat_completion(self, raw_request: Request):
        """OpenAI chat completion API.

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    @abstractmethod
    async def completions(self, raw_request: Request):
        """OpenAI completions API.

        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def get_workload(self, raw_request: Request):
        """Get workload metrics from AsyncVLLMServer."""
        raise NotImplementedError

    @abstractmethod
    async def cancel_requests(self, raw_request: Request):
        """Get workload metrics from AsyncVLLMServer"""
        raise NotImplementedError


class AsyncServerProxyManager:
    """Used for shared-card and separated mode (pure inference)."""

    def __init__(
            self,
            tokenizer_name_or_path,
            worker_group,
            infer_service,
            *,
            scheduler_kwargs: Dict[str, Any] = None
    ):
        self.weight_offloaded = True
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.worker_group = worker_group
        self.server_addresses = None

        self.dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        if get_work_mode() == "hybrid" and self.worker_group is not None:
            self.rollout_tp_size = self.config.infer_tensor_parallel_size
            self.rollout_dp_size = self.worker_group.num_npus // self.rollout_tp_size // self.dp_size
        else:
            self.rollout_tp_size = 0
            self.rollout_dp_size = 0

        self.infer_router = None
        self.infer_service = infer_service

    async def init(self):
        unready_dp_ranks = set(range(self.rollout_dp_size))
        self.infer_router = await InferRouter.create()
        await self.infer_router.init(self.infer_service)
        await self.infer_router.wake_up(
            model_name=self.infer_service,
            kwargs_list=[
                {
                    "config": None,
                    "tokenizer_name_or_path": self.tokenizer_name_or_path,
                    "vllm_dp_size": self.rollout_dp_size,
                    "vllm_dp_rank": rollout_dp_rank,
                    "wg_prefix": "",
                }
                for rollout_dp_rank in unready_dp_ranks
            ]
        )

    def get_weight_offloaded(self):
        return self.weight_offloaded

    async def wake_up(self):
        """Wake up all instances."""
        await self.infer_router.wake_up(model_name=self.infer_service)
        self.weight_offloaded = False

    async def sleep(self):
        """Sleep all instances."""
        await self.infer_router.sleep(model_name=self.infer_service)
        self.weight_offloaded = True

    async def update_weights(self, path):
        kwargs_list = [{"path": path}]
        logger.info(f"update_weights, kwargs_list: {kwargs_list}")
        await self.infer_router.update_weights(model_name=self.infer_service, kwargs_list=kwargs_list)


class AsyncServerManager:
    """Used for separated mode. MindSpeed RL is not fully decoupled, training depends on this class initialization."""

    def __init__(self, config, tokenizer_name_or_path, worker_group, *, scheduler_kwargs: Dict[str, Any] = None):
        """Initialize AsyncServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
            scheduler_kwargs: Dict[str, Any], kwargs for chat scheduler.
        """
        self.weight_offloaded = True
        self.config = config
        self.worker_group = worker_group

        self.rollout_infer_backend = self.config.infer_backend
        logger.info(f"self.config.infer_backend: {self.config.infer_backend}")

        self.rollout_tp_size = self.config.infer_tensor_parallel_size
        dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        self.rollout_dp_size = self.worker_group.num_npus // self.rollout_tp_size // dp_size

        workers_info = ray.get(worker_group.execute_async_command("get_worker_info"))
        workers_info = {int(i): n for i, n in workers_info}

        self.async_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        from agentic_rl.runner.infer_adapter.infer_registry import async_server_class
        server_class = async_server_class(
            infer_backend=self.rollout_infer_backend,
        )
        logger.info(f"server_class: {server_class}")

        self.update_ranktable_from_workers_info(workers_info)

        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: ray.remote(server_class).options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size * dp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, tokenizer_name_or_path, self.rollout_dp_size, rollout_dp_rank, "")
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    if is_pd_separate():
                        infer_mode = ray.get(server.get_infer_mode.remote())
                        address = f"{infer_mode}-{address}"
                        logger.info(f"curr_address: {address}\ninfer_mode: {infer_mode}")
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except ray.exceptions.RayTaskError as e:
                    ray.kill(server)
                    print(
                        f"rollout server {rollout_dp_rank} failed due to {e}, "
                        f"maybe address already in use, restarting..."
                    )

        ray.get([server.init_engine.remote() for server in self.async_servers])

    def get_mapping_id_to_ip(self):
        nodes_info = {}
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call ray.init() first.")
        all_nodes = ray.nodes()
        for node in all_nodes:
            curr_node_id = node["NodeID"]
            curr_node_ip = node["NodeManagerAddress"]
            nodes_info[curr_node_id] = curr_node_ip

        return nodes_info

    def update_ranktable_from_workers_info(self, workers_info: dict):
        if not is_pd_separate():
            logger.info("Using common infer mode, there is no need to generate a ranktable.")
            return

        nodes_info = self.get_mapping_id_to_ip()

        prefill_device_cnt = int(os.getenv("P_INSTANCE_NUM_DEVICE", "0"))
        decode_device_cnt = int(os.getenv("D_INSTANCE_NUM_DEVICE", "0"))
        total_device_cnt = prefill_device_cnt + decode_device_cnt
        if len(workers_info) != total_device_cnt:
            raise ValueError(
                f"The actual number of devices ({len(workers_info)}) does not match "
                f"the device count: {total_device_cnt}"
            )

        file_path = os.getenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", None)
        if file_path is None:
            raise ValueError("DISAGGREGATED_PREFILL_RANK_TABLE_PATH must be set.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ranktable file must exist but not found: {file_path}")
        with open(file_path, 'r') as f:
            ranktable = json.load(f)
        original_devices = ranktable.get("prefill_device_list", []) + ranktable.get("decode_device_list", [])
        if len(original_devices) != total_device_cnt:
            raise ValueError(
                f"Ranktable has {len(original_devices)} devices, "
                f"but expected {total_device_cnt} from workers_info."
            )

        idx = 0
        new_prefill_device_list = []
        new_decode_device_list = []
        while idx < total_device_cnt:
            node_id = workers_info[idx]
            node_ip = nodes_info[node_id]
            prev_idx = idx
            for device_info in original_devices:
                if device_info['server_id'] == node_ip:
                    new_device_info = device_info.copy()
                    new_device_info['cluster_id'] = str(idx + 1)
                    if idx < prefill_device_cnt:
                        new_prefill_device_list.append(new_device_info)
                    else:
                        new_decode_device_list.append(new_device_info)
                    idx += 1
            if idx == prev_idx:
                raise ValueError(f"Node mismatch. Current node: {node_ip}. Original devices: {original_devices}")

        if total_device_cnt != len(new_prefill_device_list) + len(new_decode_device_list):
            raise ValueError("Device count mismatch after processing ranktable")
        logger.debug(f"\n\noriginal ranktable: \n\n{ranktable}\n\n")

        ranktable["prefill_device_list"] = new_prefill_device_list
        ranktable["decode_device_list"] = new_decode_device_list
        logger.info(f"\n\nupdated ranktable: \n\n{ranktable}\n\n")
        with open(file_path, 'w') as f:
            json.dump(ranktable, f, indent=4)

        self.rewrite_ranktable_to_all_nodes(workers_info, ranktable, file_path)

    def rewrite_ranktable_to_all_nodes(self, workers_info, ranktable, file_path):
        all_node_id = set()
        for idx, node_id in workers_info.items():
            all_node_id.add(node_id)
        write_futures = []
        for node_id in all_node_id:
            future = _write_ranktable_on_node.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            ).remote(ranktable, file_path)
            write_futures.append(future)

        written_nodes = ray.get(write_futures)
        logger.info(f"Ranktable written to nodes: {written_nodes}")

    def get_weight_offloaded(self):
        return self.weight_offloaded

    async def wake_up(self):
        """Wake up all instances."""
        ray.get([server.wake_up.remote() for server in self.async_servers])
        self.weight_offloaded = False

    async def sleep(self):
        """Sleep all instances."""
        ray.get([server.sleep.remote() for server in self.async_servers])
        self.weight_offloaded = True

    async def update_weights(self, path):
        for server in self.async_servers:
            await server.collective_rpc.remote("update_weights", args=path)

    async def reset_prefix_cache(self):
        ray.get([server.reset_prefix_cache.remote() for server in self.async_servers])

    async def vllm_statistics(self):
        for server in self.async_servers:
            await server.collective_rpc.remote("vllm_statistics")