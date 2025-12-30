#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import time

import ray
from ray.exceptions import RayError
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig
from agentic_rl.runner.infer_adapter.async_server_base import AsyncServerBase
from agentic_rl.runner.infer_adapter.infer_registry import async_server_class
from agentic_rl.base.utils.file_utils import FileCheck

logger = Loggers(__name__)

# Re-export AsyncServerBase for backward compatibility
__all__ = ["AsyncServerBase", "AsyncServerManager"]


def _safe_kill_ray(server_ref):
    try:
        ray.kill(server_ref)
    except ValueError as e:
        logger.warning(f"Failed to kill ray: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error occurred, failed to kill ray: {e}")


class AsyncServerManager:
    """AsyncServerManager manage a group of infer instances."""

    MAX_SERVER_START_RETRIES: int = 5
    SERVER_HEALTH_CHECK_TIMEOUT_SECONDS: int = 30

    def __init__(
            self,
            config,
            agentic_rl_config,
            tokenizer_name_or_path,
            worker_group
    ):
        """Initialize AsyncServerManager."""
        from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
        if not isinstance(config, GenConfig):
            raise ValueError(f"config must be a GenConfig, but got {type(config)}")
        if not isinstance(agentic_rl_config, AgenticRLConfig):
            raise ValueError(f"agentic_rl_config must be a AgenticRLConfig, but got {type(agentic_rl_config)}")
        if not isinstance(worker_group, RayActorGroup):
            raise ValueError(f"worker_group must be a RayActorGroup, but got {type(worker_group)}")
        FileCheck.check_data_path_is_valid(tokenizer_name_or_path)
        self.config = config
        self.agentic_rl_config = agentic_rl_config
        self.worker_group = worker_group

        self.rollout_infer_backend = self.agentic_rl_config.infer_backend
        logger.info(f"self.config.infer_backend: {self.agentic_rl_config.infer_backend}")

        self.rollout_tp_size = self.config.infer_tensor_parallel_size

        try:
            self.rollout_dp_size = self.worker_group.num_npus // self.rollout_tp_size
        except ZeroDivisionError as e:
            raise RuntimeError("rollout_tp_size is zero") from e
        except Exception as e:
            raise RuntimeError(f"Failed to calculate rollout_dp_size: {e}") from e

        # Retrieve worker info from Ray actors
        try:
            workers_info = ray.get(worker_group.execute_async_command("get_worker_info"))
            workers_info = {int(i): n for i, n in workers_info}
        except (ValueError, TypeError, RayError) as e:
            raise RuntimeError(f"Failed to retrieve worker info: {e}") from e

        self.async_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        # Get server class for the specified inference backend
        server_class = async_server_class(infer_backend=self.rollout_infer_backend)
        if server_class is None:
            raise ValueError(f"Unsupported inference backend: '{self.rollout_infer_backend}'. "
                             f"Please check if the backend is registered in the infer_registry.")
        logger.info(f"server_class: {server_class}")

        try:
            self._start_server_instances(server_class, workers_info, config, agentic_rl_config, tokenizer_name_or_path)
        except RayError as e:
            self._cleanup_servers()
            raise RuntimeError(f"Failed to start server instances: {e}") from e

        # All server instances are ready, init AsyncLLM engine.
        try:
            ray.get([server.init_engine.remote() for server in self.async_servers])
        except RayError as e:
            self._cleanup_servers()
            raise RuntimeError(f"Failed to initialize AsyncLLM engines: {e}") from e

    def wake_up(self):
        """Wake up all instances."""
        try:
            ray.get([server.wake_up.remote() for server in self.async_servers])
            logger.info("All server instances woken up successfully")
        except RayError as e:
            raise RuntimeError(f"Failed to wake up server instances: {e}") from e

    def sleep(self):
        """Sleep all instances."""
        try:
            ray.get([server.sleep.remote() for server in self.async_servers])
            logger.info("All server instances put to sleep successfully")
        except RayError as e:
            raise RuntimeError(f"Failed to put server instances to sleep: {e}") from e

    def _cleanup_servers(self):
        """Clean up partially created server actors."""
        if not hasattr(self, 'async_servers') or self.async_servers is None:
            return

        logger.info("Cleaning up server actors...")
        for i, server in enumerate(self.async_servers):
            if server is not None:
                _safe_kill_ray(server)

        # Reset server lists
        self.async_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size
        logger.info("Server cleanup completed")

    def _start_server_instances(self, server_class, workers_info, config, agentic_rl_config, tokenizer_name_or_path):
        dp_size = 1  # reference RayEnvVarsConfig().VLLM_DP_SIZE
        unready_dp_ranks = set(range(self.rollout_dp_size))
        logger.info(
            f"Starting {self.rollout_dp_size} server instances with up to {self.MAX_SERVER_START_RETRIES} retries...")

        for attempt in range(self.MAX_SERVER_START_RETRIES):
            if not unready_dp_ranks:
                break
            servers = {}
            for rollout_dp_rank in unready_dp_ranks:
                node_id = workers_info[rollout_dp_rank * self.rollout_tp_size * dp_size]
                logger.debug(f"Creating server at rank {rollout_dp_rank} on node {node_id}")

                try:
                    server = server_class.options(
                        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                            node_id=node_id,
                            soft=False,
                        ),
                        name=f"async_llm_server_{rollout_dp_rank}",
                    ).remote(
                        config, agentic_rl_config, tokenizer_name_or_path,
                        self.rollout_dp_size, rollout_dp_rank, ""
                    )
                    servers[rollout_dp_rank] = server
                except RayError as e:
                    logger.warning(
                        f"Failed to create server at rank {rollout_dp_rank} on node {node_id}: "
                        f"{type(e).__name__}: {e}"
                    )

            for rollout_dp_rank, server in servers.items():
                try:
                    ray.get(server.__ray_ready__.remote(), timeout=self.SERVER_HEALTH_CHECK_TIMEOUT_SECONDS)
                    self.server_addresses[rollout_dp_rank] = server
                    self.async_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except RayError as e:
                    logger.warning(f"Rollout server at rank {rollout_dp_rank} failed: "
                                   f"{type(e).__name__}: {e}. Maybe address already in use, restarting...")
                    _safe_kill_ray(server)

            if unready_dp_ranks:
                if attempt == self.MAX_SERVER_START_RETRIES - 1:
                    raise RuntimeError(
                        f"Failed to start server instances after {self.MAX_SERVER_START_RETRIES} attempts. "
                        f"Failed ranks: {sorted(unready_dp_ranks)}"
                    )
                # Brief backoff before next attempt to reduce address-in-use races
                time.sleep(0.5)
