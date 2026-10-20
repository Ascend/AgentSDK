# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
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
#
import os
import threading

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.trainer.main_ppo import run_ppo

from aura.base.log.loggers import Loggers
from aura.trainer.train_adapter.verl.full_async.train_main import FullyAsyncTaskRunner

logger = Loggers(__name__).get_logger()


@ray.remote
def start_train(work_mode, train_config, rollout_config, agent_service, infer_service):
    """Launch fully-async training: create the SampleQueue, start the rollout,
    then run PPO with ``FullyAsyncTaskRunner``."""
    from aura.trainer.rollout.rollout_service import start_fully_async_rollout
    from aura.controllers.rollout_controller.sample_queue import create_sample_queue

    async_conf = _require_async_config(train_config)
    required_samples, max_required_samples = _compute_sample_bounds(train_config, async_conf)
    logger.info(
        f"fully_async max_required_samples={max_required_samples} "
        f"(required_samples={required_samples})"
    )

    create_sample_queue(max_queue_size=max_required_samples)
    _inject_max_required_samples(rollout_config, max_required_samples)

    ref = start_fully_async_rollout.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )
    ).remote(rollout_config=rollout_config, agent_service=agent_service, infer_service=infer_service)

    _launch_rollout_monitor(ref)

    logger.info(f"fully_async mode, {train_config=}, {work_mode=}")

    from time import time

    start_time = time()
    run_ppo(train_config, task_runner_class=FullyAsyncTaskRunner)
    logger.info(f"fully_async total time: {time() - start_time:.2f} seconds")


def _require_async_config(train_config):
    """Return the ``async_training`` config or raise if missing."""
    async_conf = getattr(train_config, "async_training", None)
    if not async_conf:
        raise RuntimeError("async_training config is required for fully_async mode")
    return async_conf


def _compute_sample_bounds(train_config, async_conf):
    """Compute ``required_samples`` and ``max_required_samples`` for the SampleQueue."""
    staleness_threshold = float(getattr(async_conf, "staleness_threshold", 0))
    trigger_step = int(getattr(async_conf, "trigger_parameter_sync_step", 1))
    require_batches = int(getattr(async_conf, "require_batches", 1))
    ppo_mini_batch_size = int(getattr(train_config.actor_rollout_ref.actor, "ppo_mini_batch_size", 1))
    required_samples = ppo_mini_batch_size * require_batches
    max_required_samples = int(required_samples * (1 + staleness_threshold) * trigger_step)
    return required_samples, max_required_samples


def _inject_max_required_samples(rollout_config, max_required_samples):
    """Write ``max_required_samples`` into either a dict or OmegaConf rollout config."""
    if isinstance(rollout_config, dict):
        rollout_config["max_required_samples"] = max_required_samples
        return
    if rollout_config is None:
        return
    try:
        from omegaconf import OmegaConf
        OmegaConf.update(rollout_config, "max_required_samples", max_required_samples)
    except Exception as e:
        logger.debug(f"failed to inject max_required_samples into rollout_config: {e}")


def _launch_rollout_monitor(ref):
    """Start a daemon thread that exits the process if the rollout fails."""

    def monitor_rollout():
        try:
            ray.get(ref)
        except Exception as e:
            logger.error(f"fully_async rollout failed: {e}", exc_info=True)
            os._exit(1)

    threading.Thread(target=monitor_rollout, daemon=True).start()
