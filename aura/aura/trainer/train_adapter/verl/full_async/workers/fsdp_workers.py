# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from time import time

import ray
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class FsdpDetachActorWorker(DetachActorWorker):
    """FSDP-based actor worker that supports detached weight saving and synchronization."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def bind_replay_owner(self, owner):
        if getattr(self, '_replay_owner', None) is not None:
            raise RuntimeError('Recovery requires fresh isolated FSDP workers; owner already bound')
        self._replay_owner = owner

    def _check_replay_owner(self, owner):
        if getattr(self, '_replay_owner', None) != owner:
            raise RuntimeError('FSDP worker owner mismatch')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def confirm_replay_update(self, version, owner):
        """Confirm actual optimizer/scheduler progress before the replay ACK.

        The pinned FSDP engine returns normally when non-finite gradients skip
        optimizer.step(). Its optimizer watermark then remains at the old
        version, so the existing receipt check rejects the update on every rank.
        """
        self._check_replay_owner(owner)
        return self._replay_checkpoint_receipt(version, version)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def write_replay_checkpoint(self, path, step, version, owner):
        self._check_replay_owner(owner)
        if type(step) is not int or type(version) is not int or step != version:
            raise ValueError('Native checkpoint requires one update per policy version')
        self.actor.save_checkpoint(path, None, version, None)
        from aura.memory.replay.codec import canonical
        from aura.memory.replay.joint import _write_once
        from pathlib import Path
        receipt = self._replay_checkpoint_receipt(step, version)
        _write_once(Path(path) / f'replay_rank_{receipt["rank"]}.json', canonical(receipt))
        return receipt

    def _replay_checkpoint_receipt(self, step, version):
        engine = self.actor.engine
        if engine.is_optimizer_offload_enabled:
            # VERL offloads optimizer state with non_blocking=True. Host reads
            # must wait for the device-to-CPU copy before checking its counters.
            get_torch_device().synchronize()
        steps = {int(value['step']) for value in engine.optimizer.state.values() if 'step' in value}
        if (step != version or steps != ({version} if version else set())
                or engine.lr_scheduler.state_dict()['last_epoch'] != version):
            raise RuntimeError('Native optimizer/scheduler version differs from replay checkpoint')
        manager = engine.checkpoint_manager
        return dict(step=step, policy_version=version, rank=manager.rank, world_size=manager.world_size)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_replay_checkpoint(self, path, step, version, owner):
        self._check_replay_owner(owner)
        self.actor.load_checkpoint(path, None, False)
        return self._replay_checkpoint_receipt(step, version)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_infer_params_to_cpu(self, weight_save_dir: str) -> None:
        """Save FSDP model weights to CPU in safetensors format (rank 0 only).

        Args:
            weight_save_dir: Directory path where weights will be saved.
        """
        logger.info(f"start saving weights, path={weight_save_dir}")
        os.makedirs(weight_save_dir, exist_ok=True)

        start_time = time()
        model_sd = get_model_state_dict(self.actor.engine.module)
        dcp.save(state_dict=model_sd, checkpoint_id=weight_save_dir)

        logger.info(
            f"weight saving completed, "
            f"rank={torch.distributed.get_rank()}, "
            f"path={weight_save_dir}, time={time() - start_time}"
        )

        w_actor = ray.get_actor("weight_updater", namespace="controller_raygroup")
        w_actor.weight_saved.remote(weight_save_dir)
        logger.info("send a weight saving completion notification")
