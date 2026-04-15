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
import json
import os
import shutil
import threading
import time
from pathlib import Path

import ray
import requests
import torch
from fastapi import FastAPI

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.http_server import start_server
from agentic_rl.controllers.train_controller.dispatch_actor import DispatchActor
from agentic_rl.controllers.train_controller.train_server import TrainServer
from agentic_rl.controllers.train_controller.train_weight_updater import WeightUpdateActor
from agentic_rl.controllers.utils.controller_config import ControllerConfig
from agentic_rl.controllers.utils.http_status import HTTP_OK_200
from agentic_rl.controllers.utils.utils import (
    create_actor, DEFAULT_SLEEP_TIME,
    MAX_CPUS, MAX_TIMEOUT, DEFAULT_URL_METHOD)

logger = Loggers(__name__).get_logger()


class TrainController:
    def __init__(
            self,
            global_batch_size,
            n_samples_per_prompt,
            validate_num_samples,
            init_num_group_batches,
            max_queue_size,
            train_iters,
            weight_save_dir,
            delta,
            data_loader,
            actor_worker,
            initialize_rollout_dataloader,
            consumed_train_samples,
            data_optimized
    ) -> None:
        self.dispatch_actor = None
        self.train_server = None
        self.weight_update_actor = None

        self.actor_worker = actor_worker
        self.data_loader = data_loader
        self.timing_training_unit = []
        self.initialization_timeout = MAX_TIMEOUT

        self.global_batch_size = global_batch_size
        self.n_samples_per_prompt = n_samples_per_prompt
        self.validate_num_samples = validate_num_samples
        self.init_num_group_batches = init_num_group_batches

        self.max_queue_size = max_queue_size
        self.train_iters = train_iters
        self.weight_save_dir = weight_save_dir
        self.delta = delta
        self.initialize_rollout_dataloader = initialize_rollout_dataloader
        self.consumed_train_samples = consumed_train_samples
        self.data_optimized = data_optimized

        controller_config = ControllerConfig()
        self.train_server_addr = controller_config.train_server_addr
        self.rollout_server_addr = controller_config.rollout_server_addr

    def pre_initialize(self):
        self.initialize_dispatch()
        self.initialize_train_server()
        self.initialize_weight_updater()

    def initialize_rollout(self):
        self.send_initial_batch_groups_to_rollout()
        self.unlock_rollout_unit()
        self.clean_train_updated_weights()

    def initialize_dispatch(self):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        self.dispatch_actor = create_actor(
            name="dispatch",
            cls=DispatchActor,
            namespace="controller_raygroup",
            options={
                "num_cpus": MAX_CPUS, 'scheduling_strategy': NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().node_id,
                    soft=False)  # Compulsory strong affinity
            },
            actor_args=(self.n_samples_per_prompt,
                        self.validate_num_samples,
                        self.global_batch_size,
                        self.train_iters,
                        self.data_loader,
                        self.initialize_rollout_dataloader,
                        self.consumed_train_samples,
                        self.data_optimized,
                        self.rollout_server_addr),
        )
        ray.get(self.dispatch_actor.init_done.remote())
        logger.info(f">>> dispatch actor create success")

    def initialize_train_server(self):
        logger.info(">>> initializing train server")
        parts = self.train_server_addr.split(":")

        self.train_server = TrainServer(
            max_queue_size=self.max_queue_size,
            global_batch_size=self.global_batch_size,
            n_samples_per_prompt=self.n_samples_per_prompt
        )

        app = FastAPI()
        app.include_router(self.train_server.router)

        # Create a new thread to run the server
        threading.Thread(target=start_server, args=('Train Server', app, parts[0], parts[1])).start()
        logger.info(f">>> train server start success")

    def initialize_weight_updater(self):
        self.weight_update_actor = create_actor(
            name="weight_updater",
            cls=WeightUpdateActor,
            namespace="controller_raygroup",
            options={"num_cpus": MAX_CPUS},
            actor_args=(self.dispatch_actor,
                        self.actor_worker),
        )
        ray.get(self.weight_update_actor.init_done.remote())
        logger.info(f">>> weight update actor create success")

    def send_initial_batch_groups_to_rollout(self):
        self.dispatch_actor.send_batch_groups.remote(self.init_num_group_batches)

    # If you need single unit controls from the driver:
    def unlock_rollout_unit(self):
        self.dispatch_actor.unlock_rollout_unit.remote()

    def clean_train_updated_weights(self) -> None:
        # Check if the directory exists
        if os.path.exists(self.weight_save_dir) and os.path.isdir(self.weight_save_dir):
            # Iterate through the two subfolders inside 'rollout'
            for root, dirs, files in os.walk(self.weight_save_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    def wait_for_rollout_unit_ready(self):
        init_time = time.time()
        logger.info("wait for all rollout units ready ...")
        while not ray.get(self.dispatch_actor.check_rollout_unit_ready.remote()):
            time.sleep(DEFAULT_SLEEP_TIME)
            if time.time() - init_time > self.initialization_timeout:
                raise TimeoutError("rollout unit did not signal its availability within timeout.")
        logger.info("all rollout units are ready!")

    def _training_batch_queue_ready(self):
        url = f"{DEFAULT_URL_METHOD}://{self.train_server_addr}/train/is_batch_ready"
        response = requests.post(url)
        if response.status_code != HTTP_OK_200:
            return False
        data = response.json()
        return data.get("is_ready", False)

    def get_next_training_batch(self, last_iteration: bool = False):
        logger.info(f"waiting next training batch ...")
        while not self._training_batch_queue_ready():
            time.sleep(DEFAULT_SLEEP_TIME)

        logger.info(">>> start to get a mini batch ....")
        start_time = time.time()
        url = f"{DEFAULT_URL_METHOD}://{self.train_server_addr}/train/get_minibatch"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            metric = {}
            metadata_json = response.headers.get('X-Metrics-Metadata')
            if metadata_json:
                metric = json.loads(metadata_json)

            file_data_bytes = response.content
            file_buffer = io.BytesIO(file_data_bytes)
            file_buffer.seek(0)
            outputs = torch.load(file_buffer, map_location="cpu")
        except Exception as e:
            logger.warning(f">>> get mini batch failed: {e}")
            return None, None
        logger.warning(f">>> get mini batch succeed, cost {time.time() - start_time:.2f} s")
        # A batch of data has been retrieved. The queue is idle. Unconditionally unlock the rollout unit.
        if not last_iteration:
            self.unlock_rollout_unit()
        return outputs, metric

    def finish_training_iteration(self, iteration: int):
        self.timing_training_unit.append(time.time())
        self.dispatch_actor.set_current_training_iter.remote(iteration + 1)

    def finish_training(self):
        # Shut down one's own server
        logger.info(f"stop train server succeed")
        ray.get(self.dispatch_actor.shutdown.remote())
        # stop dispatch actor
        ray.kill(self.dispatch_actor)
        logger.info("stop dispatch actor succeed")
        # weight update actor
        ray.kill(self.weight_update_actor)
        logger.info("stop weight update actor succeed")

    def update_rollout_weights(self, iteration: int):
        logger.info(f">>> start to export weights, iteration={iteration}...")
        start_time = time.time()
        weights_path = self._create_weight_dir(iteration)
        self.weight_update_actor.update_weights_to_file.remote(weight_save_dir=str(weights_path), iteration=iteration)
        while not ray.get(self.weight_update_actor.update_weights_finished.remote()):
            time.sleep(DEFAULT_SLEEP_TIME)
        logger.info(f">>> finish export weights, iteration={iteration}, cost: {time.time() - start_time}")

    def _get_old_iter_dirs(self):
        ckpt_dir = Path(self.weight_save_dir)
        # checkpoint filenames look like <ckpt_dir>/iter_<N>/
        all_iters = [iter_path for iter_path in ckpt_dir.glob("iter_*") if iter_path.is_dir()]
        # Sort by modification time. The directories created earlier should be deleted first.
        all_iters = sorted(all_iters, key=lambda iter_path: iter_path.stat().st_mtime)
        return ckpt_dir, all_iters

    def _clean_old_iters_than_delta(self, ckpt_dir, all_iters):
        # Remove the iters older than delta
        if len(all_iters) > self.delta:
            for iter_idx in range(len(all_iters) - self.delta):
                old_path = all_iters[iter_idx]
                logger.info(f"remove the old weight dir: {old_path}")
                shutil.rmtree(old_path)

    def _create_weight_dir(self, iteration: int) -> Path:
        ckpt_dir, all_iters = self._get_old_iter_dirs()
        self._clean_old_iters_than_delta(ckpt_dir, all_iters)
        return ckpt_dir / f"iter_{str(iteration).zfill(7)}"
