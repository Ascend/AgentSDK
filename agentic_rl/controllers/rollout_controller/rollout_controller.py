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
import os
import threading
import time

import ray
from fastapi import FastAPI

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX
from agentic_rl.base.utils.http_server import start_server
from agentic_rl.controllers.rollout_controller.rollout_client import RolloutClient
from agentic_rl.controllers.rollout_controller.rollout_queue import RolloutQueueActor
from agentic_rl.controllers.rollout_controller.rollout_server import RolloutServer
from agentic_rl.controllers.rollout_controller.rollout_weight_manager import RolloutWeightManager
from agentic_rl.controllers.utils.controller_config import ControllerConfig
from agentic_rl.controllers.utils.utils import create_actor, MAX_CONCURRENCY, DEFAULT_CPUS

logger = Loggers(__name__).get_logger()


def clean_rollout_weights(weight_save_dir):
    inference_save_path = weight_save_dir + ROLLOUT_WEIGHTS_PREFIX
    if os.path.exists(inference_save_path) and os.path.isdir(inference_save_path):
        for root, dirs, files in os.walk(inference_save_path):
            for file in files:
                os.remove(os.path.join(root, file))


class RolloutController:
    def __init__(self, **kwargs):
        self.rollout_server = None
        self.rollout_queue_actor = None
        self.rollout_weight_manager = None

        controller_config = ControllerConfig()
        self.rollout_client = None
        self.rollout_server_addr = controller_config.rollout_server_addr

        self.rollout_client = RolloutClient()
        self.initialize_rollout_weight_manager(**kwargs)
        self.initialize_rollout_queue_actor()
        self.initialize_rollout_server()

    def send_ready_to_train(self):
        self.rollout_client.send_ready_to_train()

    def get_weight_manager(self):
        return self.rollout_weight_manager

    def initialize_rollout_queue_actor(self):
        self.rollout_queue_actor = create_actor(
            name="rollout_queue",
            cls=RolloutQueueActor,
            namespace="controller_raygroup",
            options={"num_cpus": DEFAULT_CPUS, "max_concurrency": MAX_CONCURRENCY},
        )
        ray.get(self.rollout_queue_actor.init_done.remote())
        logger.info(f">>> rollout queue actor create succeed")

    def initialize_rollout_server(self):
        parts = self.rollout_server_addr.split(":")

        self.rollout_server = RolloutServer(
            rollout_queue=self.rollout_queue_actor,
            rollout_weight_manager=self.rollout_weight_manager
        )

        app = FastAPI()
        app.include_router(self.rollout_server.router)

        # Create a new thread to run the server
        threading.Thread(target=start_server, args=('Rollout Server', app, parts[0], parts[1])).start()
        logger.info(f">>> rollout server start succeed")

    def initialize_rollout_weight_manager(self, **kwargs):
        self.rollout_weight_manager = create_actor(
            name="rollout_weight_manager",
            cls=RolloutWeightManager,
            namespace="controller_raygroup",
            options={"num_cpus": DEFAULT_CPUS, "max_concurrency": MAX_CONCURRENCY},
            actor_kwargs=kwargs
        )
        ray.get(self.rollout_weight_manager.init_done.remote())
        logger.info(">>> initialized rollout weight manager succeed")
        clean_rollout_weights(kwargs["weight_save_dir"])

    def running(self):
        return self.rollout_server.running()

    def finish_rollout(self):
        while not self.rollout_server.is_shutdown:
            time.sleep(3)
            logger.info(f"Rollout wait for shutdown ...")
        # stop rollout server
        logger.info(f"stop rollout server succeed")
        # stop dispatch actor
        ray.kill(self.rollout_queue_actor)
        logger.info("stop rollout queue actor succeed")
        # stop rollout_weight_manager
        ray.kill(self.rollout_weight_manager)
        logger.info("stop rollout rollout weight manager succeed")
