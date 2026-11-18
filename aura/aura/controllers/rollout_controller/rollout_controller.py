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

from aura.base.log.loggers import Loggers
from aura.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX
from aura.base.utils.http_server import create_server, start_server, stop_server
from aura.controllers.rollout_controller.rollout_client import RolloutClient
from aura.controllers.rollout_controller.rollout_queue import RolloutQueueActor
from aura.controllers.rollout_controller.rollout_server import RolloutServer
from aura.controllers.rollout_controller.rollout_weight_manager import RolloutWeightManager
from aura.controllers.utils.controller_config import ControllerConfig
from aura.controllers.utils.utils import (
    create_actor,
    kill_actor,
    MAX_CONCURRENCY,
    DEFAULT_CPUS,
    DEFAULT_SLEEP_TIME,
    TRAIN_CONTROLLER_NAMESPACE,
)
from aura.trainer.rollout.resource_ledger import DEFAULT_RELEASE_TIMEOUT

logger = Loggers(__name__).get_logger()

SHUTDOWN_LOG_INTERVAL = 30


def clean_rollout_weights(weight_save_dir):
    inference_save_path = weight_save_dir + ROLLOUT_WEIGHTS_PREFIX
    if os.path.exists(inference_save_path) and os.path.isdir(inference_save_path):
        for root, dirs, files in os.walk(inference_save_path):
            for file in files:
                os.remove(os.path.join(root, file))


class RolloutController:
    def __init__(self, enabled=False, **kwargs):
        self.enabled = enabled
        self.rollout_server = None
        self.rollout_queue_actor = None
        self.rollout_weight_manager = None
        self._http_server = None
        self._http_thread = None

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
            namespace=TRAIN_CONTROLLER_NAMESPACE,
            options={"num_cpus": DEFAULT_CPUS, "max_concurrency": MAX_CONCURRENCY},
        )
        ray.get(self.rollout_queue_actor.init_done.remote())
        logger.info(">>> rollout queue actor create succeed")

    def initialize_rollout_server(self):
        parts = self.rollout_server_addr.split(":")

        self.rollout_server = RolloutServer(
            rollout_queue=self.rollout_queue_actor, rollout_weight_manager=self.rollout_weight_manager
        )

        app = FastAPI()
        app.include_router(self.rollout_server.router)

        if not self.enabled:
            threading.Thread(target=start_server, args=('Rollout Server', app, parts[0], parts[1])).start()
            logger.info(">>> rollout server start succeed")
            return

        self._http_server = create_server('Rollout Server', app, parts[0], parts[1])
        self._http_thread = threading.Thread(target=self._http_server.run)
        self._http_thread.start()
        logger.info(">>> rollout server start succeed")

    def initialize_rollout_weight_manager(self, **kwargs):
        self.rollout_weight_manager = create_actor(
            name="rollout_weight_manager",
            cls=RolloutWeightManager,
            namespace=TRAIN_CONTROLLER_NAMESPACE,
            options={"num_cpus": DEFAULT_CPUS, "max_concurrency": MAX_CONCURRENCY},
            actor_kwargs=kwargs,
        )
        ray.get(self.rollout_weight_manager.init_done.remote())
        logger.info(">>> initialized rollout weight manager succeed")
        clean_rollout_weights(kwargs["weight_save_dir"])

    def running(self):
        return self.rollout_server.running()

    def finish_rollout(self):
        if not self.enabled:
            while not self.rollout_server.is_shutdown:
                time.sleep(DEFAULT_SLEEP_TIME)
                logger.info("Rollout wait for shutdown ...")
            logger.info("shutdown rollout server succeed")
            kill_actor(self.rollout_queue_actor)
            logger.info("stop rollout queue actor succeed")
            kill_actor(self.rollout_weight_manager)
            logger.info("stop rollout rollout weight manager succeed")
            self.rollout_server.already_quit = True
            return

        started = time.time()
        last_log = started
        while not self.rollout_server.is_shutdown:
            time.sleep(DEFAULT_SLEEP_TIME)
            now = time.time()
            if now - last_log >= SHUTDOWN_LOG_INTERVAL:
                logger.info(f"Rollout wait for shutdown ... elapsed={now - started:.1f}s")
                last_log = now
        # Reuse the idempotent close* methods so that the aggregate subprocess
        # is reaped before its host actor is killed, and the rollout HTTP port
        # is released while this process is still alive.
        self.close_weight_manager()
        self.close_queue()
        self.close_server()

    def close_weight_manager(self):
        """Stop the weight manager: terminate its aggregate subprocess, then kill it."""
        actor = self.rollout_weight_manager
        if actor is None:
            return
        self.rollout_weight_manager = None
        try:
            ray.get(actor.close.remote(), timeout=DEFAULT_RELEASE_TIMEOUT)
        except BaseException as exp:
            logger.warning(f"rollout weight manager close failed: {type(exp).__name__}")
        kill_actor(actor)
        logger.info("stop rollout weight manager succeed")

    def close_queue(self):
        """Kill the named rollout queue actor. Idempotent."""
        actor = self.rollout_queue_actor
        if actor is None:
            return
        self.rollout_queue_actor = None
        kill_actor(actor)
        logger.info("stop rollout queue actor succeed")

    @classmethod
    def release_named_actors(cls):
        """Kill the controller named actors by name; callable from any process."""
        released = []
        for name in ("rollout_weight_manager", "rollout_queue"):
            try:
                actor = ray.get_actor(name, namespace=TRAIN_CONTROLLER_NAMESPACE)
            except Exception:
                continue
            try:
                if name == "rollout_weight_manager":
                    ray.get(actor.close.remote(), timeout=DEFAULT_RELEASE_TIMEOUT)
            except BaseException as exp:
                logger.warning(f"{name} close failed: {type(exp).__name__}")
            kill_actor(actor)
            released.append(name)
        return released

    def close_server(self):
        """Stop the rollout HTTP server and release its port. Idempotent."""
        server, thread = self._http_server, self._http_thread
        self._http_server, self._http_thread = None, None
        if self.rollout_server is not None:
            self.rollout_server.already_quit = True
        if server is None:
            return
        if not stop_server(server, thread, DEFAULT_RELEASE_TIMEOUT):
            logger.warning("rollout server thread did not exit within timeout")
        else:
            logger.info("stop rollout server succeed")
