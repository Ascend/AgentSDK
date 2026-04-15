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

import ray
import requests
import torch

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.utils import singleton
from agentic_rl.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
from agentic_rl.controllers.utils.controller_config import ControllerConfig
from agentic_rl.controllers.utils.http_status import HTTP_OK_200
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME, READ_TIMEOUT, DEFAULT_URL_METHOD

logger = Loggers(__name__).get_logger()


def send_ready_to_train_remote(addr):
    logger.info(f"Sending ready to train: {addr} ...")
    queue_actor = get_rollout_queue_actor()
    while not ray.get(queue_actor.is_shutdown.remote()):
        url = f"{DEFAULT_URL_METHOD}://{addr}/train/is_ready"
        try:
            response = requests.post(url)
            if response.status_code == HTTP_OK_200:
                logger.info(f"Send ready to train: {addr} succeed")
                return
            else:
                logger.error(f"server returned: {response.status_code}, {url}, retrying in {DEFAULT_SLEEP_TIME}s...")
        except Exception as e:
            logger.debug(f"send failed: {e}, retrying in {DEFAULT_SLEEP_TIME}s...")
        time.sleep(DEFAULT_SLEEP_TIME)


def send_outputs_to_train_server_remote(addr, outputs, metric, backoff=DEFAULT_SLEEP_TIME):
    logger.info(f"send output to train: {addr} ...")
    queue_actor = get_rollout_queue_actor()
    while not ray.get(queue_actor.is_shutdown.remote()):
        try:
            buf = io.BytesIO()
            torch.save(outputs, buf)
            buf.seek(0)

            files = {"file": ("batch.pt", buf.read(), "application/octet-stream")}
            url = f"{DEFAULT_URL_METHOD}://{addr}/train/send_minibatch"
            logger.info(f"rollout send output to Train server: {url}, start ... ")
            req = requests.post(f"{DEFAULT_URL_METHOD}://{addr}/train/send_minibatch",
                                data=metric, files=files, timeout=READ_TIMEOUT)
            if req.status_code == HTTP_OK_200:
                logger.info(f"rollout send output to Train server, request message: {req.json()}")
                return req.json()
            else:
                logger.warning(f"Server returned {req.status_code}, retrying in {backoff}s...")
        except requests.RequestException as e:
            logger.debug(f"request failed: {e}, retrying in {backoff}s...")
        except Exception as e:
            logger.error(f"request failed: {e}, retrying in {backoff}s...")
        time.sleep(backoff)
    logger.info(f"send output to train: {addr} failed: {ray.get(queue_actor.is_shutdown.remote())}.")


@singleton
class RolloutClient:
    def __init__(self):
        controller_config = ControllerConfig()
        self.train_server_addr = controller_config.train_server_addr

    def send_outputs_to_train_server(self, outputs, metric):
        send_outputs_to_train_server_remote(self.train_server_addr, outputs, metric)

    def send_ready_to_train(self):
        send_ready_to_train_remote(self.train_server_addr)
