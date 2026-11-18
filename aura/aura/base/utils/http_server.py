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
import uvicorn

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

DEFAULT_STOP_TIMEOUT = 5.0


def create_server(server_name, server_app, server_host='0.0.0.0', server_port=8000):
    """Build a uvicorn server without running it."""
    logger.info(f"Start {server_name}: {server_host}:{server_port}")
    config = uvicorn.Config(
        server_app,
        host=server_host,
        port=int(server_port),
        log_level="info",
        reload=False,  # In the development mode, hot reloading can be enabled.
    )
    return uvicorn.Server(config)


def stop_server(server, thread=None, timeout=DEFAULT_STOP_TIMEOUT):
    """Stop a server created by :func:`create_server` and release its port."""
    if server is not None:
        server.should_exit = True
    if thread is None:
        return True
    thread.join(timeout)
    return not thread.is_alive()


def start_server(server_name, server_app, server_host='0.0.0.0', server_port=8000):
    logger.info(f"Start {server_name}: {server_host}:{server_port}")

    # Configure the running parameters of Uvicorn
    uvicorn.run(
        server_app,
        host=server_host,
        port=int(server_port),
        log_level="info",
        reload=False  # In the development mode, hot reloading can be enabled.
    )
