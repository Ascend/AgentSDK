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

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


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
