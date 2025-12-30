#!/usr/bin/env python3
# coding=utf-8
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
from __future__ import annotations

import os
from typing import Optional, Dict, Any

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def _ensure_local_env(bind_ip: str) -> None:
    os.environ["RAY_NODE_IP_ADDRESS"] = bind_ip
    os.environ["RAY_API_SERVER_ADDRESS"] = bind_ip
    os.environ["RAY_DASHBOARD_HOST"] = bind_ip
    os.environ["RAY_GRAFANA_HOST"] = bind_ip
    os.environ["RAY_PROMETHEUS_HOST"] = bind_ip
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_LOG_TO_DRIVER"] = "0"
    os.environ["RAY_SCHEDULER_EVENTS"] = "0"
    os.environ["RAY_DISABLE_METRICS"] = "1"
    os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "0"
    os.environ["RAY_INCLUDE_DASHBOARD"] = "0"
    os.environ["RAY_TMPDIR"] = os.path.expanduser("~/.ray/tmp")
    os.environ["TMPDIR"] = os.path.expanduser("~/.ray/tmp")

    # Harden defaults
    os.umask(0o027)


def ray_secure_init(
    address: Optional[str] = None,
    *,
    bind_ip: str = "127.0.0.1",
    extra_init_kwargs: Optional[Dict[str, Any]] = None,
):
    """Initialize Ray securely"""
    # Import ray lazily to avoid side effects before env is set
    import ray  # type: ignore

    kwargs: Dict[str, Any] = dict(extra_init_kwargs or {})

    if address is None:
        # Secure local-only startup
        _ensure_local_env(bind_ip)
        kwargs.update({"_node_ip_address": bind_ip, "logging_level": "ERROR", "include_dashboard": False})
        ctx = ray.init(address="local", **kwargs)
    else:
        # Connecting to explicit address (already validated)
        ctx = ray.init(address=address, **kwargs)
    logger.info(f"Ray init successfully: {ctx}")
    return ctx
