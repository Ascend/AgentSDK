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

"""Six-class inbound message semantics (P5).

Classifier → CommandRouter → ControlBridge → InboundRuntimeRouter.
No natural-language auto-judgment for ``interrupt``/``contextOnly``;
both require structured metadata or existing control/bridge entry points.
"""

from __future__ import annotations

from .classifier import MessageClassifier
from .command_router import CommandRoute, CommandRouter
from .control_bridge import ControlBridge, ControlTarget
from .runtime_router import InboundRuntimeRouter, RoutingDecision

__all__ = [
    "CommandRoute",
    "CommandRouter",
    "ControlBridge",
    "ControlTarget",
    "InboundRuntimeRouter",
    "MessageClassifier",
    "RoutingDecision",
]
