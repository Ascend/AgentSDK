#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Bridge SDK stub (NOT the CCR bridge; see ``src/bridge/`` for that work).

The package name collides with ``src/bridge/`` (the CCR bridge
implementation tracked in ``my-docs/ch16-remote-refactoring-plan.md``),
so this stub provides its own minimal ``BridgeSession`` / ``BridgeAuth``
/ ``BridgeTransport`` classes for ``tests/bridge/test_bridge.py`` and
similar unit tests. New code targeting the CCR bridge should import
from ``src.bridge`` directly.
"""

from __future__ import annotations

from clawcodex_ext.services.bridge.auth import BridgeAuth, BridgeToken
from clawcodex_ext.services.bridge.session import (
    BridgeSession,
    BridgeSessionConfig,
    BridgeSessionState,
)
from clawcodex_ext.services.bridge.transport import BridgeTransport, WebSocketTransport

__all__ = [
    "BridgeAuth",
    "BridgeSession",
    "BridgeSessionConfig",
    "BridgeSessionState",
    "BridgeToken",
    "BridgeTransport",
    "WebSocketTransport",
]
