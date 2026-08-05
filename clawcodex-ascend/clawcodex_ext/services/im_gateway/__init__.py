#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""IM Message Gateway service.

Unified IM entry point: inbound dispatch, session routing, capability
gated outbound delivery, and file-based reliability. The Gateway daemon
process (``extensions/im_gateway/``) hosts this service and exposes it
to REPL/orchestrator opt-in clients over POSIX UDS.

v1 (P0–P5) ships: capability contract + registry, gateway skeleton,
WeChat iLink text closed-loop, Orchestrator event push, reliability
hardening, and six-class message semantics.
"""

from __future__ import annotations

from .binding import BindingEntry, BindingPolicy
from .capability_gate import CapabilityGate
from .config import (
    CommandAllowlistConfig,
    GatewayConfig,
    ReliabilityConfig,
    load_config,
    save_config,
)
from .dispatcher import InboundDispatcher
from .gateway import MessageGateway
from .ipc_protocol import GatewayFrame, FrameType, PROTOCOL_VERSION
from .models import (
    AckLayer,
    AckReceipt,
    CircuitState,
    InboundMessage,
    MessageSemantics,
    OriginKey,
    OutboundMessage,
    SessionTarget,
)
from .outbound import OutboundDispatcher
from .processing_status import ProcessingStatusManager
from .reliability import ReliabilityStore
from .router import SessionRouter
from .text import strip_markdown, split_text, maybe_truncate_with_liveview

__all__ = [
    "AckLayer",
    "AckReceipt",
    "BindingEntry",
    "BindingPolicy",
    "CircuitState",
    "CapabilityGate",
    "CommandAllowlistConfig",
    "FrameType",
    "GatewayConfig",
    "GatewayFrame",
    "InboundDispatcher",
    "InboundMessage",
    "MessageGateway",
    "MessageSemantics",
    "OriginKey",
    "OutboundDispatcher",
    "OutboundMessage",
    "PROTOCOL_VERSION",
    "ProcessingStatusManager",
    "ReliabilityConfig",
    "ReliabilityStore",
    "SessionRouter",
    "SessionTarget",
    "load_config",
    "maybe_truncate_with_liveview",
    "save_config",
    "split_text",
    "strip_markdown",
]
