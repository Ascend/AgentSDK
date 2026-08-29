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

"""Monitor service layer.

Provides the controller, watch compatibility shim, generic text tail follower,
and stall-watchdog exemption hook used by both the ``/monitor`` slash command
and the ``Monitor`` built-in tool.
"""

from __future__ import annotations

from .controller import MonitorController, MonitorStartResult
from .install import install_monitor_extensions
from .stall_guard import StallWatchdogExemptor
from .text_tail import TextTailBuffer, TextTailFollower
from .watch_compat import normalize_watch_command

__all__ = [
    "MonitorController",
    "MonitorStartResult",
    "StallWatchdogExemptor",
    "TextTailBuffer",
    "TextTailFollower",
    "install_monitor_extensions",
    "normalize_watch_command",
]
