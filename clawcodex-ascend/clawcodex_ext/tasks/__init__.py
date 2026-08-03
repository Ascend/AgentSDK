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

"""F-94 BG_SESSIONS — 后台会话统一管理。

整合现有 ``background_runner`` / ``RuntimeTaskRegistry`` / ``ResumeAgent``
零散能力，提供全局 index、跨进程 discover、统一状态机、orphan cleanup、
``BgSessionTool`` 与 ``/bg`` 命令族。

设计原则（CLAUDE.md 黄金法则 1/6）：本包为 Layer 1 补丁层，**不修改**
``src/agent/background_runner.py`` 的 fork/subprocess 路径；仅在 marker
写入后追加 index upsert（见 ``bg_session_manager.upsert_after_launch``）。
``bg_sessions=off`` 时退化为现有 marker 行为（验收标准 1）。
"""

from __future__ import annotations

from .bg_session import (
    BgSession,
    BgSessionAlreadyRunningError,
    BgSessionAttachError,
    BgSessionConfig,
    BgSessionEvent,
    BgSessionNotFoundError,
    BgSessionOrphanedError,
    BgSessionPermissionError,
    BgSessionStatus,
    BgSessionStopError,
    BgSessionsDisabledError,
    DEFAULT_INDEX_PATH,
    is_bg_sessions_enabled,
    marker_path_for,
    replace_session,
    transcript_path_for,
)
from .bg_session_health import HealthAssessment, assess, reconcile
from .bg_session_registry import BgSessionRegistry
from .bg_session_manager import BgSessionManager

__all__ = [
    "BgSession",
    "BgSessionConfig",
    "BgSessionEvent",
    "BgSessionStatus",
    "BgSessionsDisabledError",
    "BgSessionNotFoundError",
    "BgSessionAlreadyRunningError",
    "BgSessionAttachError",
    "BgSessionPermissionError",
    "BgSessionOrphanedError",
    "BgSessionStopError",
    "DEFAULT_INDEX_PATH",
    "HealthAssessment",
    "assess",
    "is_bg_sessions_enabled",
    "marker_path_for",
    "reconcile",
    "replace_session",
    "transcript_path_for",
    "BgSessionRegistry",
    "BgSessionManager",
]
