#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Background-session lifecycle support."""

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
