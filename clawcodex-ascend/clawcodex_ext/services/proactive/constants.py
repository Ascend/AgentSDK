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

from __future__ import annotations

TICK_INTERVAL_MS: int = 30_000
TICK_TAG: str = "tick"
CONTEXT_BLOCKED_TTL_SEC: int = 60
DEFAULT_JITTER_FRACTION: float = 0.05

FOCUS_LEVELS: tuple[str, ...] = ("full", "medium", "minimal")
DEFAULT_FOCUS_LEVEL: str = "medium"
MAX_LAST_TICK_SUMMARY_CHARS: int = 800
