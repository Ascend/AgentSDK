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

"""Kairos / Brief scheduling service-layer exceptions."""


class KairosError(RuntimeError):
    """Base error for kairos operations."""


class TickConfigError(KairosError):
    """Raised when a :class:`TickConfig` fails validation."""


class SchedulerStateError(KairosError):
    """Raised when a scheduler operation is invalid for the current state.

    Examples: starting an already-running scheduler, stopping one that
    was never started, or registering a callback after shutdown.
    """


class DailyLogError(KairosError):
    """Raised when the daily log writer cannot append or read an entry."""


class BriefGenerationError(KairosError):
    """Raised when a brief cannot be produced from the supplied snapshot."""
