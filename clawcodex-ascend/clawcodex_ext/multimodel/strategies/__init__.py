#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Scheduling strategies used by :class:`MultiModelRouter`."""

from .base import MultiModelStrategyBase
from .fallback import FallbackStrategy
from .parallel import ParallelStrategy
from .routing import RoutingRule, RoutingStrategy
from .voting import VotingStrategy

__all__ = [
    "FallbackStrategy",
    "MultiModelStrategyBase",
    "ParallelStrategy",
    "RoutingRule",
    "RoutingStrategy",
    "VotingStrategy",
]
