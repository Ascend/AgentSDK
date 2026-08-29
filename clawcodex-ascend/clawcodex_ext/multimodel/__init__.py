#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Multi-model scheduling extension.

The router implements the existing provider interface, so it can be injected
without changing core query-loop packages.
"""

from .aggregators import (
    FirstSuccessAggregator,
    FusionAggregator,
    MajorityVoteAggregator,
    PassThroughAggregator,
    RankAggregator,
    ScoringAggregator,
)
from .config import GroupConfig, MultiModelConfig, SlotConfig
from .display import MultiModelBridge
from .factory import build_router
from .router import MultiModelRouter, RouterConfig
from .session_bridge import SessionBridge
from .slots import ProviderSlot
from .strategies import FallbackStrategy, ParallelStrategy, RoutingRule, RoutingStrategy, VotingStrategy

__all__ = [
    "FirstSuccessAggregator",
    "FusionAggregator",
    "MajorityVoteAggregator",
    "PassThroughAggregator",
    "RankAggregator",
    "ScoringAggregator",
    "GroupConfig",
    "MultiModelConfig",
    "SlotConfig",
    "MultiModelBridge",
    "build_router",
    "MultiModelRouter",
    "ParallelStrategy",
    "ProviderSlot",
    "RouterConfig",
    "RoutingRule",
    "RoutingStrategy",
    "FallbackStrategy",
    "SessionBridge",
    "VotingStrategy",
]
