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

"""Textual widgets used by the multi-model display bridge."""

from .diff_panel import MultiModelDiffPanel
from .progress_bar import ModelProgressBars
from .result_card import ModelResultCard
from .selection_list import MultiModelSelectionList
from .summary_panel import MultiModelSummaryPanel
from .tab_bar import ModelTabBar
from .tab_panel import ModelTabPanel
from .live_panel import MultiModelLivePanel

__all__ = [
    "ModelProgressBars",
    "ModelResultCard",
    "ModelTabBar",
    "ModelTabPanel",
    "MultiModelDiffPanel",
    "MultiModelSelectionList",
    "MultiModelSummaryPanel",
    "MultiModelLivePanel",
]
