#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

from __future__ import annotations

import time

from clawcodex_ext.repl.proactive_integration import format_proactive_status
from clawcodex_ext.services.proactive.state import AutomationState


def test_format_proactive_status_hides_inactive() -> None:
    assert format_proactive_status(AutomationState(phase="inactive")) == ""


def test_format_proactive_status_shows_active_countdown() -> None:
    state = AutomationState(
        phase="active",
        next_tick_at=time.time() * 1000 + 5_000,
    )

    text = format_proactive_status(state)

    assert text.startswith("proactive:active ")
    assert text.endswith("s")


def test_format_proactive_status_shows_paused() -> None:
    assert format_proactive_status(AutomationState(phase="paused")) == "proactive:paused"
