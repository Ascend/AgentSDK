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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

from __future__ import annotations

import time

from clawcodex_ext.services.proactive import AutomationState, get_default_controller


def format_proactive_status(state: AutomationState | None = None) -> str:
    state = state or get_default_controller().state
    if state.phase == "inactive":
        return ""
    label = {
        "active": "proactive:active",
        "paused": "proactive:paused",
        "sleeping": "proactive:sleeping",
        "blocked": "proactive:blocked",
    }.get(state.phase, f"proactive:{state.phase}")
    if state.phase == "active" and state.next_tick_at is not None:
        remaining = max(0, int((state.next_tick_at - time.time() * 1000) / 1000))
        return f"{label} {remaining}s"
    if state.phase == "sleeping" and state.last_sleep_until is not None:
        remaining = max(0, int((state.last_sleep_until - time.time() * 1000) / 1000))
        return f"{label} {remaining}s"
    if state.phase == "blocked" and state.blocked_until is not None:
        remaining = max(0, int((state.blocked_until - time.time() * 1000) / 1000))
        return f"{label} {remaining}s"
    return label


__all__ = ["format_proactive_status"]
