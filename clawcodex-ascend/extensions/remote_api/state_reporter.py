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

from __future__ import annotations

# Pylint cannot infer dynamically exposed clawcodex_ext subpackages.
# pylint: disable=no-name-in-module

from typing import Any

from clawcodex_ext.services.proactive import get_default_controller


class ProactiveAutomationStateReporter:
    def automation_state(self) -> dict[str, Any]:
        return get_default_controller().state.to_dict()


def current_automation_state() -> dict[str, Any]:
    return ProactiveAutomationStateReporter().automation_state()


def set_proactive_focus(level: str) -> dict[str, Any]:
    ctrl = get_default_controller()
    ctrl.set_focus(level)  # type: ignore[arg-type]
    return ctrl.state.to_dict()


__all__ = [
    "ProactiveAutomationStateReporter",
    "current_automation_state",
    "set_proactive_focus",
]
