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
#

"""Automation-state protocols for the orchestrator and its observers.

* :class:`AutomationStateReporter` — pull model: answers
  ``automation_state()`` with a JSON-serialisable snapshot.
* :class:`AutomationStateObserver` — push model: receives
  ``on_automation_state(snapshot)`` callbacks (e.g. Feishu activity sink).
* :class:`AutomationStateSource` — combined pull + subscribe contract
  (e.g. orchestrator ``status_dashboard.StatusDashboard``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class AutomationStateReporter(Protocol):
    def automation_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable automation state snapshot."""
        ...


@runtime_checkable
class AutomationStateObserver(Protocol):
    """Push-mode consumer; must be cheap and idempotent (may be called rapidly)."""

    def on_automation_state(self, snapshot: dict[str, Any]) -> None:
        """Receive a fresh automation-state snapshot from a source."""
        ...


@runtime_checkable
class AutomationStateSource(Protocol):
    """An automation-state object supporting both pull and subscribe."""

    def automation_state(self) -> dict[str, Any]:
        """Return the current snapshot."""
        ...

    def subscribe(self, observer: AutomationStateObserver) -> None:
        """Register ``observer`` to receive future snapshots."""
        ...


__all__ = [
    "AutomationStateObserver",
    "AutomationStateReporter",
    "AutomationStateSource",
]
