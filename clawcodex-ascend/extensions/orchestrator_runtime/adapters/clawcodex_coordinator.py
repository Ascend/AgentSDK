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

"""ClawcodexCoordinatorProvider — concrete ``CoordinatorContextProvider`` adapter.

Thin wrapper over ``coordinator_mode_context`` / ``is_coordinator_mode``
so agent_runner does not import ``clawcodex_ext.coordinator.mode``.

``enter()`` returns the upstream context manager. ``is_active()``
forwards ``is_coordinator_mode()``. No local state — each call hits
upstream, matching the original behavior.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from extensions.orchestrator_runtime.protocols.coordinator import (
    CoordinatorContextProvider,
)

if TYPE_CHECKING:
    pass


class ClawcodexCoordinatorProvider(CoordinatorContextProvider):
    """Forward to ``clawcodex_ext.coordinator.mode``."""

    def is_active(self) -> bool:
        from clawcodex_ext.coordinator.mode import is_coordinator_mode

        return bool(is_coordinator_mode())

    def enter(self, enabled: bool = True) -> AbstractContextManager[None]:
        from clawcodex_ext.coordinator.mode import coordinator_mode_context

        # Pass through the ``enabled`` flag so callers (e.g. agent_runner.run)
        # can keep their dynamic coordinator-mode toggle semantics.
        return coordinator_mode_context(enabled)  # type: ignore[return-value]


__all__ = ["ClawcodexCoordinatorProvider"]
