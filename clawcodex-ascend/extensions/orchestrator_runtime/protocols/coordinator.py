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

"""Coordinator context provider protocol.

Multi-agent coordinator mode enters and leaves a coordination context
through ``CoordinatorContextProvider``, so orchestrator need not import
``clawcodex_ext.coordinator.mode.coordinator_mode_context``.
"""

from __future__ import annotations
# pylint: disable=W2301

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable


@runtime_checkable
class CoordinatorContextProvider(Protocol):
    """Bridge over ``clawcodex_ext.coordinator.mode.coordinator_mode_context``.

    Phase 3+ uses this Protocol to decouple orchestrator from clawcodex
    coordinator implementation. Today the ``extensions/orchestrator/`` code
    uses ``coordinator_mode_context`` directly; Phase 3 routes via this
    Protocol.
    """

    def is_active(self) -> bool: ...

    def enter(self, enabled: bool) -> AbstractContextManager[None]:
        """Returns a context manager; entering with ``enabled=True`` flips
        the coordinator-mode gate for the lifetime of the block.

        Mirrors the upstream ``coordinator_mode_context(enabled)`` semantics.
        """
        ...


__all__ = ["CoordinatorContextProvider"]
