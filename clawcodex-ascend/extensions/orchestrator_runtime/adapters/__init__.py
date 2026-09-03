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

"""Adapters that wrap clawcodex implementations behind runtime protocols.

``clawcodex_compat`` re-exports selected ``clawcodex_ext.*`` symbols so
``extensions.orchestrator`` can switch import paths without changing
call-site behavior.

Concrete adapters (injectable via kwargs on ``AgentRunner`` /
``OrchestratorGatewayClient``):

* :class:`ClawcodexAgentRuntime` — wraps ``extensions.api.query.QueryRunner``
* :class:`ClawcodexSessionStorage` — wraps ``clawcodex_ext.services.session_storage``
* :class:`ClawcodexCoordinatorProvider` — wraps ``clawcodex_ext.coordinator.mode``
* :class:`ClawcodexImChannel` — wraps ``OrchestratorGatewayClient``
* :class:`ClawcodexBootstrapState` — wraps ``clawcodex_ext.bootstrap.state``

``build_default_*()`` returns a cached singleton. Production code can
call ``build_default_agent_runtime()``; tests replace it with a stub.
"""

from __future__ import annotations

from extensions.orchestrator_runtime.adapters.clawcodex_agent_runtime import (
    ClawcodexAgentRuntime,
)
from extensions.orchestrator_runtime.adapters.clawcodex_bootstrap_state import (
    ClawcodexBootstrapState,
)
from extensions.orchestrator_runtime.adapters.clawcodex_coordinator import (
    ClawcodexCoordinatorProvider,
)
from extensions.orchestrator_runtime.adapters.clawcodex_im_channel import (
    ClawcodexImChannel,
)
from extensions.orchestrator_runtime.adapters.clawcodex_session_storage import (
    ClawcodexSessionStorage,
)

# ─── Singletons (lazy via module-level globals; tests can ``del`` to reset) ──

_agent_runtime_singleton: ClawcodexAgentRuntime | None = None
_session_storage_singleton: ClawcodexSessionStorage | None = None
_coordinator_singleton: ClawcodexCoordinatorProvider | None = None
_bootstrap_state_singleton: ClawcodexBootstrapState | None = None


def build_default_agent_runtime() -> ClawcodexAgentRuntime:
    """Return a cached default ``AgentRuntime`` adapter."""
    global _agent_runtime_singleton
    if _agent_runtime_singleton is None:
        _agent_runtime_singleton = ClawcodexAgentRuntime()
    return _agent_runtime_singleton


def build_default_session_storage() -> ClawcodexSessionStorage:
    """Return a cached default ``SessionStorage`` adapter."""
    global _session_storage_singleton
    if _session_storage_singleton is None:
        _session_storage_singleton = ClawcodexSessionStorage()
    return _session_storage_singleton


def build_default_coordinator_provider() -> ClawcodexCoordinatorProvider:
    """Return a cached default ``CoordinatorContextProvider`` adapter."""
    global _coordinator_singleton
    if _coordinator_singleton is None:
        _coordinator_singleton = ClawcodexCoordinatorProvider()
    return _coordinator_singleton


def build_default_bootstrap_state() -> ClawcodexBootstrapState:
    """Return a cached default ``BootstrapState`` adapter."""
    global _bootstrap_state_singleton
    if _bootstrap_state_singleton is None:
        _bootstrap_state_singleton = ClawcodexBootstrapState()
    return _bootstrap_state_singleton


def reset_adapters_for_tests() -> None:
    """Drop all singletons so the next ``build_default_*()`` recreates them.

    Tests that want to swap a default implementation should call this after
    patching the underlying module attribute.
    """
    global _agent_runtime_singleton
    global _session_storage_singleton
    global _coordinator_singleton
    global _bootstrap_state_singleton
    _agent_runtime_singleton = None
    _session_storage_singleton = None
    _coordinator_singleton = None
    _bootstrap_state_singleton = None


__all__ = [
    "ClawcodexAgentRuntime",
    "ClawcodexBootstrapState",
    "ClawcodexCoordinatorProvider",
    "ClawcodexImChannel",
    "ClawcodexSessionStorage",
    "build_default_agent_runtime",
    "build_default_bootstrap_state",
    "build_default_coordinator_provider",
    "build_default_session_storage",
    "reset_adapters_for_tests",
]
