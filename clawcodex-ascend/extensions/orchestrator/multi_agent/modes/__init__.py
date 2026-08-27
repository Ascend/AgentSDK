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
"""Collaboration mode registry for the orchestrator.

This package exposes the ``ModeRunner`` Protocol plus a small registry so
new modes (Pipeline, Coordinator, Debate) can be plugged in without
touching the orchestrator core. Phase-1 ships only the ``single`` mode,
which is a transparent wrapper over the existing ``AgentRunner.run``
loop — behavior is byte-identical to the pre-mode code path.

Public surface
--------------

* ``ModeRunner``    — Protocol every mode implementation must satisfy
* ``ModeDecision``  — dataclass returned by ``ModeSelector.choose``
* ``register``      — register a ``ModeRunner`` under a string key
* ``get``           — fetch a registered runner by key (or raise KeyError)
* ``available``     — list registered mode keys
* ``DEFAULT_MODE``  — fallback when ModeSelector fails or returns unknown
"""

from __future__ import annotations

import logging

from .base import DEFAULT_MODE, ModeDecision, ModeRunner

logger = logging.getLogger(__name__)

_registry: dict[str, ModeRunner] = {}


def register(key: str, runner: ModeRunner) -> None:
    """Register a ``ModeRunner`` under ``key``.

    Re-registering the same key overwrites the prior entry — this is
    intentional so tests / plugins can swap implementations cleanly.
    """
    if key in _registry:
        logger.warning("Replacing already registered collaboration mode %r", key)
    _registry[key] = runner


def get(key: str) -> ModeRunner:
    """Fetch the runner for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not registered. Callers that want a graceful
        fallback should catch this and use ``DEFAULT_MODE``.
    """
    return _registry[key]


def available() -> list[str]:
    """Return the list of currently registered mode keys."""
    return sorted(_registry.keys())


__all__ = [
    "DEFAULT_MODE",
    "ModeDecision",
    "ModeRunner",
    "register",
    "get",
    "available",
]
