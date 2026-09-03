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

"""Orchestrator-local bootstrap-state protocol.

``AgentRunner._save_json_snapshot`` needs the eight cost/timing/line
getters from ``clawcodex_ext.bootstrap.state`` to rebuild a resume
snapshot's cost_block. This protocol exposes them as
``self._bootstrap_state.get_total_cost_usd()`` instead of importing
clawcodex_ext.

Must not import ``clawcodex_ext.*``. Default implementation:
:mod:`extensions.orchestrator_runtime.adapters.clawcodex_bootstrap_state`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BootstrapState(Protocol):
    """Facade over ``clawcodex_ext.bootstrap.state`` cost/timing getters.

    Default impl (``ClawcodexBootstrapState``) forwards every call to the
    upstream module; tests can substitute a stub to control session-level
    accumulators.
    """

    def get_total_cost_usd(self) -> float: ...

    def get_total_api_duration(self) -> int: ...

    def get_total_api_duration_without_retries(self) -> int: ...

    def get_total_tool_duration(self) -> int: ...

    def get_total_lines_added(self) -> int: ...

    def get_total_lines_removed(self) -> int: ...

    def get_start_time(self) -> int | None: ...

    def get_model_usage(self) -> dict[str, Any]: ...


__all__ = ["BootstrapState"]
