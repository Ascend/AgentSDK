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

"""ClawcodexBootstrapState — concrete ``BootstrapState`` Protocol adapter.

Thin wrapper over the eight cost/timing getters on
``clawcodex_ext.bootstrap.state`` so ``agent_runner._save_json_snapshot``
does not import that module directly.

Getters forward 1:1 with no extra state or cache. Upstream is imported
lazily in the constructor (adapters may reference clawcodex_ext).
"""

from __future__ import annotations

from typing import Any

from extensions.orchestrator_runtime.utils.bootstrap_state import BootstrapState


class ClawcodexBootstrapState(BootstrapState):
    """Forward every getter to ``clawcodex_ext.bootstrap.state``."""

    def get_total_cost_usd(self) -> float:
        from clawcodex_ext.bootstrap.state import get_total_cost_usd

        return float(get_total_cost_usd())

    def get_total_api_duration(self) -> int:
        from clawcodex_ext.bootstrap.state import get_total_api_duration

        return int(get_total_api_duration())

    def get_total_api_duration_without_retries(self) -> int:
        from clawcodex_ext.bootstrap.state import (
            get_total_api_duration_without_retries,
        )

        return int(get_total_api_duration_without_retries())

    def get_total_tool_duration(self) -> int:
        from clawcodex_ext.bootstrap.state import get_total_tool_duration

        return int(get_total_tool_duration())

    def get_total_lines_added(self) -> int:
        from clawcodex_ext.bootstrap.state import get_total_lines_added

        return int(get_total_lines_added())

    def get_total_lines_removed(self) -> int:
        from clawcodex_ext.bootstrap.state import get_total_lines_removed

        return int(get_total_lines_removed())

    def get_start_time(self) -> int | None:
        from clawcodex_ext.bootstrap.state import get_start_time

        return get_start_time()

    def get_model_usage(self) -> dict[str, Any]:
        from clawcodex_ext.bootstrap.state import get_model_usage

        return dict(get_model_usage())


__all__ = ["ClawcodexBootstrapState"]
