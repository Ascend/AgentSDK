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

"""SOPDashboardSource — placeholder adapter for SOP execution state.

The SOP Converter currently has no unified runtime state object, so this
source is intentionally minimal: it accepts an optional provider callable
and forwards whatever entries the provider returns.  When a future SOP
execution engine exposes stage state, passing ``sop_state_provider`` will
automatically surface those stages on the dashboard without changing this
module.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from extensions.capabilities.dashboard_entry import DashboardEntry

logger = logging.getLogger(__name__)

__all__ = ["SOPDashboardSource", "register_sop_dashboard_source"]


class SOPDashboardSource:
    """Read-only dashboard source backed by an optional SOP state provider."""

    source_name = "sop"

    def __init__(
        self,
        sop_state_provider: Optional[Callable[[], list[DashboardEntry] | None]] = None,
        *,
        cache_ttl_ms: int = 5_000,
    ) -> None:
        self._provider = sop_state_provider or (lambda: [])
        self._cache_ttl_ms = int(cache_ttl_ms)

    @property
    def cache_ttl_ms(self) -> int:
        return self._cache_ttl_ms

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        try:
            entries = self._provider()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SOPDashboardSource provider failed: %s", exc)
            return []
        if entries is None:
            return []
        if not isinstance(entries, list):
            return []
        valid: list[DashboardEntry] = []
        for entry in entries:
            if isinstance(entry, DashboardEntry):
                valid.append(entry)
            else:
                logger.debug("SOPDashboardSource skipped non-DashboardEntry: %s", type(entry))
        return valid


def register_sop_dashboard_source(
    provider: Optional[Callable[[], list[DashboardEntry] | None]] = None,
) -> SOPDashboardSource:
    """Register a SOP dashboard source against the default registry.

    Convenience helper for future SOP execution engines; no-op by default.
    """
    from extensions.agent_dashboard import register_dashboard_source

    source = SOPDashboardSource(sop_state_provider=provider)
    register_dashboard_source(source)
    return source
