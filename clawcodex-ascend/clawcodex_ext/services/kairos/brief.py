#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Brief summary builder: deterministic status summaries from a snapshot.

The :class:`BriefSummaryBuilder` produces a one- or two-line Markdown
summary suitable for CLI status bars, daily logs, or a chat reply
when an agent is asked "what's going on?". Output is deterministic for
a given snapshot + generation time so tests and replays stay stable.

The builder deliberately does no LLM calls; the headline is a small
templated string and the body is a controlled enumeration of the
snapshot's fields. This keeps the service dependency-free and lets
downstream callers layer an LLM polisher on top if they want one.

Naming note: this is a *service-layer* formatter. The user-facing
:class:`BriefTool` (in :mod:`src.tool_system.tools.brief`) emits a
brief informational message and is a separate concern; the two share
the "brief" word but produce different artifacts (a status summary vs.
a tool message).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .exceptions import BriefGenerationError
from .models import BriefSummarySnapshot, format_local_timestamp


class BriefSummaryBuilder:
    """Deterministic brief summary builder."""

    def __init__(
        self,
        *,
        max_pending_tasks: int = 5,
        include_metadata: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(max_pending_tasks, int) or max_pending_tasks < 1:
            raise ValueError(f"max_pending_tasks must be a positive int (got {max_pending_tasks!r})")
        self._max_pending = max_pending_tasks
        self._include_metadata = bool(include_metadata)
        self._clock = clock or time.time

    def build(self, snapshot: BriefSummarySnapshot) -> str:
        """Build the brief summary as a Markdown string.

        Renamed from ``generate`` so the public method name matches the
        class name (``BriefSummaryBuilder.build``).
        """
        if not isinstance(snapshot, BriefSummarySnapshot):
            raise BriefGenerationError("build() expects a BriefSummarySnapshot")
        lines: list[str] = []
        # Headline: agent + session + tick number.
        headline = f"[brief] agent={snapshot.agent_id} session={snapshot.session_id} tick=#{snapshot.tick_number}"
        lines.append(headline)
        if snapshot.last_action:
            lines.append(f"last: {snapshot.last_action}")
        if snapshot.pending_tasks:
            shown = snapshot.pending_tasks[: self._max_pending]
            extra = len(snapshot.pending_tasks) - len(shown)
            lines.append("pending:")
            for task in shown:
                lines.append(f"  - {task}")
            if extra > 0:
                lines.append(f"  …(+{extra} more)")
        else:
            lines.append("pending: (none)")
        if self._include_metadata and snapshot.metadata:
            lines.append("meta:")
            for key, value in snapshot.metadata.items():
                lines.append(f"  {key}: {value}")
        lines.append(f"captured: {format_local_timestamp(snapshot.captured_at)}")
        return "\n".join(lines)

    # Backwards-compatible alias kept during the rename so existing
    # callers (and any tests written against the original name) keep
    # working until they migrate.
    def generate(self, snapshot: BriefSummarySnapshot) -> str:
        """Alias for :meth:`build` retained for back-compat during rename."""
        return self.build(snapshot)


__all__ = [
    "BriefSummaryBuilder",
]
