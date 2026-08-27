#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# ruff: noqa: UP009

"""Deliver formatted orchestrator events through an injected channel callback."""

from __future__ import annotations

import logging
from collections.abc import Callable

from .events.formatter import format_event
from .events.types import OrchestratorEvent

logger = logging.getLogger(__name__)


class ChannelProgressSink:
    """Format events and pass them to the configured channel callback."""

    def __init__(
        self,
        deliver: Callable[[OrchestratorEvent, str], None],
    ) -> None:
        self._deliver = deliver

    def __call__(self, event: OrchestratorEvent) -> None:
        text = format_event(event)
        try:
            self._deliver(event, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChannelProgressSink deliver failed error_type=%s",
                type(exc).__name__,
            )


__all__ = ["ChannelProgressSink"]
