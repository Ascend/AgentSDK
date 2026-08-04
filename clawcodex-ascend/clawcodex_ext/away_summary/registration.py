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

"""Command registration for Away Summary."""

from __future__ import annotations

import logging
from typing import Any

from clawcodex_ext.away_summary.config import (  # pylint: disable=no-name-in-module
    load_away_summary_config,
)

from .command import build_recap_command

logger = logging.getLogger(__name__)


def register_away_summary_commands(registry: Any | None = None) -> None:
    """Register /recap when enabled.

    This is intentionally separate from the Away Summary service/controller so
    future removal of the slash command can leave automatic summaries intact.
    """

    from src.command_system.registry import get_command_registry

    reg = registry or get_command_registry()
    if not load_away_summary_config().recap_command_enabled:
        try:
            reg.unregister("recap")
        except Exception:
            logger.debug("Away Summary command unregister failed", exc_info=True)
        return

    reg.register(build_recap_command())
