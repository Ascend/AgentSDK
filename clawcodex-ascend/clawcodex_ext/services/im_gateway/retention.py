#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Retention cleanup for IM gateway data."""

from __future__ import annotations

import logging

from .config import ReliabilityConfig
from .store import ReliabilityStore

logger = logging.getLogger(__name__)


def run_retention_sweep(
    store: ReliabilityStore,
    reliability: ReliabilityConfig,
) -> dict[str, int]:
    """Apply retention limits to IM gateway data files."""
    if not reliability.retention_enabled:
        return {}
    try:
        removed = store.purge_all(reliability)
        if any(removed.values()):
            logger.info("im_gateway retention sweep: %s", removed)
        return removed
    except Exception:  # noqa: BLE001
        logger.exception("im_gateway retention sweep failed")
        return {}


__all__ = ["run_retention_sweep"]
