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

"""Dream thresholds + enable flag — F-100.

Mirrors ``typescript/src/services/autoDream/config.ts`` +
``autoDream.ts::getConfig()``.

Two layers:

1. **Enable** — :func:`is_auto_dream_enabled`. User setting overrides
   the env var; the env var overrides the default. Mirrors the
   upstream ``isAutoDreamEnabled`` shape (setting > flag > default).
2. **Thresholds** — :class:`DreamConfig` carries the scheduling knobs
   (``min_hours`` and ``min_sessions``). Settable via
   :func:`set_dream_config` so tests can swap defaults cheaply.

Note: clawcodex does **not** model the upstream ``KAIROS`` /
``KAIROS_DREAM`` feature flags. Decision #1 in PROGRESS.md §十三 —
F-100 ships dream unconditionally under the user setting (no
A/B gate). KAIROS branches stay available via
:func:`clawcodex_ext.dreaming.paths.is_kairos_active` for follow-up
work.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DREAM_CONFIG",
    "DreamConfig",
    "get_dream_config",
    "is_auto_dream_enabled",
    "set_dream_config",
]


@dataclass(frozen=True)
class DreamConfig:
    """Scheduling thresholds for auto-dream.

    Attributes:
        min_hours: Hours that must elapse since ``lastConsolidatedAt``
            before the time gate opens.
        min_sessions: Number of session transcripts touched since
            ``lastConsolidatedAt`` required for the session gate to
            open. Set to ``0`` to disable the session gate.
    """

    min_hours: float = 24.0
    min_sessions: int = 5


DEFAULT_DREAM_CONFIG = DreamConfig(min_hours=24.0, min_sessions=5)

# Mutable copy used by the service main loop. Tests overwrite via
# :func:`set_dream_config`.
_active_config: DreamConfig = DEFAULT_DREAM_CONFIG


def get_dream_config() -> DreamConfig:
    """Return the currently-active dream config."""
    return _active_config


def set_dream_config(cfg: DreamConfig) -> None:
    """Override the active dream config. Tests use this; production
    should leave it alone. Not thread-safe — call only at startup.
    """
    global _active_config
    _active_config = cfg


def is_auto_dream_enabled() -> bool:
    """Whether background memory consolidation should run.

    Resolution order:

    1. ``CLAWCODEX_DISABLE_AUTO_DREAM`` env (truthy → off; falsy → on).
    2. ``clawcodex.toml`` / merged settings key ``auto_dream_enabled``
       (any source — supports project-level opt-out).
    3. Default: enabled when auto-memory is enabled.
    """
    env = os.environ.get("CLAWCODEX_DISABLE_AUTO_DREAM", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return False
    if env in ("0", "false", "no", "off"):
        return True

    try:
        from clawcodex_ext.settings.settings import get_settings  # pylint: disable=no-name-in-module

        flag = getattr(get_settings(), "auto_dream_enabled", None)
        if flag is False:
            return False
    except Exception:
        # Settings module errors should not silently disable dream;
        # fall through to the auto-memory gate.
        logger.debug("Dream settings lookup failed; using fallback", exc_info=True)

    try:
        from clawcodex_ext.dreaming.paths import is_auto_memory_enabled

        return is_auto_memory_enabled()
    except Exception:
        return True
