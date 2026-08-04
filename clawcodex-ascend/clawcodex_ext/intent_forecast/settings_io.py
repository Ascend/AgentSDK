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

"""Small settings writer for ``settings.intent_forecast``."""

from __future__ import annotations

from typing import Any


def update_intent_forecast_settings(updates: dict[str, Any]) -> None:
    """Persist Intent Forecast settings to the user/global config.

    The settings loader reads extension settings from the nested
    ``settings`` section in ``~/.clawcodex/config.json``. Keep this writer
    intentionally narrow so ``/forecast on|off`` does not serialize merged
    project/local config back into the global file.
    """

    from clawcodex_ext.config import _get_default_manager  # pylint: disable=no-name-in-module
    from clawcodex_ext.settings.settings import invalidate_settings_cache  # pylint: disable=no-name-in-module

    mgr = _get_default_manager()
    cfg = mgr.load_global()
    settings = cfg.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    section = settings.get("intent_forecast")
    if not isinstance(section, dict):
        section = {}
    section.update(updates)
    settings["intent_forecast"] = section
    cfg["settings"] = settings
    mgr.save_global(cfg)
    invalidate_settings_cache()
