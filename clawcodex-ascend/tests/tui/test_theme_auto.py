#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Unit tests for :func:`src.tui.theme.resolve_auto_theme`."""

from __future__ import annotations


from src.tui.theme import get_palette, list_theme_names, resolve_auto_theme


def test_list_theme_names_starts_with_auto():
    names = list_theme_names()
    assert names[0] == "auto"
    assert {"dark", "light", "claude"}.issubset(set(names))


def test_forced_env_wins():
    resolved = resolve_auto_theme(env={"CLAWCODEX_THEME": "light"})
    assert resolved == "light"


def test_forced_env_unknown_falls_through():
    resolved = resolve_auto_theme(env={"CLAWCODEX_THEME": "solarized"})
    # Unknown forced value must not short-circuit — fallback wins.
    assert resolved == "dark"


def test_colorfgbg_dark_detected():
    resolved = resolve_auto_theme(env={"COLORFGBG": "15;0"})
    assert resolved == "dark"


def test_colorfgbg_light_detected():
    resolved = resolve_auto_theme(env={"COLORFGBG": "0;15"})
    assert resolved == "light"


def test_empty_env_falls_back_to_dark():
    resolved = resolve_auto_theme(env={})
    assert resolved == "dark"


def test_get_palette_auto_honours_env():
    palette = get_palette("auto", env={"COLORFGBG": "0;15"})
    assert palette.name == "light"
