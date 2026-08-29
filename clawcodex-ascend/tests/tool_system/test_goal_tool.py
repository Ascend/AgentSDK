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

"""Spec-1 negative assertions for the removed legacy ``Goal`` tool."""

from __future__ import annotations

import importlib

import pytest

from src.tool_system.defaults import build_default_registry


def test_legacy_goal_tool_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("clawcodex_ext.goal.tool")


def test_default_tool_registry_does_not_register_legacy_goal_tool():
    registry = build_default_registry(include_user_tools=False, load_agent_tools=False)

    assert registry.get("Goal") is None
    assert all(tool.name != "Goal" for tool in registry.list_tools())


def test_extension_tool_bundle_does_not_include_legacy_goal_tool():
    from extensions.tool_system_ext.registration import EXTENSION_TOOLS

    assert all(tool.name != "Goal" for tool in EXTENSION_TOOLS)
