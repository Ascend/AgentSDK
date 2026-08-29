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

from __future__ import annotations

import pytest
from src.plugins.builtin_plugins import (
    BUILTIN_MARKETPLACE_NAME,
    clear_builtin_plugins,
    get_builtin_plugin_definition,
    get_builtin_plugins,
    is_builtin_plugin_id,
    register_builtin_plugin,
)
from src.plugins.types import BuiltinPluginDefinition


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_builtin_plugins()
    yield  # type: ignore[misc]
    clear_builtin_plugins()


class TestBuiltinPlugins:
    def test_register_and_get(self) -> None:
        register_builtin_plugin(
            BuiltinPluginDefinition(
                name="test-plugin",
                description="A test plugin",
            )
        )
        result = get_builtin_plugins()
        assert len(result["enabled"]) == 1
        assert result["enabled"][0].name == "test-plugin"
        assert result["enabled"][0].is_builtin is True

    def test_disabled_by_default(self) -> None:
        register_builtin_plugin(
            BuiltinPluginDefinition(
                name="disabled-plugin",
                description="Disabled",
                default_enabled=False,
            )
        )
        result = get_builtin_plugins()
        assert len(result["enabled"]) == 0
        assert len(result["disabled"]) == 1
        assert result["disabled"][0].name == "disabled-plugin"

    def test_is_available(self) -> None:
        register_builtin_plugin(
            BuiltinPluginDefinition(
                name="unavailable",
                description="Not available",
                is_available=lambda: False,
            )
        )
        result = get_builtin_plugins()
        assert len(result["enabled"]) == 0
        assert len(result["disabled"]) == 0

    def test_get_definition(self) -> None:
        register_builtin_plugin(
            BuiltinPluginDefinition(
                name="my-plugin",
                description="My plugin",
                version="2.0.0",
            )
        )
        defn = get_builtin_plugin_definition("my-plugin")
        assert defn is not None
        assert defn.version == "2.0.0"

    def test_get_nonexistent_definition(self) -> None:
        assert get_builtin_plugin_definition("nonexistent") is None


class TestIsBuiltinPluginId:
    def test_builtin_id(self) -> None:
        assert is_builtin_plugin_id(f"test@{BUILTIN_MARKETPLACE_NAME}") is True

    def test_non_builtin_id(self) -> None:
        assert is_builtin_plugin_id("test@marketplace") is False

    def test_no_at_sign(self) -> None:
        assert is_builtin_plugin_id("test") is False


class TestClearBuiltinPlugins:
    def test_clear(self) -> None:
        register_builtin_plugin(BuiltinPluginDefinition(name="temp", description="temp"))
        assert len(get_builtin_plugins()["enabled"]) == 1
        clear_builtin_plugins()
        assert len(get_builtin_plugins()["enabled"]) == 0
