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

"""Tests for ``clawcodex_ext.feature_gate`` package-level API."""

import pytest

from clawcodex_ext.feature_gate import (
    FeatureFlag,
    FeatureRegistry,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def fresh_singleton():
    """Reset the global singleton before each test."""
    reset_registry()
    yield
    reset_registry()


class TestSingleton:
    """Test the get_registry() / reset_registry() singleton pattern."""

    def test_get_registry_returns_instance(self):
        reg = get_registry()
        assert isinstance(reg, FeatureRegistry)

    def test_get_registry_returns_same_instance(self):
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_reset_registry_returns_new_instance(self):
        reg1 = get_registry()
        reg1.register(FeatureFlag(name="before_reset", default=True))
        assert reg1.is_enabled("before_reset") is True

        new_reg = reset_registry()
        assert new_reg is not reg1
        assert new_reg.is_enabled("before_reset") is False

    def test_reset_registry_is_idempotent(self):
        reset_registry()
        reset_registry()
        # Should not raise


class TestPackageImports:
    """Verify all public symbols are importable from the package."""

    def test_import_feature_flag(self):
        from clawcodex_ext.feature_gate import FeatureFlag

        flag = FeatureFlag(name="test")
        assert flag.name == "test"

    def test_import_feature_registry(self):
        from clawcodex_ext.feature_gate import FeatureRegistry

        reg = FeatureRegistry()
        assert isinstance(reg, FeatureRegistry)

    def test_import_get_registry(self):
        from clawcodex_ext.feature_gate import get_registry

        assert callable(get_registry)

    def test_import_reset_registry(self):
        from clawcodex_ext.feature_gate import reset_registry

        assert callable(reset_registry)

    def test_import_decorators(self):
        from clawcodex_ext.feature_gate import feature_gated, feature_gated_class

        assert callable(feature_gated)
        assert callable(feature_gated_class)

    def test_import_guarded_helpers(self):
        from clawcodex_ext.feature_gate import guarded_call, guarded_is_enabled

        assert callable(guarded_call)
        assert callable(guarded_is_enabled)
