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

"""Tests for ``clawcodex_ext.services.feature_gate`` facade module."""

from __future__ import annotations

import pytest

from clawcodex_ext.feature_gate import reset_registry
from clawcodex_ext.services.feature_gate import (
    FeatureFlag,
    FeatureRegistry,
    ConfigStore,
    feature_gated,
    feature_gated_class,
    guarded_call,
    guarded_is_enabled,
    get_registry,
    register_defaults,
    conditional_register,
    feature_gated_function,
    add_feature_gate_args,
    apply_feature_gate_args,
    handle_list_features,
    FeatureConfigStore,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_registry()
    yield
    reset_registry()


class TestFacadeImports:
    """Verify all public symbols are importable from the services facade."""

    def test_core_types(self):
        assert FeatureFlag is not None
        assert FeatureRegistry is not None
        assert ConfigStore is not None
        assert FeatureConfigStore is ConfigStore

    def test_decorators(self):
        assert callable(feature_gated)
        assert callable(feature_gated_class)
        assert callable(guarded_call)
        assert callable(guarded_is_enabled)

    def test_singleton(self):
        assert callable(get_registry)
        assert callable(register_defaults)

    def test_compat_aliases(self):
        assert callable(feature_gated_function)
        assert callable(add_feature_gate_args)
        assert callable(apply_feature_gate_args)
        assert callable(handle_list_features)
        assert callable(conditional_register)


class TestFacadeConditionalRegister:
    """Test the conditional_register helper."""

    def test_returns_cls_when_enabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("cr_on", default=True))

        class Target:
            pass

        result = conditional_register("cr_on", Target)
        assert result is Target

    def test_returns_none_when_disabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("cr_off", default=False))

        class Target:
            pass

        result = conditional_register("cr_off", Target)
        assert result is None


class TestFacadeFeatureGatedFunction:
    """Test the feature_gated_function alias."""

    def test_returns_decorator(self):
        reg = get_registry()
        reg.register(FeatureFlag("fgf", default=True))

        decorated = feature_gated_function("fgf")

        @decorated
        def fn():
            return 42

        assert fn() == 42


class TestFacadeArgsIntegration:
    """Test add_feature_gate_args / apply_feature_gate_args."""

    def test_add_args_creates_mutually_exclusive_group(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_feature_gate_args(parser)
        # Should not raise
        parsed = parser.parse_args(["--enable", "FOO"])
        assert parsed.enable == ["FOO"]

    def test_apply_enable(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_feature_gate_args(parser)
        parsed = parser.parse_args(["--enable", "TEST_FEAT"])

        reg = get_registry()
        reg.register(FeatureFlag("TEST_FEAT", default=False))
        apply_feature_gate_args(parsed)
        assert reg.is_enabled("TEST_FEAT") is True

    def test_apply_disable(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_feature_gate_args(parser)
        parsed = parser.parse_args(["--disable", "TEST_FEAT2"])

        reg = get_registry()
        reg.register(FeatureFlag("TEST_FEAT2", default=True))
        apply_feature_gate_args(parsed)
        assert reg.is_enabled("TEST_FEAT2") is False
