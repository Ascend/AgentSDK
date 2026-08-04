#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for ``clawcodex_ext.feature_gate.decorators``."""

from unittest.mock import MagicMock

import pytest

from clawcodex_ext.feature_gate.decorators import (
    feature_gated,
    feature_gated_class,
    guarded_call,
    guarded_is_enabled,
)
from clawcodex_ext.feature_gate import get_registry
from clawcodex_ext.feature_gate.types import FeatureFlag


@pytest.fixture(autouse=True)
def fresh_registry():
    """Reset the global singleton before and after each test."""
    from clawcodex_ext.feature_gate import reset_registry

    reset_registry()
    yield
    reset_registry()  # cleanup


def _reg():
    """Convenience: get the global registry (singleton)."""
    return get_registry()


class TestFeatureGatedDecorator:
    """Test @feature_gated function decorator."""

    def test_enabled_returns_original(self):
        """When feature is enabled, the original function is returned."""
        _reg().register(FeatureFlag(name="fg_on", default=True))

        @feature_gated("fg_on")
        def my_func():
            return 42

        assert my_func() == 42

    def test_disabled_returns_noop(self):
        """When feature is disabled and no fallback, a no-op stub is returned."""
        _reg().register(FeatureFlag(name="fg_off", default=False))

        @feature_gated("fg_off")
        def my_func():
            return 42

        result = my_func()
        assert result is None

    def test_disabled_with_fallback(self):
        """When feature is disabled, fallback is returned."""
        _reg().register(FeatureFlag(name="fg_fb", default=False))

        def fallback():
            return "fallback_value"

        @feature_gated("fg_fb", fallback=fallback)
        def primary():
            return "primary_value"

        assert primary() == "fallback_value"

    def test_preserves_metadata(self):
        """The decorator preserves the original function's __name__ and __doc__."""
        _reg().register(FeatureFlag(name="fg_meta", default=True))

        @feature_gated("fg_meta")
        def documented_func():
            """This is a docstring."""
            return True

        assert documented_func.__name__ == "documented_func"


class TestFeatureGatedClassDecorator:
    """Test @feature_gated_class class decorator."""

    def test_enabled_returns_class(self):
        """When feature is enabled, the class is returned as-is."""
        _reg().register(FeatureFlag(name="fgc_on", default=True))

        @feature_gated_class("fgc_on")
        class MyClass:
            value = 10

        assert MyClass.value == 10

    def test_disabled_no_fallback(self):
        """When feature is disabled and no fallback, the class is returned unchanged."""
        _reg().register(FeatureFlag(name="fgc_off", default=False))

        @feature_gated_class("fgc_off")
        class MyClass:
            value = 20

        assert MyClass.value == 20

    def test_disabled_with_fallback(self):
        """When feature is disabled, fallback class is returned."""
        _reg().register(FeatureFlag(name="fgc_fb", default=False))

        class FallbackClass:
            value = 99

        @feature_gated_class("fgc_fb", fallback_cls=FallbackClass)
        class PrimaryClass:
            value = 1

        assert PrimaryClass.value == 99

    def test_dependency_failure(self):
        """When feature is enabled but deps are missing, raises RuntimeError."""
        _reg().register(FeatureFlag(name="dep_parent", default=False))
        _reg().register(
            FeatureFlag(
                name="dep_child",
                default=True,
                deps=["dep_parent"],
            )
        )

        with pytest.raises(RuntimeError, match="requires but is missing"):

            @feature_gated_class("dep_child")
            class ChildClass:
                pass

    def test_mutex_failure(self):
        """When feature is enabled but mutex conflicts exist, raises RuntimeError."""
        _reg().register(FeatureFlag(name="mutex_a", default=True))
        _reg().register(
            FeatureFlag(
                name="mutex_b",
                default=True,
                mutex_with=["mutex_a"],
            )
        )

        with pytest.raises(RuntimeError, match="conflicts with"):

            @feature_gated_class("mutex_b")
            class BClass:
                pass


class TestGuardedCall:
    """Test guarded_call helper."""

    def test_enabled_invokes_function(self):
        """guarded_call invokes the function when feature is enabled."""
        _reg().register(FeatureFlag(name="gc_on", default=True))

        mock_fn = MagicMock(return_value="result")
        result = guarded_call("gc_on", mock_fn, "arg1", kwarg="val")
        assert result == "result"
        mock_fn.assert_called_once_with("arg1", kwarg="val")

    def test_disabled_returns_none(self):
        """guarded_call returns None when feature is disabled."""
        _reg().register(FeatureFlag(name="gc_off", default=False))

        mock_fn = MagicMock(return_value="result")
        result = guarded_call("gc_off", mock_fn)
        assert result is None
        mock_fn.assert_not_called()


class TestGuardedIsEnabled:
    """Test guarded_is_enabled helper."""

    def test_reflects_registry_state(self):
        """guarded_is_enabled reflects the registry state."""
        _reg().register(FeatureFlag(name="gie_test", default=True))
        assert guarded_is_enabled("gie_test") is True

        _reg().disable_feature("gie_test")
        assert guarded_is_enabled("gie_test") is False

    def test_unknown_returns_false(self):
        """Unknown features return False."""
        assert guarded_is_enabled("nonexistent_feature") is False
