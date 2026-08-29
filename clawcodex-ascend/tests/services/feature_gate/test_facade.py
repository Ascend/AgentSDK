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

"""Tests for ``src.services.feature_gate`` (Layer 0 facade).

These tests verify that the ``src/services/feature_gate/`` re-export
package correctly exposes the canonical Layer 1 implementation from
``clawcodex_ext/feature_gate/``.
"""
# pylint: disable=redefined-outer-name, reimported

from __future__ import annotations

import pytest

from src.services.feature_gate import (
    ConfigStore,
    FeatureFlag,
    FeatureRegistry,
    add_feature_gate_args,
    apply_feature_gate_args,
    conditional_register,
    feature_gated,
    feature_gated_class,
    feature_gated_function,
    get_registry,
    guarded_call,
    guarded_is_enabled,
    handle_list_features,
    reset_registry,
    run_feature_command,
)


class TestFacadeImports:
    """Verify that all expected symbols are importable from src.services.feature_gate."""

    def test_get_registry_returns_singleton(self):
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
        assert isinstance(reg1, FeatureRegistry)

    def test_reset_registry_returns_fresh(self):
        reg1 = get_registry()
        reg2 = reset_registry()
        assert isinstance(reg2, FeatureRegistry)
        assert reg1 is not reg2  # reset creates a new instance

    def test_feature_flag_creation(self):
        flag = FeatureFlag(name="test", default=True, deps=["a"], mutex_with=["b"], description="desc")
        assert flag.name == "test"
        assert flag.default is True
        assert flag.deps == ["a"]
        assert flag.mutex_with == ["b"]
        assert flag.description == "desc"

    def test_config_store_class(self):
        assert ConfigStore is not None

    def test_decorator_symbols(self):
        assert callable(feature_gated)
        assert callable(feature_gated_class)
        assert callable(feature_gated_function)

    def test_guard_helpers(self):
        assert callable(guarded_call)
        assert callable(guarded_is_enabled)

    def test_cli_symbols(self):
        assert callable(run_feature_command)
        assert callable(handle_list_features)
        assert callable(add_feature_gate_args)
        assert callable(apply_feature_gate_args)

    def test_conditional_register(self):
        assert callable(conditional_register)


class TestFacadeFunctionality:
    """Minimal functional tests through the facade."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        reset_registry()
        yield
        reset_registry()

    def test_register_and_is_enabled(self):
        reg = get_registry()
        reg.register(FeatureFlag(name="facade_feat", default=True))
        assert reg.is_enabled("facade_feat") is True

    def test_env_override(self, monkeypatch):
        reg = get_registry()
        reg.register(FeatureFlag(name="env_feat", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_feat", "true")
        assert reg.is_enabled("env_feat") is True

    def test_dependency_check(self):
        reg = get_registry()
        reg.register(FeatureFlag(name="parent", default=False))
        reg.register(FeatureFlag(name="child", default=True, deps=["parent"]))
        missing = reg.check_deps("child")
        assert "parent" in missing

    def test_mutex_check(self):
        reg = get_registry()
        reg.register(FeatureFlag(name="a", default=True))
        reg.register(FeatureFlag(name="b", default=False, mutex_with=["a"]))
        reg.enable_feature("b")
        conflicts = reg.check_mutex("b")
        assert "a" in conflicts

    def test_feature_gated_decorator(self):
        reg = get_registry()
        reg.register(FeatureFlag(name="on", default=True))

        @feature_gated("on")
        def my_func():
            return 42

        assert my_func() == 42

    def test_feature_gated_fallback(self):
        reg = get_registry()
        reg.register(FeatureFlag(name="off", default=False))

        def fallback():
            return "fallback"

        @feature_gated("off", fallback=fallback)
        def my_func():
            return 42

        assert my_func() == "fallback"

    def test_config_load_save(self, tmp_path):
        reg = get_registry()
        reg.register(FeatureFlag(name="persist", default=False))
        reg.enable_feature("persist")
        reg.save_config()

        # Verify via a fresh ConfigStore that the file was written
        from src.services.feature_gate import ConfigStore

        store = ConfigStore(config_dir=reg._config_store.config_dir)
        assert store.get("persist") is True

    def test_cli_add_args(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_feature_gate_args(parser)
        args = parser.parse_args(["--enable", "a", "b"])
        assert args.enable == ["a", "b"]

    def test_cli_apply_args(self):
        import argparse

        reg = get_registry()
        reg.register(FeatureFlag(name="cli_feat", default=False))
        parser = argparse.ArgumentParser()
        add_feature_gate_args(parser)
        args = parser.parse_args(["--enable", "cli_feat"])
        apply_feature_gate_args(args)
        assert reg.is_enabled("cli_feat") is True
