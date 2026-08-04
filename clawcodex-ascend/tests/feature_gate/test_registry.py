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

"""Tests for ``clawcodex_ext.feature_gate.registry``."""

import tempfile
from pathlib import Path

import pytest

from clawcodex_ext.feature_gate.config import ConfigStore
from clawcodex_ext.feature_gate.registry import FeatureRegistry
from clawcodex_ext.feature_gate.types import FeatureFlag


@pytest.fixture
def registry():
    """Return a fresh FeatureRegistry with a temp config store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        store = ConfigStore(config_dir=config_dir)
        reg = FeatureRegistry(config_store=store)
        yield reg


@pytest.fixture
def sample_flags():
    """Return a list of FeatureFlag instances for testing."""
    return [
        FeatureFlag(name="feature_a", default=True, deps=[], mutex_with=[]),
        FeatureFlag(
            name="feature_b",
            default=False,
            deps=["feature_a"],
            mutex_with=["feature_c"],
        ),
        FeatureFlag(
            name="feature_c",
            default=False,
            deps=[],
            mutex_with=["feature_b"],
        ),
        FeatureFlag(
            name="feature_d",
            default=False,
            deps=["feature_b", "feature_a"],
            mutex_with=[],
        ),
    ]


class TestRegistration:
    """Test feature flag registration."""

    def test_register_single(self, registry):
        flag = FeatureFlag(name="test", default=True)
        registry.register(flag)
        assert registry.is_enabled("test") is True

    def test_register_duplicate_raises(self, registry):
        registry.register(FeatureFlag(name="dup", default=True))
        with pytest.raises(ValueError, match="Duplicate"):
            registry.register(FeatureFlag(name="dup", default=False))

    def test_register_many(self, registry, sample_flags):
        registry.register_many(sample_flags)
        assert set(registry.list_features()) == {"feature_a", "feature_b", "feature_c", "feature_d"}

    def test_unregister(self, registry):
        registry.register(FeatureFlag(name="temp", default=True))
        assert registry.is_enabled("temp") is True
        registry.unregister("temp")
        assert registry.is_enabled("temp") is False

    def test_unregister_unknown_is_noop(self, registry):
        registry.unregister("nonexistent")  # Should not raise

    def test_get_flag(self, registry):
        flag = FeatureFlag(name="gfg", default=True, description="test")
        registry.register(flag)
        retrieved = registry.get_flag("gfg")
        assert retrieved is not None
        assert retrieved.name == "gfg"
        assert retrieved.description == "test"

    def test_get_flag_unknown_returns_none(self, registry):
        assert registry.get_flag("unknown") is None

    def test_get_state_returns_none_for_unknown(self, registry):
        assert registry.get_state("unknown") is None

    def test_get_state_returns_bool_for_known(self, registry):
        registry.register(FeatureFlag(name="gs", default=True))
        assert registry.get_state("gs") is True


class TestResolution:
    """Test is_enabled resolution order."""

    def test_default_value(self, registry, sample_flags):
        registry.register_many(sample_flags)
        assert registry.is_enabled("feature_a") is True  # default=True
        assert registry.is_enabled("feature_b") is False  # default=False

    def test_override_wins(self, registry):
        registry.register(FeatureFlag(name="ov", default=False))
        registry.set_override("ov", True)
        assert registry.is_enabled("ov") is True

    def test_disable_override(self, registry):
        registry.register(FeatureFlag(name="dd", default=True))
        registry.set_override("dd", False)
        assert registry.is_enabled("dd") is False

    def test_clear_override(self, registry):
        registry.register(FeatureFlag(name="co", default=False))
        registry.set_override("co", True)
        assert registry.is_enabled("co") is True
        registry.clear_override("co")
        assert registry.is_enabled("co") is False

    def test_clear_all_overrides(self, registry):
        registry.register(FeatureFlag(name="ca", default=False))
        registry.register(FeatureFlag(name="cb", default=True))
        registry.set_override("ca", True)
        registry.set_override("cb", False)
        registry.clear_all_overrides()
        assert registry.is_enabled("ca") is False
        assert registry.is_enabled("cb") is True

    def test_env_var_override(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="env_test", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_test", "true")
        assert registry.is_enabled("env_test") is True

    def test_env_var_yes(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="env_yes", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_yes", "yes")
        assert registry.is_enabled("env_yes") is True

    def test_env_var_1(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="env_1", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_1", "1")
        assert registry.is_enabled("env_1") is True

    def test_env_var_case_insensitive(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="env_ci", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_ci", "TRUE")
        assert registry.is_enabled("env_ci") is True

    def test_env_var_disabled(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="env_off", default=True))
        monkeypatch.setenv("CLAWCODEX_FEATURE_env_off", "false")
        assert registry.is_enabled("env_off") is False

    def test_override_takes_priority_over_env(self, registry, monkeypatch):
        registry.register(FeatureFlag(name="pri", default=False))
        monkeypatch.setenv("CLAWCODEX_FEATURE_pri", "true")
        registry.set_override("pri", False)
        assert registry.is_enabled("pri") is False

    def test_unknown_feature_disabled(self, registry):
        assert registry.is_enabled("totally_unknown") is False

    def test_config_file_persistence(self, registry, sample_flags):
        """Test that config file values are respected."""
        import json

        registry.register_many(sample_flags)

        # Write a config that enables feature_b
        config_file = registry._config_store.config_file
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({"feature_b": True, "feature_c": True}, f)

        # Reload to pick up the config
        registry._config_store.reload()

        assert registry.is_enabled("feature_b") is True
        assert registry.is_enabled("feature_c") is True


class TestDepsAndMutex:
    """Test dependency and mutex checks."""

    def test_check_deps_satisfied(self, registry, sample_flags):
        registry.register_many(sample_flags)
        registry.enable_feature("feature_a")
        missing = registry.check_deps("feature_b")
        assert missing == []

    def test_check_deps_missing(self, registry):
        """feature_b depends on feature_a; feature_a defaults to disabled."""
        registry.register(FeatureFlag(name="feature_a", default=False))
        registry.register(
            FeatureFlag(
                name="feature_b",
                default=True,
                deps=["feature_a"],
            )
        )
        missing = registry.check_deps("feature_b")
        assert "feature_a" in missing

    def test_check_deps_multi_missing(self, registry):
        registry.register(FeatureFlag(name="dep_base", default=False))
        registry.register(FeatureFlag(name="dep_child", default=False, deps=["dep_base", "another"]))
        missing = registry.check_deps("dep_child")
        assert "dep_base" in missing
        assert "another" in missing

    def test_check_deps_no_deps(self, registry):
        registry.register(FeatureFlag(name="no_deps", default=True))
        assert registry.check_deps("no_deps") == []

    def test_check_deps_unknown_feature(self, registry):
        assert registry.check_deps("nonexistent") == []

    def test_check_mutex_no_conflict(self, registry, sample_flags):
        registry.register_many(sample_flags)
        conflicts = registry.check_mutex("feature_b")
        assert "feature_c" not in conflicts

    def test_check_mutex_conflict(self, registry, sample_flags):
        registry.register_many(sample_flags)
        registry.enable_feature("feature_c")
        conflicts = registry.check_mutex("feature_b")
        assert "feature_c" in conflicts

    def test_check_mutex_no_mutex(self, registry):
        registry.register(FeatureFlag(name="no_mutex", default=True))
        assert registry.check_mutex("no_mutex") == []

    def test_validate_registration_clean(self, registry, sample_flags):
        registry.register_many(sample_flags)
        registry.enable_feature("feature_a")
        ok, errors = registry.validate_registration("feature_b")
        assert ok is True
        assert errors == []

    def test_validate_registration_missing_deps(self, registry):
        """Validate fails when dependency is not enabled."""
        registry.register(FeatureFlag(name="dep_parent", default=False))
        registry.register(
            FeatureFlag(
                name="dep_child",
                default=True,
                deps=["dep_parent"],
            )
        )
        ok, errors = registry.validate_registration("dep_child")
        assert ok is False
        assert any("Missing dependencies" in e for e in errors)

    def test_validate_registration_mutex_conflict(self, registry, sample_flags):
        registry.register_many(sample_flags)
        registry.enable_feature("feature_c")
        ok, errors = registry.validate_registration("feature_b")
        assert ok is False
        assert any("Mutex conflicts" in e for e in errors)


class TestBulkOperations:
    """Test bulk enable/disable and config save."""

    def test_enable_feature(self, registry):
        registry.register(FeatureFlag(name="bf", default=False))
        registry.enable_feature("bf")
        assert registry.is_enabled("bf") is True

    def test_disable_feature(self, registry):
        registry.register(FeatureFlag(name="df", default=True))
        registry.disable_feature("df")
        assert registry.is_enabled("df") is False

    def test_save_config(self, registry, sample_flags):
        registry.register_many(sample_flags)
        registry.enable_feature("feature_a")
        registry.disable_feature("feature_b")
        registry.save_config()

        # Verify saved state
        states = registry.get_effective_states()
        assert states["feature_a"] is True
        assert states["feature_b"] is False

    def test_get_effective_states(self, registry, sample_flags):
        registry.register_many(sample_flags)
        states = registry.get_effective_states()
        assert states["feature_a"] is True
        assert states["feature_b"] is False
        assert states["feature_c"] is False

    def test_list_features(self, registry, sample_flags):
        registry.register_many(sample_flags)
        features = registry.list_features()
        assert set(features) == {"feature_a", "feature_b", "feature_c", "feature_d"}
