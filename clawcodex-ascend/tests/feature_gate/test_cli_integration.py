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

"""Tests for feature gate CLI integration in dispatch."""

import pytest

from clawcodex_ext.feature_gate import reset_registry


@pytest.fixture(autouse=True)
def _reset():
    reset_registry()
    yield
    reset_registry()


class TestDispatchFeatureGateOverrides:
    """Test that CLI --enable-feature/--disable-feature apply overrides."""

    def test_apply_feature_gate_overrides_enable(self):
        """--enable-feature sets programmatic override."""
        from clawcodex_ext.cli.dispatch import _apply_feature_gate_overrides
        from clawcodex_ext.feature_gate import get_registry, FeatureFlag

        reg = get_registry()
        reg.register(FeatureFlag("cli_test", default=False))

        class Args:
            enable_feature = ["cli_test"]
            disable_feature = []

        _apply_feature_gate_overrides(Args())
        assert reg.is_enabled("cli_test") is True

    def test_apply_feature_gate_overrides_disable(self):
        """--disable-feature sets programmatic override to False."""
        from clawcodex_ext.cli.dispatch import _apply_feature_gate_overrides
        from clawcodex_ext.feature_gate import get_registry, FeatureFlag

        reg = get_registry()
        reg.register(FeatureFlag("cli_test2", default=True))

        class Args:
            enable_feature = []
            disable_feature = ["cli_test2"]

        _apply_feature_gate_overrides(Args())
        assert reg.is_enabled("cli_test2") is False

    def test_apply_feature_gate_overrides_both(self):
        """Both --enable and --disable can be used together; enable wins."""
        from clawcodex_ext.cli.dispatch import _apply_feature_gate_overrides
        from clawcodex_ext.feature_gate import get_registry, FeatureFlag

        reg = get_registry()
        reg.register(FeatureFlag("cli_test3", default=False))

        class Args:
            enable_feature = ["cli_test3"]
            disable_feature = ["cli_test3"]

        _apply_feature_gate_overrides(Args())
        # Enable is processed first, then disable overrides it
        assert reg.is_enabled("cli_test3") is False

    def test_apply_feature_gate_overrides_unknown_feature(self):
        """Overriding unknown features doesn't crash."""
        from clawcodex_ext.cli.dispatch import _apply_feature_gate_overrides

        class Args:
            enable_feature = ["unknown_xyz"]
            disable_feature = []

        _apply_feature_gate_overrides(Args())  # Should not raise

    def test_apply_feature_gate_overrides_no_args(self):
        """No feature args is a no-op."""
        from clawcodex_ext.cli.dispatch import _apply_feature_gate_overrides
        from clawcodex_ext.feature_gate import get_registry, FeatureFlag

        reg = get_registry()
        reg.register(FeatureFlag("no_op", default=True))

        class Args:
            enable_feature = None
            disable_feature = None

        _apply_feature_gate_overrides(Args())
        assert reg.is_enabled("no_op") is True
