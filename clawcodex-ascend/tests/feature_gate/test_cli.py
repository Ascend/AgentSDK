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

"""Tests for feature-gate CLI subcommand (clawcodex_ext.feature_gate.cli)."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from clawcodex_ext.feature_gate import get_registry, reset_registry
from clawcodex_ext.feature_gate.cli import (
    run_feature_command,
    _handle_list,
    _handle_get,
    _handle_set,
    _handle_reload,
    _handle_reset,
)
from clawcodex_ext.feature_gate.types import FeatureFlag


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestFeatureCommandDispatch:
    """Tests for the main feature subcommand dispatcher."""

    def test_no_args_shows_usage(self):
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = run_feature_command([])
        assert rc == 2
        stderr_text = stderr_buf.getvalue()
        assert "usage" in stderr_text.lower() or "feature" in stderr_text.lower()

    def test_unknown_subcommand(self):
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = run_feature_command(["bogus"])
        assert rc == 2

    def test_list_subcommand(self):
        """'feature list' should succeed even with no features."""
        rc = run_feature_command(["list"])
        assert rc == 0

    def test_get_unknown_feature(self):
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = run_feature_command(["get", "nonexistent"])
        assert rc == 1
        assert "unknown" in stderr_buf.getvalue().lower()

    def test_set_unknown_feature(self):
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = run_feature_command(["set", "nonexistent", "--on"])
        assert rc == 1
        assert "unknown" in stderr_buf.getvalue().lower()


class TestHandleList:
    """Tests for the list sub-handler."""

    def test_list_shows_registered_features(self):
        reg = get_registry()
        reg.register(FeatureFlag("custom_feat", default=True))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_list([])
        assert rc == 0
        output = stdout_buf.getvalue()
        assert "custom_feat" in output

    def test_list_filtered_enabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("en", default=True))
        reg.register(FeatureFlag("dis", default=False))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_list(["--enabled"])
        assert rc == 0
        output = stdout_buf.getvalue()
        assert "en" in output
        assert "dis" not in output

    def test_list_filtered_disabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("en", default=True))
        reg.register(FeatureFlag("dis", default=False))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_list(["--disabled"])
        assert rc == 0
        output = stdout_buf.getvalue()
        assert "dis" in output
        assert "en" not in output

    def test_list_json_output(self):
        import json

        reg = get_registry()
        reg.register(FeatureFlag("json_test", default=True))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_list(["--json"])
        assert rc == 0
        data = json.loads(stdout_buf.getvalue())
        names = [entry["name"] for entry in data]
        assert "json_test" in names


class TestHandleGet:
    """Tests for the get sub-handler."""

    def test_get_enabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("gf", default=True))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_get(["gf"])
        assert rc == 0
        assert "enabled" in stdout_buf.getvalue()

    def test_get_disabled(self):
        reg = get_registry()
        reg.register(FeatureFlag("gf_off", default=False))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_get(["gf_off"])
        assert rc == 0
        assert "disabled" in stdout_buf.getvalue()

    def test_get_shows_deps(self):
        reg = get_registry()
        reg.register(FeatureFlag("dep_base", default=True))
        reg.register(FeatureFlag("dep_child", default=False, deps=["dep_base"]))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_get(["dep_child"])
        assert rc == 0
        output = stdout_buf.getvalue()
        assert "deps" in output.lower()


class TestHandleSet:
    """Tests for the set sub-handler."""

    def test_set_on(self):
        reg = get_registry()
        reg.register(FeatureFlag("sf", default=False))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_set(["sf", "--on"])
        assert rc == 0
        assert reg.is_enabled("sf") is True

    def test_set_off(self):
        reg = get_registry()
        reg.register(FeatureFlag("sf_off", default=True))
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_set(["sf_off", "--off"])
        assert rc == 0
        assert reg.is_enabled("sf_off") is False

    def test_set_on_with_missing_dep_fails(self):
        reg = get_registry()
        reg.register(FeatureFlag("dep_req", default=False))
        reg.register(FeatureFlag("dep_child", default=False, deps=["dep_req"]))
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = _handle_set(["dep_child", "--on"])
        assert rc == 1
        assert "missing" in stderr_buf.getvalue().lower() or "deps" in stderr_buf.getvalue().lower()

    def test_set_on_with_mutex_conflict_fails(self):
        reg = get_registry()
        reg.register(FeatureFlag("mx_a", default=True))
        reg.register(FeatureFlag("mx_b", default=False, mutex_with=["mx_a"]))
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = _handle_set(["mx_b", "--on"])
        assert rc == 1
        assert "conflict" in stderr_buf.getvalue().lower()


class TestHandleReload:
    """Tests for the reload sub-handler."""

    def test_reload(self):
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_reload()
        assert rc == 0
        assert "reloaded" in stdout_buf.getvalue().lower()


class TestHandleReset:
    """Tests for the reset sub-handler."""

    def test_reset_clears_overrides(self):
        reg = get_registry()
        reg.register(FeatureFlag("rf", default=False))
        reg.enable_feature("rf")
        assert reg.is_enabled("rf") is True
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = _handle_reset()
        assert rc == 0
        assert reg.is_enabled("rf") is False
