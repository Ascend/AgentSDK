#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# 
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# 
#        http://license.coscl.org.cn/MulanPSL2
# 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import pytest

from agentic_rl.base.utils.work_mode import set_work_mode, get_work_mode


class TestSetWorkMode:
    """Test cases for set_work_mode function."""

    def test_set_work_mode_standalone(self):
        """Test setting work mode to 'standalone'."""
        set_work_mode("standalone")
        assert os.environ.get("work_mode") == "standalone"

    def test_set_work_mode_hybrid(self):
        """Test setting work mode to 'hybrid'."""
        set_work_mode("hybrid")
        assert os.environ.get("work_mode") == "hybrid"

    def test_set_work_mode_distributed(self):
        """Test setting work mode to 'distributed'."""
        set_work_mode("distributed")
        assert os.environ.get("work_mode") == "distributed"

    def test_set_work_mode_empty_string(self):
        """Test setting work mode to empty string."""
        set_work_mode("")
        assert os.environ.get("work_mode") == ""

    def test_set_work_mode_overwrites_previous(self):
        """Test that setting work mode overwrites previous value."""
        set_work_mode("standalone")
        assert os.environ.get("work_mode") == "standalone"
        set_work_mode("hybrid")
        assert os.environ.get("work_mode") == "hybrid"


class TestGetWorkMode:
    """Test cases for get_work_mode function."""

    def test_get_work_mode_default(self):
        """Test getting default work mode when not set."""
        # Ensure work_mode is not set
        if "work_mode" in os.environ:
            del os.environ["work_mode"]
        assert get_work_mode() == "hybrid"

    def test_get_work_mode_after_set(self):
        """Test getting work mode after setting it."""
        set_work_mode("standalone")
        assert get_work_mode() == "standalone"

    def test_get_work_mode_returns_current_value(self):
        """Test that get_work_mode returns the current environment value."""
        os.environ["work_mode"] = "distributed"
        assert get_work_mode() == "distributed"

    def test_get_work_mode_returns_string(self):
        """Test that get_work_mode always returns a string."""
        result = get_work_mode()
        assert isinstance(result, str)

    def test_get_work_mode_with_empty_env(self):
        """Test get_work_mode when environment variable is empty string."""
        os.environ["work_mode"] = ""
        assert get_work_mode() == ""


class TestWorkModeIntegration:
    """Integration tests for work_mode functions."""

    def test_set_then_get(self):
        """Test setting and then getting work mode."""
        set_work_mode("standalone")
        assert get_work_mode() == "standalone"

    def test_multiple_set_get_cycles(self):
        """Test multiple set/get cycles."""
        modes = ["standalone", "hybrid", "distributed"]
        for mode in modes:
            set_work_mode(mode)
            assert get_work_mode() == mode

    def test_default_after_delete(self):
        """Test that default is returned after deleting environment variable."""
        set_work_mode("standalone")
        del os.environ["work_mode"]
        assert get_work_mode() == "hybrid"


@pytest.fixture(autouse=True)
def cleanup_work_mode():
    """Fixture to clean up work_mode environment variable after each test."""
    original_value = os.environ.get("work_mode")
    yield
    if original_value is not None:
        os.environ["work_mode"] = original_value
    elif "work_mode" in os.environ:
        del os.environ["work_mode"]
