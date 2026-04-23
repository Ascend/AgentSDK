#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
import pytest
import unittest.mock as mock

from agentic_rl.base.utils.load_object_by_path import load_object_by_path


class TestLoadObjectByPath:
    def test_load_standard_library_function(self):
        """Test loading a function from standard library."""
        # Test loading os.path.join function
        join_func = load_object_by_path("os.path.join")
        assert join_func is not None
        # Verify function works correctly
        # Use os.path.join to construct expected path, compatible with different OS
        expected_path = os.path.join("a", "b", "c")
        assert join_func("a", "b", "c") == expected_path

    def test_load_standard_library_class(self):
        """Test loading a class from standard library."""
        # Test loading datetime.datetime class
        datetime_class = load_object_by_path("datetime.datetime")
        assert datetime_class is not None
        # Verify class can be instantiated
        now = datetime_class.now()
        assert hasattr(now, "year")
        assert hasattr(now, "month")
        assert hasattr(now, "day")

    def test_load_project_function(self):
        """Test loading a function from the project."""
        # Test loading load_object_by_path function itself
        loaded_func = load_object_by_path("agentic_rl.base.utils.load_object_by_path.load_object_by_path")
        assert loaded_func is not None
        assert loaded_func is load_object_by_path

    def test_load_project_constant(self):
        """Test loading a constant from the project."""
        # Test loading constant from globals.py
        gcp_project_id = load_object_by_path("agentic_rl.base.utils.globals.GCP_PROJECT_ID")
        assert gcp_project_id == "cloud-llm-test"

    def test_invalid_path_format(self):
        """Test loading with invalid path format."""
        # Test path without dots
        with pytest.raises((ImportError, ValueError)):
            load_object_by_path("invalid_path")

    def test_nonexistent_module(self):
        """Test loading from a nonexistent module."""
        # Test non-existent module
        with pytest.raises(ImportError):
            load_object_by_path("nonexistent.module.ClassName")

    def test_nonexistent_object(self):
        """Test loading a nonexistent object from an existing module."""
        # Test non-existent object from existing module
        with pytest.raises(ImportError):
            load_object_by_path("os.nonexistent_function")

    @mock.patch("importlib.import_module")
    def test_import_error_propagation(self, mock_import_module):
        """Test that import errors are properly propagated."""
        # Mock importlib.import_module to raise ImportError
        mock_import_module.side_effect = ImportError("Mock import error")
        
        with pytest.raises(ImportError) as excinfo:
            load_object_by_path("os.path.join")
        
        assert "Failed to load object 'os.path.join'" in str(excinfo.value)
        assert "Mock import error" in str(excinfo.value)

    @mock.patch("importlib.import_module")
    def test_attribute_error_propagation(self, mock_import_module):
        """Test that attribute errors are properly propagated."""
        # Mock successful module import but object doesn't exist
        # Use spec=[] to create a Mock object with no attributes
        mock_module = mock.Mock(spec=[])
        mock_import_module.return_value = mock_module
        
        with pytest.raises(ImportError) as excinfo:
            load_object_by_path("os.nonexistent_function")
        
        assert "Failed to load object 'os.nonexistent_function'" in str(excinfo.value)

    def test_load_nested_module(self):
        """Test loading from a deeply nested module."""
        # Test loading a deeply nested standard library object
        chain_map = load_object_by_path("collections.ChainMap")
        assert chain_map is not None
        # Verify it works correctly
        cm = chain_map({"a": 1}, {"b": 2})
        assert cm["a"] == 1
        assert cm["b"] == 2