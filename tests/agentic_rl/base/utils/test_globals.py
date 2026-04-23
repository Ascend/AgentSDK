#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from agentic_rl.base.utils.globals import (
    GCP_PROJECT_ID,
    GCP_LOCATION,
    GEMINI_MODEL,
    OAI_RM_MODEL,
    THOUGHT_DELIMITER_START,
    THOUGHT_DELIMITER_END,
    SWEBENCH_DATASET_NAME,
    MAX_WORKERS,
    FORCE_REBUILD,
    CACHE_LEVEL,
    CLEAN,
    OPEN_FILE_LIMIT,
    TIMEOUT,
    NAMESPACE,
    REWRITE_REPORTS,
    SPLIT,
    INSTANCE_IMAGE_TAG,
    REPORT_DIR,
    ROLLOUT_WEIGHTS_PREFIX,
    TRAIN_CLUSTER,
    ROLLOUT_CLUSTER,
    set_cluster_mode,
    get_cluster_mode,
    is_pd_separate
)


class TestGlobals:
    def test_constants(self):
        """Test that all constants are correctly defined."""
        # Gemini Vertex AI Config
        assert GCP_PROJECT_ID == "cloud-llm-test"
        assert GCP_LOCATION == "us-central1"
        assert GEMINI_MODEL == "gemini-1.5-pro-002"
        assert OAI_RM_MODEL == "gpt-4o-mini"
        
        # Reward function constants
        assert THOUGHT_DELIMITER_START == "<think>"
        assert THOUGHT_DELIMITER_END == "</think>"
        
        # SWEBench Harness Config
        assert SWEBENCH_DATASET_NAME == "princeton-nlp/SWE-bench_Verified"
        assert MAX_WORKERS == 4
        assert FORCE_REBUILD is False
        assert CACHE_LEVEL == "None"
        assert CLEAN is False
        assert OPEN_FILE_LIMIT == 4096
        assert TIMEOUT == 1800
        assert NAMESPACE is None
        assert REWRITE_REPORTS is False
        assert SPLIT == "test"
        assert INSTANCE_IMAGE_TAG == "latest"
        assert REPORT_DIR == "../.."
        
        # Other constants
        assert ROLLOUT_WEIGHTS_PREFIX == "/rollout"
        assert TRAIN_CLUSTER == "train"
        assert ROLLOUT_CLUSTER == "rollout"

    def test_set_and_get_cluster_mode(self):
        """Test that cluster mode can be set and retrieved correctly."""
        # Save original environment variable value if it exists
        original_mode = os.environ.get("CLUSTER_MODE")
        
        try:
            # Test setting and getting cluster mode
            test_mode = "test_mode"
            set_cluster_mode(test_mode)
            assert get_cluster_mode() == test_mode
            assert os.environ["CLUSTER_MODE"] == test_mode
            
            # Test another mode
            another_mode = "another_mode"
            set_cluster_mode(another_mode)
            assert get_cluster_mode() == another_mode
            assert os.environ["CLUSTER_MODE"] == another_mode
        finally:
            # Restore original environment variable
            if original_mode is not None:
                os.environ["CLUSTER_MODE"] = original_mode
            elif "CLUSTER_MODE" in os.environ:
                del os.environ["CLUSTER_MODE"]

    def test_is_pd_separate(self):
        """Test that is_pd_separate function works correctly with different environment variable values."""
        # Save original environment variable value if it exists
        original_use_pd = os.environ.get("USE_PD")
        
        try:
            # Test default value (USE_PD not set)
            if "USE_PD" in os.environ:
                del os.environ["USE_PD"]
            assert is_pd_separate() is False
            
            # Test USE_PD=0
            os.environ["USE_PD"] = "0"
            assert is_pd_separate() is False
            
            # Test USE_PD=1
            os.environ["USE_PD"] = "1"
            assert is_pd_separate() is True
            
            # Test USE_PD=other values (should be treated as True)
            os.environ["USE_PD"] = "2"
            assert is_pd_separate() is True  # Note: non-zero integers are True when converted to bool
            
            os.environ["USE_PD"] = "true"
            with pytest.raises(ValueError):
                is_pd_separate()  # Non-numeric values raise ValueError
        finally:
            # Restore original environment variable
            if original_use_pd is not None:
                os.environ["USE_PD"] = original_use_pd
            elif "USE_PD" in os.environ:
                del os.environ["USE_PD"]

    def test_get_cluster_mode_missing(self):
        """Test that get_cluster_mode raises KeyError when CLUSTER_MODE is not set."""
        # Save original environment variable value if it exists
        original_mode = os.environ.get("CLUSTER_MODE")
        
        try:
            # Ensure environment variable doesn't exist
            if "CLUSTER_MODE" in os.environ:
                del os.environ["CLUSTER_MODE"]
            
            # Calling get_cluster_mode should raise KeyError
            with pytest.raises(KeyError):
                get_cluster_mode()
        finally:
            # Restore original environment variable
            if original_mode is not None:
                os.environ["CLUSTER_MODE"] = original_mode
            elif "CLUSTER_MODE" in os.environ:
                del os.environ["CLUSTER_MODE"]