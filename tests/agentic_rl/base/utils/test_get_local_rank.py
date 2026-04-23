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
import unittest.mock as mock

from agentic_rl.base.utils.get_local_rank import get_local_rank


class TestGetLocalRank:
    def test_local_rank_from_environment(self):
        """Test that local rank is correctly retrieved from environment variable."""
        # Set environment variable
        os.environ["LOCAL_RANK"] = "1"

        try:
            # Call function
            rank = get_local_rank()

            # Verify result
            assert rank == 1
        finally:
            # Clean up environment variable
            del os.environ["LOCAL_RANK"]

    @mock.patch("agentic_rl.base.utils.get_local_rank.ray")
    def test_local_rank_from_ray(self, mock_ray):
        """Test that local rank is correctly retrieved from ray runtime context."""
        # Mock ray runtime context
        mock_context = mock.Mock()
        mock_context.get_accelerator_ids.return_value = {"NPU": ["2"]}
        mock_ray.get_runtime_context.return_value = mock_context

        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]

        rank = get_local_rank()

        assert rank == 2
        assert "LOCAL_RANK" in os.environ
        assert os.environ["LOCAL_RANK"] == "2"

        del os.environ["LOCAL_RANK"]

    @mock.patch("agentic_rl.base.utils.get_local_rank.ray")
    def test_ray_exception_fallback(self, mock_ray):
        """Test that function falls back to default when ray raises an exception."""
        mock_ray.get_runtime_context.side_effect = Exception("Ray error")

        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]

        rank = get_local_rank()

        assert rank == 0

    @mock.patch("agentic_rl.base.utils.get_local_rank.ray")
    def test_default_rank(self, mock_ray):
        """Test that default rank 0 is returned when no context is available."""
        mock_context = mock.Mock()
        mock_context.get_accelerator_ids.return_value = {"CPU": ["0"]}
        mock_ray.get_runtime_context.return_value = mock_context

        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]

        rank = get_local_rank()

        assert rank == 0

    @mock.patch("agentic_rl.base.utils.get_local_rank.ray")
    def test_with_custom_logger_name(self, mock_ray):
        """Test that function works with custom logger name."""
        mock_ray.get_runtime_context.side_effect = Exception("Ray error")

        if "LOCAL_RANK" in os.environ:
            del os.environ["LOCAL_RANK"]

        rank = get_local_rank(logger_name="test_logger")

        assert rank == 0
