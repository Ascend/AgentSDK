#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestWorkerV1Sleep:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "vllm_ascend": MagicMock(),
            "vllm_ascend.device_allocator": MagicMock(),
            "vllm_ascend.device_allocator.camem": MagicMock(),
            "vllm_ascend.platform": MagicMock()
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch("vllm_ascend.platform.NPUPlatform.mem_get_info") as mock_mem_get_info:
            mock_mem_get_info.return_value = 100, 100
            yield

    def test_worker_v1_sleep_failed_with_params(self, patch_target):
        worker_v1 = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep
        with pytest.raises(ValueError, match="sleep level should be 0 or 1 or 2"):
            worker_v1_sleep(worker_v1, 3)

    def test_worker_v1_sleep_failed_with_set_device(self, patch_target):
        worker_v1 = MagicMock()
        worker_v1.device = MagicMock()

        with patch("agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep."
                   "NPUPlatform.set_device") as mock_set_device:
            from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep

            mock_set_device.side_effect = AttributeError("test")
            with pytest.raises(AttributeError, match="set device failed with attribute"):
                worker_v1_sleep(worker_v1, 1)

            mock_set_device.side_effect = ValueError("test")
            with pytest.raises(RuntimeError, match="Unexpected error occurred when set device"):
                worker_v1_sleep(worker_v1, 1)

    def test_worker_v1_sleep_failed_with_allocator_sleep(self, patch_target):
        worker_v1 = MagicMock()
        worker_v1.device = MagicMock()

        with patch("agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep."
                   "CaMemAllocator") as mock_allocator_class:
            allocator_instance = MagicMock()
            mock_allocator_class.get_instance.return_value = allocator_instance

            from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep

            allocator_instance.sleep.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="sleep failed with runtime error"):
                worker_v1_sleep(worker_v1, 1)

            allocator_instance.sleep.side_effect = AttributeError("test")
            with pytest.raises(RuntimeError, match="Unexpected error occurred when sleeping"):
                worker_v1_sleep(worker_v1, 2)

            allocator_instance.sleep.side_effect = ValueError("test")
            with pytest.raises(RuntimeError, match="Unexpected error occurred when sleeping"):
                worker_v1_sleep(worker_v1, 0)

    def test_worker_v1_sleep_failed_with_mem_get_info(self, patch_target):
        worker_v1 = MagicMock()
        worker_v1.device = MagicMock()

        with patch("agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep."
                   "NPUPlatform.mem_get_info") as mock_mem_get_info:
            from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep

            mock_mem_get_info.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="get memory info failed with runtime error"):
                worker_v1_sleep(worker_v1, 1)

            mock_mem_get_info.side_effect = ValueError("test")
            with pytest.raises(RuntimeError, match="Unexpected error occurred when get memory info"):
                worker_v1_sleep(worker_v1, 1)

            def make_get_info():
                cnt = 0

                def get_info():
                    nonlocal cnt
                    if cnt % 2 == 0:
                        cnt = cnt + 1
                        return 100, 100
                    else:
                        cnt = cnt + 1
                        return 10, 10

                return get_info

            mock_mem_get_info.side_effect = make_get_info()
            with pytest.raises(RuntimeError, match="Memory usage increased after sleeping"):
                worker_v1_sleep(worker_v1, 1)

    def test_worker_v1_sleep_success(self, patch_target):
        worker_v1 = MagicMock()
        from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep
        worker_v1_sleep(worker_v1, 1)
