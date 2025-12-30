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


class TestCaMemAllocatorSleep:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "torch": MagicMock(),
            "acl": MagicMock(),
            "acl.rt": MagicMock(),
            "vllm": MagicMock(),
            "vllm.utils": MagicMock(),
            "vllm_ascend": MagicMock(),
            "vllm_ascend.device_allocator": MagicMock(),
            "vllm_ascend.device_allocator.camem": MagicMock()
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch("torch.empty"), \
                patch("acl.rt.memcpy"), \
                patch("vllm.utils.is_pin_memory_available"), \
                patch("vllm_ascend.device_allocator.camem.unmap_and_release"):
            yield

    def test_sleep_failed_with_input(self, patch_target):
        offload_tags = []
        with pytest.raises(TypeError, match="offload_tags must be a tuple"):
            from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep
            ca_mem_allocator_sleep(MagicMock(), offload_tags)

        offload_tags = (1, 2, 3)
        with pytest.raises(TypeError, match="offload_tags must be a tuple of strings"):
            from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep
            ca_mem_allocator_sleep(MagicMock(), offload_tags)

    def test_sleep_failed_with_attribute(self, patch_target):
        offload_tags = "kv_caches"

        ca_mem_allocator = MagicMock()
        ca_mem_allocator.pointer_to_data = "invalid"

        with pytest.raises(AttributeError, match="CaMemAllocator not initialized, pointer_to_data is not valid."):
            from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep
            ca_mem_allocator_sleep(ca_mem_allocator, offload_tags)

    def test_sleep_failed_with_operation(self, patch_target):
        offload_tags = "kv_caches"

        ca_mem_allocator = MagicMock()

        fake_data = MagicMock()
        fake_data.tag = "kv_caches"
        fake_data.handle = [1, 100]
        ca_mem_allocator.pointer_to_data = {1: fake_data}

        with patch("agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep.memcpy") as mock_memcpy:
            from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep

            mock_memcpy.side_effect = AttributeError("test")
            with pytest.raises(AttributeError):
                ca_mem_allocator_sleep(ca_mem_allocator, offload_tags)

            mock_memcpy.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError):
                ca_mem_allocator_sleep(ca_mem_allocator, offload_tags)

            mock_memcpy.side_effect = ValueError("test")
            with pytest.raises(RuntimeError):
                ca_mem_allocator_sleep(ca_mem_allocator, offload_tags)

    def test_sleep_success(self, patch_target):
        offload_tags = "kv_caches"

        ca_mem_allocator = MagicMock()

        fake_data = MagicMock()
        fake_data.tag = "kv_caches"
        fake_data.handle = [1, 100]
        ca_mem_allocator.pointer_to_data = {1: fake_data}

        from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep
        ca_mem_allocator_sleep(ca_mem_allocator, offload_tags)
