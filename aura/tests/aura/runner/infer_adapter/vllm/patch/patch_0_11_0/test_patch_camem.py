#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import sys
import types
from unittest.mock import MagicMock, patch
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_camem
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_camem_env():
    # ---- Fake acl.rt ----
    fake_acl = types.ModuleType("acl")
    fake_acl.__path__ = []
    fake_acl_rt = types.ModuleType("acl.rt")
    fake_acl_rt.memcpy = MagicMock()

    # ---- Fake vllm.utils ----
    fake_vllm_utils = types.ModuleType("vllm.utils")
    fake_vllm_utils.is_pin_memory_available = MagicMock(return_value=True)

    # ---- Fake vllm_ascend.device_allocator.camem ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_da = types.ModuleType("vllm_ascend.device_allocator")
    fake_vllm_ascend_da.__path__ = []
    fake_vllm_ascend_camem = types.ModuleType("vllm_ascend.device_allocator.camem")

    class CaMemAllocator:
        default_tag = "default"

    fake_vllm_ascend_camem.CaMemAllocator = CaMemAllocator
    fake_vllm_ascend_camem.unmap_and_release = MagicMock()

    # ---- aura packages ----
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = [os.path.join(base_path, "runner/infer_adapter")]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm/patch")]
    fake_0_11_0_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_11_0")
    fake_0_11_0_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_11_0")]

    all_fakes = {
        "acl": fake_acl,
        "acl.rt": fake_acl_rt,
        "vllm.utils": fake_vllm_utils,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.device_allocator": fake_vllm_ascend_da,
        "vllm_ascend.device_allocator.camem": fake_vllm_ascend_camem,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
    }
    for name, mod in all_fakes.items():
        sys.modules[name] = mod

    target_module = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem"
    if target_module in sys.modules:
        del sys.modules[target_module]

    yield {
        "memcpy": fake_acl_rt.memcpy,
        "is_pin_memory_available": fake_vllm_utils.is_pin_memory_available,
        "CaMemAllocator": fake_vllm_ascend_camem.CaMemAllocator,
        "unmap_and_release": fake_vllm_ascend_camem.unmap_and_release,
    }

    # Cleanup
    for name in list(all_fakes.keys()):
        if name in sys.modules:
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Helper: data entry as a plain class to guarantee real string tags
# ---------------------------------------------------------------------------
class DataEntry:
    def __init__(self, tag: str, handle_size: int):
        self.tag = tag
        self.handle = (MagicMock(), handle_size)


def make_allocator(entries=None):
    alloc = MagicMock()
    alloc.pointer_to_data = entries if entries else {}
    return alloc


def make_data_entry(tag="default", handle_size=128):
    return DataEntry(tag, handle_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCamemSleep:
    def test_offload_tags_none(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem

        alloc = make_allocator({1: make_data_entry("default"), 2: make_data_entry("other")})
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.data_ptr.return_value = 0x1000

        with patch("torch.empty", return_value=mock_tensor) as mock_empty, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            camem.camem_sleep(alloc, None)

        mock_empty.assert_called_once()
        fake_camem_env["memcpy"].assert_called_once()
        fake_camem_env["unmap_and_release"].assert_called_once_with(alloc.pointer_to_data[1].handle)
        mock_empty_cache.assert_called_once()

    def test_offload_tags_string(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem

        alloc = make_allocator({10: make_data_entry("custom"), 20: make_data_entry("default")})
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.data_ptr.return_value = 0x2000

        with patch("torch.empty", return_value=mock_tensor) as mock_empty, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            camem.camem_sleep(alloc, "custom")

        mock_empty.assert_called_once()
        fake_camem_env["memcpy"].assert_called_once()
        fake_camem_env["unmap_and_release"].assert_called_once_with(alloc.pointer_to_data[10].handle)
        mock_empty_cache.assert_called_once()

    def test_offload_tags_tuple(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem

        alloc = make_allocator({1: make_data_entry("a"), 2: make_data_entry("b"), 3: make_data_entry("c")})
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.data_ptr.return_value = 0x3000

        with patch("torch.empty", return_value=mock_tensor) as mock_empty, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            camem.camem_sleep(alloc, ("a", "c"))

        assert mock_empty.call_count == 2
        assert fake_camem_env["memcpy"].call_count == 2
        assert fake_camem_env["unmap_and_release"].call_count == 2
        mock_empty_cache.assert_called_once()

    def test_offload_tags_invalid_type(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem
        alloc = make_allocator()

        with pytest.raises(TypeError, match="offload_tags must be a tuple"):
            camem.camem_sleep(alloc, 123)

    def test_offload_tags_tuple_none_behavior(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem

        alloc = make_allocator({1: make_data_entry("other")})
        with patch("torch.empty") as mock_empty, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            camem.camem_sleep(alloc, None)

        mock_empty.assert_not_called()
        fake_camem_env["memcpy"].assert_not_called()
        fake_camem_env["unmap_and_release"].assert_not_called()
        mock_empty_cache.assert_called_once()

    def test_memcpy_parameters(self, fake_camem_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_camem as camem

        alloc = make_allocator({1: make_data_entry("default", handle_size=256)})
        mock_tensor = MagicMock(spec=torch.Tensor)
        cpu_ptr = 0x5000
        mock_tensor.data_ptr.return_value = cpu_ptr

        with patch("torch.empty", return_value=mock_tensor) as mock_empty, \
             patch("torch.cuda.empty_cache"):
            camem.camem_sleep(alloc, None)

        fake_camem_env["memcpy"].assert_called_once_with(
            cpu_ptr,                    # dest
            cpu_ptr + 256 * 2,          # dest_max
            1,                          # ptr (dictionary key)
            256,                        # size_in_bytes
            2,                          # ACL_MEMCPY_DEVICE_TO_HOST
        )
