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
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import sys
import types
import json
import unittest
import importlib
from unittest.mock import patch, MagicMock


# =============================================================================
# Fake Torch (DO NOT import real torch, otherwise _C docstring issues)
# =============================================================================

class FakeDType:
    """Fake torch.dtype with a name attribute."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


class FakeDevice:
    """Fake torch.device that mimics device string parsing."""
    def __init__(self, dev, index=None):
        if isinstance(dev, FakeDevice):
            self.type = dev.type
            self.index = dev.index
        else:
            if isinstance(dev, str):
                if ":" in dev:
                    self.type, idx = dev.split(":")
                    self.index = int(idx)
                else:
                    self.type = dev
                    self.index = index
            else:
                self.type = str(dev)
                self.index = index

    def __repr__(self):
        if self.index is None:
            return f"torch.device('{self.type}')"
        return f"torch.device('{self.type}:{self.index}')"


class FakeStorage:
    """Fake tensor storage with size tracking."""
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size

    def resize_(self, new_size):
        self._size = new_size


class FakeTensor:
    """
    Minimal fake tensor that supports shape, dtype, device, contiguity,
    and basic methods needed for tests.
    """
    def __init__(self, shape, dtype=None, device="cpu", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=str(device).split(":")[0])
        self._contiguous = contiguous
        self._storage = FakeStorage(1)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        if self.dtype and self.dtype.name in ("float16", "bfloat16"):
            return 2
        if self.dtype and self.dtype.name in ("float32",):
            return 4
        return 4

    def to(self, dtype, copy=False):
        return FakeTensor(self.shape, dtype=dtype, device=self.device.type, contiguous=True)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return FakeTensor(self.shape, dtype=self.dtype, device=self.device.type, contiguous=True)

    def is_floating_point(self):
        return True

    def permute(self, *dims):
        new_shape = [self.shape[d] for d in dims]
        return FakeTensor(new_shape, dtype=self.dtype, device=self.device.type, contiguous=False)

    def copy_(self, other, non_blocking=False):
        return self

    def data_ptr(self):
        return 123456

    def storage(self):
        return self._storage


class FakeNoGrad:
    """
    Fake torch.no_grad() that works both as context manager and decorator.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, fn):
        return fn


class FakeCuda:
    """Fake torch.cuda module."""
    def __init__(self):
        self._avail = False

    def is_available(self):
        return self._avail

    def current_device(self):
        return 0

    def empty_cache(self):
        return None

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0


class FakeNpu:
    """Fake torch.npu module."""
    def __init__(self):
        self._cur = 0
        self._avail = True

    def is_available(self):
        return self._avail

    def current_device(self):
        return self._cur

    def set_device(self, dev):
        return None

    def synchronize(self, dev=None):
        return None

    class Stream:
        def __init__(self, device=None):
            self.device = device

    def stream(self, stream_obj):
        return FakeNoGrad()


class FakeDistributed:
    """Fake torch.distributed module."""
    def __init__(self):
        self.broadcast = MagicMock(name="broadcast")


def _build_fake_torch():
    """Assemble a complete fake torch module with all needed submodules."""
    fake_torch = types.ModuleType("torch")

    fake_torch.dtype = FakeDType
    fake_torch.Tensor = FakeTensor
    fake_torch.device = FakeDevice

    fake_torch.float16 = FakeDType("float16")
    fake_torch.bfloat16 = FakeDType("bfloat16")
    fake_torch.float32 = FakeDType("float32")

    def no_grad():
        return FakeNoGrad()

    fake_torch.no_grad = no_grad

    def empty_like(t, device=None):
        return FakeTensor(t.shape, dtype=t.dtype, device=device or t.device.type)

    def empty(shape, dtype=None, device="cpu"):
        return FakeTensor(shape, dtype=dtype, device=device)

    def ones(shape, dtype=None, device="cpu"):
        return FakeTensor(shape, dtype=dtype, device=device)

    def zeros(shape, dtype=None, device="cpu"):
        return FakeTensor(shape, dtype=dtype, device=device)

    def tensor(x):
        return FakeTensor((0,), dtype=fake_torch.float32, device="cpu")

    fake_torch.empty_like = empty_like
    fake_torch.empty = empty
    fake_torch.ones = ones
    fake_torch.zeros = zeros
    fake_torch.tensor = tensor

    fake_torch.cuda = FakeCuda()
    fake_torch.npu = FakeNpu()
    fake_torch.distributed = FakeDistributed()

    return fake_torch


# =============================================================================
# External dependency fakes
# =============================================================================

def _build_fake_modules():
    """
    Create fake modules for external dependencies like acl, safetensors,
    vllm_execute_stat, and verl utilities.
    """
    fake_acl = types.ModuleType("acl")
    fake_acl_rt = types.ModuleType("acl.rt")
    fake_acl_rt.memcpy = MagicMock(name="memcpy")
    fake_acl.rt = fake_acl_rt

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.safe_open = MagicMock(name="safe_open")
    fake_safetensors.torch = fake_safetensors_torch

    fake_stat_mod = types.ModuleType(
        "agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat"
    )
    fake_stat_obj = MagicMock(name="vllm_output_statics")
    fake_stat_obj.write_stats_tofile = MagicMock()
    fake_stat_obj.clear = MagicMock()
    fake_stat_mod.vllm_output_statics = fake_stat_obj

    fake_verl = types.ModuleType("verl")
    fake_verl_utils = types.ModuleType("verl.utils")
    fake_verl_utils_device = types.ModuleType("verl.utils.device")
    fake_verl_utils_device.get_torch_device = MagicMock(name="get_torch_device")

    fake_verl_utils_vllm = types.ModuleType("verl.utils.vllm")
    fake_verl_utils_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_verl_utils_vllm_patch.patch_vllm_moe_model_weight_loader = MagicMock(
        name="patch_vllm_moe_model_weight_loader"
    )

    return {
        "torch": _build_fake_torch(),
        "torch.distributed": _build_fake_torch().distributed,
        "acl": fake_acl,
        "acl.rt": fake_acl_rt,
        "safetensors": fake_safetensors,
        "safetensors.torch": fake_safetensors_torch,
        "agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat": fake_stat_mod,
        "verl": fake_verl,
        "verl.utils": fake_verl_utils,
        "verl.utils.device": fake_verl_utils_device,
        "verl.utils.vllm": fake_verl_utils_vllm,
        "verl.utils.vllm.patch": fake_verl_utils_vllm_patch,
    }


def _import_target_module():
    """Reload and import the target module under test."""
    mod_name = "agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


class FakeSafeOpenCtx:
    """Context manager to fake safetensors.safe_open."""
    def __init__(self, tensor_map):
        self.tensor_map = tensor_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self.tensor_map.keys())

    def get_tensor(self, k):
        return self.tensor_map[k]


# =============================================================================
# Tests
# =============================================================================

class TestCustomWorkerExtensions(unittest.TestCase):
    """Unit tests for custom_worker_extensions module functions and class."""

    def setUp(self):
        """Install fake modules and reload the target module."""
        self.mod_patch = patch.dict(sys.modules, _build_fake_modules())
        self.mod_patch.start()
        self.target_mod = _import_target_module()

    def tearDown(self):
        """Stop module patching."""
        self.mod_patch.stop()

    def test_bytes_of(self):
        """_bytes_of should return tensor size in bytes (numel * element_size)."""
        torch = sys.modules["torch"]
        t = torch.zeros((2, 3), dtype=torch.float32)
        out = self.target_mod._bytes_of(t)
        self.assertEqual(out, 2 * 3 * 4)

    def test_ensure_dtype_config_cast_and_contiguous(self):
        """Ensure dtype conversion and contiguity are applied."""
        torch = sys.modules["torch"]
        t = torch.zeros((2, 3), dtype=torch.float32)
        t._contiguous = False
        out = self.target_mod._ensure_dtype_config([t], torch.float16)
        self.assertEqual(out[0].dtype.name, "float16")
        self.assertTrue(out[0].is_contiguous())

    def test_ensure_dtype_config_no_cast_still_contiguous(self):
        """Even if dtype matches, contiguity is enforced."""
        torch = sys.modules["torch"]
        t = torch.zeros((2, 3), dtype=torch.float16)
        t._contiguous = False
        out = self.target_mod._ensure_dtype_config([t], torch.float16)
        self.assertEqual(out[0].dtype.name, "float16")
        self.assertTrue(out[0].is_contiguous())

    def test_list_rank_files_filters_and_sorts(self):
        """_list_rank_files should filter by rank prefix and sort numerically."""
        with patch("os.scandir") as mock_scandir:
            entry1 = MagicMock()
            entry1.is_file.return_value = True
            entry1.name = "rank_0_002.safetensors"
            entry1.path = "/tmp/rank_0_002.safetensors"

            entry2 = MagicMock()
            entry2.is_file.return_value = True
            entry2.name = "rank_0_001.safetensors"
            entry2.path = "/tmp/rank_0_001.safetensors"

            entry3 = MagicMock()
            entry3.is_file.return_value = True
            entry3.name = "rank_1_001.safetensors"
            entry3.path = "/tmp/rank_1_001.safetensors"

            entry4 = MagicMock()
            entry4.is_file.return_value = True
            entry4.name = "rank_0_aaa.txt"
            entry4.path = "/tmp/rank_0_aaa.txt"

            mock_scandir.return_value.__enter__.return_value = [
                entry1, entry2, entry3, entry4
            ]

            files = self.target_mod._list_rank_files("/tmp", 0)
            self.assertEqual(files, ["/tmp/rank_0_001.safetensors", "/tmp/rank_0_002.safetensors"])

    def test_list_rank_files_skips_non_file(self):
        """Non-file entries should be ignored."""
        with patch("os.scandir") as mock_scandir:
            entry1 = MagicMock()
            entry1.is_file.return_value = False
            entry1.name = "rank_0_001.safetensors"
            entry1.path = "/tmp/rank_0_001.safetensors"

            mock_scandir.return_value.__enter__.return_value = [entry1]

            files = self.target_mod._list_rank_files("/tmp", 0)
            self.assertEqual(files, [])

    def test_iter_rank_tensors_basic(self):
        """Iterate over safetensors files and yield (key, tensor)."""
        torch = sys.modules["torch"]
        fake_safe_open = sys.modules["safetensors.torch"].safe_open

        t1 = torch.ones((2, 2), dtype=torch.float32)
        t2 = torch.zeros((1,), dtype=torch.float32)

        fake_safe_open.side_effect = [
            FakeSafeOpenCtx({"a": t1}),
            FakeSafeOpenCtx({"b": t2}),
        ]

        out = list(self.target_mod.iter_rank_tensors(files=["f1", "f2"], device="cpu"))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], "a")
        self.assertEqual(out[1][0], "b")

    def test_iter_rank_tensors_applies_dtype_cast(self):
        """Dtype conversion should be applied if specified."""
        torch = sys.modules["torch"]
        fake_safe_open = sys.modules["safetensors.torch"].safe_open

        t1 = torch.ones((2, 2), dtype=torch.float32)

        fake_safe_open.return_value = FakeSafeOpenCtx({"a": t1})

        out = list(self.target_mod.iter_rank_tensors(
            files=["f1"],
            device="cpu",
            dtype=torch.float16,
        ))

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1].dtype.name, "float16")

    def test_iter_rank_tensors_key_filter(self):
        """key_filter should exclude keys that don't match."""
        torch = sys.modules["torch"]
        fake_safe_open = sys.modules["safetensors.torch"].safe_open

        t1 = torch.ones((2, 2), dtype=torch.float32)
        t2 = torch.zeros((1,), dtype=torch.float32)

        fake_safe_open.return_value = FakeSafeOpenCtx({"keep": t1, "drop": t2})

        out = list(self.target_mod.iter_rank_tensors(
            files=["f1"],
            device="cpu",
            key_filter=lambda k: k == "keep",
        ))

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "keep")

    def test_load_rank_tensors_to_dict_overwrite(self):
        """On duplicate keys, overwrite keeps the last tensor."""
        torch = sys.modules["torch"]
        with patch.object(self.target_mod, "iter_rank_tensors") as mock_iter:
            t1 = torch.ones((1,))
            t2 = torch.zeros((1,))
            mock_iter.return_value = [("x", t1), ("x", t2)]

            out = self.target_mod.load_rank_tensors_to_dict(files=["f1"], on_duplicate="overwrite")
            self.assertEqual(out["x"].shape, t2.shape)

    def test_load_rank_tensors_to_dict_keep_first(self):
        """keep_first should retain the first occurrence."""
        torch = sys.modules["torch"]
        with patch.object(self.target_mod, "iter_rank_tensors") as mock_iter:
            t1 = torch.ones((1,))
            t2 = torch.zeros((1,))
            mock_iter.return_value = [("x", t1), ("x", t2)]

            out = self.target_mod.load_rank_tensors_to_dict(files=["f1"], on_duplicate="keep_first")
            self.assertEqual(out["x"].shape, t1.shape)

    def test_direct_copy_copy_path(self):
        """Direct copy without memcpy should use tensor.copy_."""
        torch = sys.modules["torch"]
        src = torch.ones((2, 2), dtype=torch.float32)
        dst = torch.zeros((2, 2), dtype=torch.float32)
        self.target_mod._direct_copy(src, dst, use_memcp=False)

    def test_direct_copy_memcpy_path(self):
        """With use_memcp=True, should call acl.rt.memcpy."""
        torch = sys.modules["torch"]
        fake_memcpy = sys.modules["acl.rt"].memcpy

        src = torch.ones((2, 2), dtype=torch.float32)
        dst = torch.zeros((2, 2), dtype=torch.float32)

        self.target_mod._direct_copy(src, dst, use_memcp=True)
        fake_memcpy.assert_called_once()

    def test_resolve_device_preferred(self):
        """Explicit device string should be returned as torch.device."""
        torch = sys.modules["torch"]
        dev = self.target_mod.resolve_device("cpu")
        self.assertEqual(dev.type, torch.device("cpu").type)

    def test_resolve_device_prefers_npu(self):
        """If no device given and npu is available, use npu."""
        out = self.target_mod.resolve_device(None)
        self.assertEqual(out.type, "npu")

    def test_resolve_device_cuda_fallback(self):
        """If npu not available, fallback to cuda."""
        torch = sys.modules["torch"]
        # remove npu -> fallback to cuda
        delattr(torch, "npu")
        torch.cuda._avail = True

        out = self.target_mod.resolve_device(None)
        self.assertEqual(out.type, "cuda")

    def test_resolve_device_cpu_fallback(self):
        """If neither npu nor cuda, fallback to cpu."""
        torch = sys.modules["torch"]
        delattr(torch, "npu")
        torch.cuda._avail = False

        out = self.target_mod.resolve_device(None)
        self.assertEqual(out.type, "cpu")

    def test_broadcast_if_gpu_skips_cpu(self):
        """Broadcast should not be called for CPU tensors."""
        torch = sys.modules["torch"]
        t = torch.zeros((2, 2), device="cpu")

        dist_mod = sys.modules["torch"].distributed
        dist_mod.broadcast.reset_mock()

        self.target_mod.broadcast_if_gpu(t, src=0, group=None)
        dist_mod.broadcast.assert_not_called()

    def test_broadcast_if_gpu_calls_on_npu(self):
        """Broadcast should be called for NPU tensors."""
        dist_mod = sys.modules["torch"].distributed
        dist_mod.broadcast.reset_mock()

        class FakeTensorObj:
            def __init__(self):
                self.device = types.SimpleNamespace(type="npu")

        self.target_mod.broadcast_if_gpu(FakeTensorObj(), src=0, group=None)
        dist_mod.broadcast.assert_called_once()

    def test_broadcast_if_gpu_calls_on_cuda(self):
        """Broadcast should be called for CUDA tensors."""
        dist_mod = sys.modules["torch"].distributed
        dist_mod.broadcast.reset_mock()

        class FakeTensorObj:
            def __init__(self):
                self.device = types.SimpleNamespace(type="cuda")

        self.target_mod.broadcast_if_gpu(FakeTensorObj(), src=0, group=None)
        dist_mod.broadcast.assert_called_once()

    def test_split_tensors_and_meta(self):
        """Separate tensors from metadata (keys starting with __)."""
        torch = sys.modules["torch"]

        params = {
            "w": torch.ones((2, 2)),
            "__simple_ep_meta__": {"a": 1},
            "not_tensor": 123,
        }

        tensors, meta = self.target_mod.split_tensors_and_meta(params)

        self.assertIn("w", tensors)
        self.assertNotIn("not_tensor", tensors)
        self.assertIn("__simple_ep_meta__", meta)
        self.assertEqual(json.loads(meta["__simple_ep_meta__"])["a"], 1)

    def test_split_tensors_and_meta_without_meta(self):
        """If no metadata keys, meta dict is empty."""
        torch = sys.modules["torch"]

        params = {
            "w": torch.ones((2, 2)),
            "not_tensor": 123,
        }

        tensors, meta = self.target_mod.split_tensors_and_meta(params)
        self.assertIn("w", tensors)
        self.assertEqual(meta, {})

    def test_custom_worker_extensions_statistics(self):
        """vllm_statistics should write stats and clear."""
        obj = self.target_mod.CustomWorkerExtensions()

        fake_stat_obj = sys.modules[
            "agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat"
        ].vllm_output_statics

        obj.vllm_statistics()

        fake_stat_obj.write_stats_tofile.assert_called_once()
        fake_stat_obj.clear.assert_called_once()

    def test_custom_worker_extensions_update_weights_use_hf_true_no_files(self):
        """When use_hf=True but no files found, should return zero moved bytes."""
        obj = self.target_mod.CustomWorkerExtensions()
        obj.rank = 0

        with patch("os.scandir") as mock_scandir:
            mock_scandir.return_value.__enter__.return_value = []
            moved = obj.update_weights(True, "tmp", "dir")

        self.assertEqual(moved["num_files"], 0)
        self.assertEqual(moved["threads"], 0)

    def test_custom_worker_extensions_update_weights_use_hf_true(self):
        """Update weights using HuggingFace format (load_weights on model)."""
        torch = sys.modules["torch"]

        obj = self.target_mod.CustomWorkerExtensions()
        obj.rank = 0

        fake_model = MagicMock()
        fake_model.load_weights.return_value = {"ok": True}
        obj.model_runner = MagicMock(model=fake_model)

        entry = MagicMock()
        entry.is_file.return_value = True
        entry.name = "x.safetensors"
        entry.path = "/tmp/x.safetensors"

        with patch("os.scandir") as mock_scandir:
            mock_scandir.return_value.__enter__.return_value = [entry]

            fake_safe_open = sys.modules["safetensors.torch"].safe_open
            fake_safe_open.return_value = FakeSafeOpenCtx({"w": torch.ones((2, 2))})

            fake_get_dev = sys.modules["verl.utils.device"].get_torch_device
            fake_get_dev.return_value.current_device.return_value = "cpu"

            patch_loader = sys.modules["verl.utils.vllm.patch"].patch_vllm_moe_model_weight_loader

            moved = obj.update_weights(True, "tmp", "dir")
            self.assertEqual(moved, 0)

            patch_loader.assert_called_once_with(fake_model)
            fake_model.load_weights.assert_called_once()

    def test_custom_worker_extensions_update_weights_use_hf_false(self):
        """Update weights using threaded NPU loading."""
        torch = sys.modules["torch"]

        obj = self.target_mod.CustomWorkerExtensions()
        obj.rank = 0

        fake_param = torch.ones((2, 2))
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w", fake_param)]
        obj.model_runner = MagicMock(model=fake_model)

        with patch.object(self.target_mod, "_current_accel_device", return_value=torch.device("cpu")):
            with patch.object(self.target_mod, "load_rank_to_npu_threaded") as mock_load:
                mock_load.return_value = {"bytes_total": 123}

                moved = obj.update_weights(False, "tmpdir")
                self.assertEqual(moved, 123)
                mock_load.assert_called_once()

    def test_current_accel_device_prefers_npu(self):
        """_current_accel_device should return npu if available."""
        out = self.target_mod._current_accel_device()
        self.assertEqual(out.type, "npu")

    def test_current_accel_device_cuda_fallback(self):
        """If npu not available, fallback to cuda."""
        torch = sys.modules["torch"]
        delattr(torch, "npu")
        torch.cuda._avail = True

        out = self.target_mod._current_accel_device()
        self.assertEqual(out.type, "cuda")

    def test_current_accel_device_raises_when_no_backend(self):
        """If no accelerator available, raise RuntimeError."""
        torch = sys.modules["torch"]
        delattr(torch, "npu")
        torch.cuda._avail = False

        with self.assertRaises(RuntimeError):
            self.target_mod._current_accel_device()

    def test_core_transfer_to_npu_alloc_targets(self):
        """Transfer CPU tensors to NPU with new allocation."""
        torch = sys.modules["torch"]

        cpu_tensors = {
            "w": torch.ones((2, 2), dtype=torch.float32, device="cpu"),
            "b": torch.ones((1,), dtype=torch.float32, device="cpu"),
        }

        out = self.target_mod._core_transfer_to_npu(
            cpu_tensors,
            target_dtype=torch.float16,
            npu_targets=None,
            device="npu:0",
            num_streams=2,
            use_memcpy=False,
        )

        self.assertEqual(set(out.keys()), {"w", "b"})
        self.assertEqual(out["w"].device.type, "npu")
        self.assertEqual(out["w"].dtype.name, "float16")

    def test_core_transfer_to_npu_reuse_targets(self):
        """Reuse pre-allocated NPU tensors."""
        torch = sys.modules["torch"]

        cpu_tensors = {
            "w": torch.ones((2, 2), dtype=torch.float32, device="cpu"),
        }

        npu_targets = {
            "w": torch.empty((2, 2), dtype=torch.float16, device="npu:0"),
        }

        out = self.target_mod._core_transfer_to_npu(
            cpu_tensors,
            target_dtype=torch.float16,
            npu_targets=npu_targets,
            device="npu:0",
            num_streams=1,
            use_memcpy=True,
        )

        self.assertEqual(set(out.keys()), {"w"})

    def test_core_transfer_to_npu_reuse_targets_with_permute_fix(self):
        """
        When source and destination shapes require permutation (3D with swapped dims),
        a permute should be applied before copy.
        """
        torch = sys.modules["torch"]

        cpu_tensors = {
            "w": torch.ones((2, 3, 4), dtype=torch.float32, device="cpu"),
        }

        # destination shape (2,4,3) triggers permute(0,2,1)
        npu_targets = {
            "w": torch.empty((2, 4, 3), dtype=torch.float16, device="npu:0"),
        }

        out = self.target_mod._core_transfer_to_npu(
            cpu_tensors,
            target_dtype=torch.float16,
            npu_targets=npu_targets,
            device="npu:0",
            num_streams=1,
            use_memcpy=False,
        )

        self.assertIn("w", out)

    def test_thread_load_and_h2d_returns_stats(self):
        """Threaded load should return statistics dictionary."""
        torch = sys.modules["torch"]

        with patch.object(self.target_mod, "load_rank_tensors_to_dict") as mock_load:
            with patch.object(self.target_mod, "_core_transfer_to_npu") as mock_transfer:
                mock_load.return_value = {
                    "w": torch.ones((2, 2), dtype=torch.float16, device="cpu"),
                    "b": torch.ones((1,), dtype=torch.float16, device="cpu"),
                }
                mock_transfer.return_value = {"w": torch.ones((2, 2), device="npu")}

                stats = self.target_mod._thread_load_and_h2d(
                    ["f1", "f2"],
                    npu_targets=None,
                    target_dtype=torch.float16,
                    device="npu:0",
                    num_streams=2,
                    key_filter=None,
                )

        self.assertEqual(stats["num_files"], 2)
        self.assertEqual(stats["num_tensors"], 2)
        self.assertIn("bytes_cpu", stats)
        self.assertIn("t_total_s", stats)

    def test_load_rank_to_npu_threaded_no_files(self):
        """If no rank files found, return zero stats."""
        with patch.object(self.target_mod, "_list_rank_files", return_value=[]):
            stats = self.target_mod.load_rank_to_npu_threaded(
                final_dir="/tmp",
                r=0,
                num_threads=4,
            )

        self.assertEqual(stats["threads"], 0)
        self.assertEqual(stats["num_files"], 0)
        self.assertEqual(stats["bytes_total"], 0)

    def test_load_rank_to_npu_threaded_runs_chunks(self):
        """
        Verify that files are split into chunks and processed concurrently.
        """
        fake_files = [
            "/tmp/rank_0_001.safetensors",
            "/tmp/rank_0_002.safetensors",
            "/tmp/rank_0_003.safetensors",
        ]

        with patch.object(self.target_mod, "_list_rank_files", return_value=fake_files):
            with patch.object(self.target_mod, "_thread_load_and_h2d") as mock_thr:
                mock_thr.side_effect = [
                    {"bytes_cpu": 10, "num_files": 1, "num_tensors": 1, "t_cpu_load_s": 0, "t_h2d_s": 0, "t_total_s": 0},
                    {"bytes_cpu": 20, "num_files": 1, "num_tensors": 1, "t_cpu_load_s": 0, "t_h2d_s": 0, "t_total_s": 0},
                    {"bytes_cpu": 30, "num_files": 1, "num_tensors": 1, "t_cpu_load_s": 0, "t_h2d_s": 0, "t_total_s": 0},
                ]

                stats = self.target_mod.load_rank_to_npu_threaded(
                    final_dir="/tmp",
                    r=0,
                    num_threads=8,
                    npu_targets=None,
                    target_dtype=None,
                    device="npu:0",
                    num_streams=1,
                    key_filter=None,
                )

        self.assertEqual(stats["num_files"], 3)
        self.assertEqual(stats["bytes_total"], 60)
        self.assertEqual(stats["note"].startswith("Loaded on threads"), True)
        self.assertGreaterEqual(stats["threads"], 1)


if __name__ == "__main__":
    unittest.main()