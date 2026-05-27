#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree, no real environment pollution
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_worker_env():
    # ---- vllm.logger ----
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_logger = MagicMock()
    fake_vllm_logger.logger = fake_logger

    # ---- vllm.utils ----
    fake_vllm_utils = types.ModuleType("vllm.utils")
    GiB_bytes = 1024**3
    fake_vllm_utils.GiB_bytes = GiB_bytes

    # ---- vllm_ascend.device_allocator.camem ----
    fake_camem = types.ModuleType("vllm_ascend.device_allocator.camem")
    ca_mock = MagicMock()
    fake_camem.CaMemAllocator = MagicMock()
    fake_camem.CaMemAllocator.get_instance.return_value = ca_mock

    # ---- vllm_ascend.platform ----
    fake_platform = types.ModuleType("vllm_ascend.platform")
    npu_platform_mock = MagicMock()
    fake_platform.NPUPlatform = npu_platform_mock

    # ---- vllm_ascend.worker.worker_v1 ----
    fake_worker = types.ModuleType("vllm_ascend.worker.worker_v1")
    class FakeNPUWorker:
        pass
    fake_worker.NPUWorker = FakeNPUWorker

    # ---- vllm_ascend.utils ----
    fake_utils = types.ModuleType("vllm_ascend.utils")
    sleep_mode_enabled_mock = MagicMock(return_value=True)
    fake_utils.sleep_mode_enabled = sleep_mode_enabled_mock

    # ---- Aura package paths (to locate the module under test) ----
    import os as _os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [_os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = [_os.path.join(base_path, "runner/infer_adapter")]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch")]
    fake_0_11_0_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_11_0")
    fake_0_11_0_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_11_0")]

    # Extra packages to prevent import errors
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_device_allocator = types.ModuleType("vllm_ascend.device_allocator")
    fake_vllm_ascend_device_allocator.__path__ = []
    fake_vllm_ascend_worker = types.ModuleType("vllm_ascend.worker")
    fake_vllm_ascend_worker.__path__ = []
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []

    fakes = {
        "vllm": fake_vllm,
        "vllm.logger": fake_vllm_logger,
        "vllm.utils": fake_vllm_utils,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.device_allocator": fake_vllm_ascend_device_allocator,
        "vllm_ascend.device_allocator.camem": fake_camem,
        "vllm_ascend.platform": fake_platform,
        "vllm_ascend.worker": fake_vllm_ascend_worker,
        "vllm_ascend.worker.worker_v1": fake_worker,
        "vllm_ascend.utils": fake_utils,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
    }

    yield {
        "fakes": fakes,
        "logger": fake_logger,
        "CaMemAllocator": fake_camem.CaMemAllocator,
        "ca_instance": ca_mock,
        "NPUPlatform": npu_platform_mock,
        "sleep_mode_enabled": sleep_mode_enabled_mock,
        "GiB_bytes": GiB_bytes,
        "FakeNPUWorker": FakeNPUWorker,
    }


# ---------------------------------------------------------------------------
# Helper: temporarily inject fake modules and import the module under test
# ---------------------------------------------------------------------------
def import_module(fake_worker_env):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_worker_v1"
    if module_name in sys.modules:
        del sys.modules[module_name]
    fakes = fake_worker_env["fakes"]
    with patch.dict(sys.modules, fakes):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_worker_v1 as mod
    return mod


# ---------------------------------------------------------------------------
# Helper: create a mock NPUWorker instance
# ---------------------------------------------------------------------------
def make_worker_mock():
    worker = MagicMock()
    model = MagicMock()
    buffer1 = MagicMock()
    buffer2 = MagicMock()
    buffer1.cpu.return_value.clone.return_value = "buf1_cloned"
    buffer2.cpu.return_value.clone.return_value = "buf2_cloned"
    model.named_buffers.return_value = [("b1", buffer1), ("b2", buffer2)]
    worker.model_runner.model = model
    return worker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMultiLevelSleep:
    @pytest.fixture(autouse=True)
    def setup(self, fake_worker_env):
        self.env = fake_worker_env
        self.mod = import_module(fake_worker_env)
        self.sleep = self.mod.multi_level_sleep
        # Reset mocks to ensure test isolation
        self.env["sleep_mode_enabled"].reset_mock()
        self.env["NPUPlatform"].mem_get_info.reset_mock()
        self.env["ca_instance"].sleep.reset_mock()
        self.env["logger"].info.reset_mock()

    def test_sleep_mode_disabled_raises(self):
        """Raises ValueError when sleep_mode_enabled returns False."""
        self.env["sleep_mode_enabled"].return_value = False
        worker = make_worker_mock()
        with pytest.raises(ValueError, match="Sleep mode is not enabled"):
            self.sleep(worker, level=1)
        self.env["sleep_mode_enabled"].assert_called_once()

    def test_level_2_saves_buffers(self):
        """Level 2 saves model buffers and calls allocator.sleep(tuple())."""
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (10 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
            (12 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
        ]
        worker = make_worker_mock()
        self.sleep(worker, level=2)
        model = worker.model_runner.model
        model.named_buffers.assert_called_once()
        expected = {"b1": "buf1_cloned", "b2": "buf2_cloned"}
        assert worker._sleep_saved_buffers == expected
        self.env["ca_instance"].sleep.assert_called_once_with(offload_tags=tuple())
        self.env["logger"].info.assert_called_once()

    def test_level_1_offload_weights(self):
        """Level 1 calls allocator.sleep(offload_tags=("weights",)) and does not save buffers."""
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (5 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
            (8 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
        ]
        worker = make_worker_mock()
        self.sleep(worker, level=1)
        self.env["ca_instance"].sleep.assert_called_once_with(offload_tags=("weights",))
        # _sleep_saved_buffers should not be set as a dict because level != 2
        assert not isinstance(worker._sleep_saved_buffers, dict)

    def test_level_0_offload_kv_cache(self):
        """Level 0 calls allocator.sleep(offload_tags=("kv_cache",))."""
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (5 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
            (8 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
        ]
        worker = make_worker_mock()
        self.sleep(worker, level=0)
        self.env["ca_instance"].sleep.assert_called_once_with(offload_tags=("kv_cache",))

    def test_other_level_offload_empty(self):
        """Levels other than 0,1,2 call allocator.sleep(offload_tags=tuple())."""
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (5 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
            (6 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
        ]
        worker = make_worker_mock()
        self.sleep(worker, level=3)
        self.env["ca_instance"].sleep.assert_called_once_with(offload_tags=tuple())

    def test_freed_bytes_negative_raises(self):
        """Raises ValueError when freed_bytes < 0."""
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (10 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
            (9 * self.env["GiB_bytes"], 20 * self.env["GiB_bytes"]),
        ]
        worker = make_worker_mock()
        with pytest.raises(ValueError, match="Memory usage increased after sleeping"):
            self.sleep(worker, level=1)
        self.env["ca_instance"].sleep.assert_called_once()

    def test_logger_info_called_with_proper_values(self):
        """Verifies that logger.info is called with the correct freed/used memory sizes."""
        free_before = 3 * self.env["GiB_bytes"]
        free_after = 7 * self.env["GiB_bytes"]
        total = 16 * self.env["GiB_bytes"]
        self.env["NPUPlatform"].mem_get_info.side_effect = [
            (free_before, total),
            (free_after, total),
        ]
        worker = make_worker_mock()
        self.sleep(worker, level=1)

        freed_bytes = free_after - free_before
        used_bytes = total - free_after
        self.env["logger"].info.assert_called_once_with(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / self.env["GiB_bytes"],
            used_bytes / self.env["GiB_bytes"],
        )

    def test_function_assigned_to_NPUWorker(self):
        """Checks that multi_level_sleep is assigned to NPUWorker.sleep."""
        NPUWorker = self.env["FakeNPUWorker"]
        assert NPUWorker.sleep is self.sleep
