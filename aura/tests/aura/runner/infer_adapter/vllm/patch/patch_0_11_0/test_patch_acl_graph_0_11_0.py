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
from contextlib import ExitStack
from unittest.mock import MagicMock, patch, AsyncMock
from enum import IntEnum
import pytest

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_acl_graph
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_patch_acl_graph_env():
    # ---- Fake vllm modules ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_compilation = types.ModuleType("vllm.compilation")
    fake_vllm_compilation.__path__ = []
    fake_vllm_compilation_counter = types.ModuleType("vllm.compilation.counter")
    fake_vllm_compilation_counter.compilation_counter = MagicMock()
    fake_vllm_compilation_monitor = types.ModuleType("vllm.compilation.monitor")
    fake_vllm_compilation_monitor.validate_cudagraph_capturing_enabled = MagicMock()
    fake_vllm_config = types.ModuleType("vllm.config")
    class CUDAGraphMode(IntEnum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2
    fake_vllm_config.CUDAGraphMode = CUDAGraphMode
    fake_vllm_forward_context = types.ModuleType("vllm.forward_context")
    fake_vllm_forward_context.get_forward_context = MagicMock()
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.logger = MagicMock()

    # ---- Fake vllm_ascend modules ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_compilation = types.ModuleType("vllm_ascend.compilation")
    fake_vllm_ascend_compilation.__path__ = []
    fake_vllm_ascend_compilation_acl_graph = types.ModuleType(
        "vllm_ascend.compilation.acl_graph"
    )
    # ACLGraphEntry and ACLGraphWrapper mocks
    class ACLGraphEntry:
        def __init__(self, batch_descriptor=None):
            self.batch_descriptor = batch_descriptor
            self.aclgraph = None
            self.input_addresses = None
            self.output = None
    class ACLGraphWrapper:
        pass
    fake_vllm_ascend_compilation_acl_graph.ACLGraphEntry = ACLGraphEntry
    fake_vllm_ascend_compilation_acl_graph.ACLGraphWrapper = ACLGraphWrapper

    # ---- Fake torch extensions ----
    fake_torch_ops = types.ModuleType("torch.ops")
    fake_torch_ops.__path__ = []
    fake_torch_ops._C_ascend = types.ModuleType("torch.ops._C_ascend")

    # Packages for aura
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [
        os.path.join(real_aura_path[0], "runner")
    ] if real_aura_path else []
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = []
    fake_aura_runner_infer_adapter_vllm = types.ModuleType(
        "aura.runner.infer_adapter.vllm"
    )
    fake_aura_runner_infer_adapter_vllm.__path__ = []
    fake_aura_runner_infer_adapter_vllm_patch = types.ModuleType(
        "aura.runner.infer_adapter.vllm.patch"
    )
    fake_aura_runner_infer_adapter_vllm_patch.__path__ = []
    fake_aura_runner_infer_adapter_vllm_patch_0_11_0 = types.ModuleType(
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0"
    )
    fake_aura_runner_infer_adapter_vllm_patch_0_11_0.__path__ = [
        os.path.join(
            real_aura_path[0],
            "runner/infer_adapter/vllm/patch/patch_0_11_0",
        )
    ] if real_aura_path else []

    all_fakes = {
        "vllm": fake_vllm,
        "vllm.compilation": fake_vllm_compilation,
        "vllm.compilation.counter": fake_vllm_compilation_counter,
        "vllm.compilation.monitor": fake_vllm_compilation_monitor,
        "vllm.config": fake_vllm_config,
        "vllm.forward_context": fake_vllm_forward_context,
        "vllm.logger": fake_vllm_logger,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.compilation": fake_vllm_ascend_compilation,
        "vllm_ascend.compilation.acl_graph": fake_vllm_ascend_compilation_acl_graph,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_aura_runner_infer_adapter_vllm,
        "aura.runner.infer_adapter.vllm.patch": fake_aura_runner_infer_adapter_vllm_patch,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_aura_runner_infer_adapter_vllm_patch_0_11_0,
    }

    fake_npu = MagicMock()
    fake_npu.NPUGraph = MagicMock()
    fake_npu.current_device = MagicMock(return_value=0)
    fake_npu.set_device = MagicMock()
    fake_npu.empty_cache = MagicMock()
    fake_distributed = MagicMock()
    fake_distributed.barrier = MagicMock()

    with patch.dict(sys.modules, all_fakes), \
         patch.object(torch, "npu", fake_npu, create=True), \
         patch.object(torch, "distributed", fake_distributed, create=True), \
         patch.object(torch.ops, "_C_ascend", fake_torch_ops._C_ascend, create=True):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_acl_graph as patch_acl
        yield {
            "module": patch_acl,
            "weak_ref_tensor": patch_acl.weak_ref_tensor,
            "weak_ref_tensors": patch_acl.weak_ref_tensors,
            "ACLGraphWrapper": patch_acl.ACLGraphWrapper,
            "fake_npu": fake_npu,
            "fake_distributed": fake_distributed,
            "fake_counter": fake_vllm_compilation_counter.compilation_counter,
            "fake_validate": fake_vllm_compilation_monitor.validate_cudagraph_capturing_enabled,
            "fake_logger": fake_vllm_logger.logger,
            "CUDAGraphMode": CUDAGraphMode,
            "fake_forward_context": fake_vllm_forward_context,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_tensor(size=(2, 3)):
    """Create a real torch tensor for isinstance checks."""
    return torch.randn(size)

def make_mock_tensor_with_as_strided(return_value=None):
    """Create a mock tensor that can call as_strided."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.as_strided.return_value = return_value or MagicMock(spec=torch.Tensor)
    # Ensure isinstance works: we need a real Tensor instance, not mock.
    real_tensor = torch.zeros(1)
    real_tensor.as_strided = MagicMock(return_value=return_value or torch.zeros(1))
    return real_tensor


# ---------------------------------------------------------------------------
# Tests for weak_ref_tensor
# ---------------------------------------------------------------------------
class TestWeakRefTensor:
    def test_tensor_input(self, fake_patch_acl_graph_env):
        weak_ref_tensor = fake_patch_acl_graph_env["weak_ref_tensor"]
        tensor = make_tensor((2, 2))
        result = weak_ref_tensor(tensor)

        mock_tensor = make_mock_tensor_with_as_strided()
        expected_result = torch.zeros(1)
        mock_tensor.as_strided.return_value = expected_result
        result = weak_ref_tensor(mock_tensor)
        mock_tensor.as_strided.assert_called_once_with(
            mock_tensor.size(), mock_tensor.stride(), mock_tensor.storage_offset()
        )
        assert result is expected_result

    def test_non_tensor_input(self, fake_patch_acl_graph_env):
        weak_ref_tensor = fake_patch_acl_graph_env["weak_ref_tensor"]
        obj = [1, 2, 3]
        result = weak_ref_tensor(obj)
        assert result is obj


# ---------------------------------------------------------------------------
# Tests for weak_ref_tensors
# ---------------------------------------------------------------------------
class TestWeakRefTensors:
    def test_single_tensor(self, fake_patch_acl_graph_env):
        weak_ref_tensors = fake_patch_acl_graph_env["weak_ref_tensors"]
        tensor = make_mock_tensor_with_as_strided()
        result = weak_ref_tensors(tensor)
        tensor.as_strided.assert_called_once()
        assert isinstance(result, torch.Tensor)

    def test_list_of_tensors(self, fake_patch_acl_graph_env):
        weak_ref_tensors = fake_patch_acl_graph_env["weak_ref_tensors"]
        t1 = make_mock_tensor_with_as_strided()
        t2 = make_mock_tensor_with_as_strided()
        result = weak_ref_tensors([t1, t2])
        t1.as_strided.assert_called_once()
        t2.as_strided.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_tuple_of_tensors(self, fake_patch_acl_graph_env):
        weak_ref_tensors = fake_patch_acl_graph_env["weak_ref_tensors"]
        t1 = make_mock_tensor_with_as_strided()
        t2 = make_mock_tensor_with_as_strided()
        result = weak_ref_tensors((t1, t2))
        t1.as_strided.assert_called_once()
        t2.as_strided.assert_called_once()
        assert isinstance(result, tuple)

    def test_mixed_list_non_tensor(self, fake_patch_acl_graph_env):
        weak_ref_tensors = fake_patch_acl_graph_env["weak_ref_tensors"]
        t = make_mock_tensor_with_as_strided()
        non_tensor = 123
        result = weak_ref_tensors([t, non_tensor])
        t.as_strided.assert_called_once()
        assert result[1] == 123  # non-tensor remains unchanged

    def test_invalid_input(self, fake_patch_acl_graph_env):
        weak_ref_tensors = fake_patch_acl_graph_env["weak_ref_tensors"]
        with pytest.raises(ValueError, match="Invalid type for tensors"):
            weak_ref_tensors("not_a_tensor")


# ---------------------------------------------------------------------------
# Tests for ACLGraphWrapper.__call__
# ---------------------------------------------------------------------------
class TestACLGraphWrapperCall:
    @pytest.fixture
    def wrapper_mock(self, fake_patch_acl_graph_env):
        """Create a mock ACLGraphWrapper instance with common attributes."""
        CUDAGraphMode = fake_patch_acl_graph_env["CUDAGraphMode"]
        wrapper = MagicMock()
        wrapper.runnable = MagicMock()
        wrapper.runtime_mode = CUDAGraphMode.FULL          # 枚举，有 .name
        wrapper.concrete_aclgraph_entries = {}
        wrapper.aclgraph_options = MagicMock()
        wrapper.aclgraph_options.debug_log_enable = False
        wrapper.aclgraph_options.gc_disable = False
        wrapper.aclgraph_options.weak_ref_output = False
        wrapper.graph_pool = None
        wrapper.is_debugging_mode = False
        return wrapper

    @pytest.fixture
    def setup_env(self, fake_patch_acl_graph_env, wrapper_mock):
        """Setup environment mocks for __call__."""
        env = fake_patch_acl_graph_env
        CUDAGraphMode = env["CUDAGraphMode"]
        forward_ctx = MagicMock()
        forward_ctx.batch_descriptor = "batch1"
        forward_ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        env["fake_forward_context"].get_forward_context.return_value = forward_ctx
        yield env
        env["fake_forward_context"].get_forward_context.reset_mock()

    def test_runtime_mode_none(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """When forward context mode is NONE, should bypass graph."""
        env = setup_env
        CUDAGraphMode = env["CUDAGraphMode"]
        env["fake_forward_context"].get_forward_context.return_value.cudagraph_runtime_mode = CUDAGraphMode.NONE
        wrapper_mock.runnable.return_value = "output"

        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, "arg1", kwarg1="val")

        wrapper_mock.runnable.assert_called_once_with("arg1", kwarg1="val")
        assert result == "output"

    def test_runtime_mode_mismatch(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """When self.runtime_mode != forward mode, bypass graph."""
        env = setup_env
        CUDAGraphMode = env["CUDAGraphMode"]
        wrapper_mock.runtime_mode = CUDAGraphMode.PIECEWISE
        wrapper_mock.runnable.return_value = "output"

        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, "arg")

        wrapper_mock.runnable.assert_called_once_with("arg")
        assert result == "output"

    def test_capture_first_time(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """First time for a batch_descriptor -> capture aclgraph."""
        env = setup_env
        wrapper_mock.aclgraph_options.debug_log_enable = True
        wrapper_mock.aclgraph_options.gc_disable = True
        wrapper_mock.aclgraph_options.weak_ref_output = True

        # runnable must return a Tensor to pass weak_ref_tensors
        output_tensor = make_mock_tensor_with_as_strided()
        wrapper_mock.runnable.return_value = output_tensor

        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, "input_tensor")

        assert "batch1" in wrapper_mock.concrete_aclgraph_entries
        entry = wrapper_mock.concrete_aclgraph_entries["batch1"]
        env["fake_validate"].assert_called_once()
        env["fake_logger"].debug.assert_called_once()
        env["fake_npu"].NPUGraph.assert_called_once()
        aclgraph = env["fake_npu"].NPUGraph.return_value
        aclgraph.capture_begin.assert_called_once()
        wrapper_mock.runnable.assert_called_once_with("input_tensor")
        aclgraph.capture_end.assert_called_once()
        # The function returns the original output (not weak ref)
        assert torch.equal(result, output_tensor)
        # entry.output should be a weak ref (different object)
        assert entry.output is not output_tensor
        assert entry.aclgraph is aclgraph

    def test_replay_existing_graph(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """When entry exists with aclgraph, replay it."""
        env = setup_env
        ACLGraphEntry = fake_patch_acl_graph_env["module"].ACLGraphEntry
        entry = ACLGraphEntry(batch_descriptor="batch1")
        entry.aclgraph = MagicMock()
        entry.output = "cached_output"
        wrapper_mock.concrete_aclgraph_entries["batch1"] = entry

        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, "tensor")

        entry.aclgraph.replay.assert_called_once()
        env["fake_logger"].info_once.assert_called_with("Replaying aclgraph")
        assert result == "cached_output"
        wrapper_mock.runnable.assert_not_called()

    def test_debugging_mode_address_match(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """In debugging mode with matching addresses, should replay."""
        env = setup_env
        ACLGraphEntry = fake_patch_acl_graph_env["module"].ACLGraphEntry
        wrapper_mock.is_debugging_mode = True
        entry = ACLGraphEntry(batch_descriptor="batch1")
        entry.aclgraph = MagicMock()
        entry.input_addresses = [100]
        entry.output = "debug_output"
        wrapper_mock.concrete_aclgraph_entries["batch1"] = entry

        tensor_mock = MagicMock(spec=torch.Tensor)
        tensor_mock.data_ptr.return_value = 100
        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, tensor_mock)

        entry.aclgraph.replay.assert_called_once()
        assert result == "debug_output"

    def test_debugging_mode_address_mismatch(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """Debugging mode with mismatched addresses raises ValueError."""
        env = setup_env
        ACLGraphEntry = fake_patch_acl_graph_env["module"].ACLGraphEntry
        wrapper_mock.is_debugging_mode = True
        entry = ACLGraphEntry(batch_descriptor="batch1")
        entry.aclgraph = MagicMock()
        entry.input_addresses = [100]
        entry.output = "output"
        wrapper_mock.concrete_aclgraph_entries["batch1"] = entry

        tensor_mock = MagicMock(spec=torch.Tensor)
        tensor_mock.data_ptr.return_value = 200
        __call__ = fake_patch_acl_graph_env["module"].__call__
        with pytest.raises(ValueError, match="Input addresses for aclgraphs are different"):
            __call__(wrapper_mock, tensor_mock)

    def test_capture_with_graph_pool(self, fake_patch_acl_graph_env, wrapper_mock, setup_env):
        """Capture with graph_pool not None -> passes pool tuple to capture_begin."""
        env = setup_env
        wrapper_mock.graph_pool = "pool_obj"

        # runnable returns a tensor
        output_tensor = make_mock_tensor_with_as_strided()
        wrapper_mock.runnable.return_value = output_tensor

        __call__ = fake_patch_acl_graph_env["module"].__call__
        result = __call__(wrapper_mock, "input")

        aclgraph = env["fake_npu"].NPUGraph.return_value
        aclgraph.capture_begin.assert_called_once_with("pool_obj")
        assert torch.equal(result, output_tensor)
