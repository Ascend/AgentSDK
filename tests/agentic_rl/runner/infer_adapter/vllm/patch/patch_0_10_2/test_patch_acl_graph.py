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

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestPatchACLGraph(unittest.TestCase):
    """Test suite for patch_acl_graph.py"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.float32 = "float32"

        class MockTensor:
            def __init__(self):
                self.data_ptr = MagicMock(return_value=12345)

        cls.MockTensor = MockTensor
        cls.mock_torch.Tensor = MockTensor

        cls.mock_torch_npu = MagicMock()
        cls.mock_torch.npu = cls.mock_torch_npu
        cls.mock_torch_npu.NPUGraph = MagicMock()
        cls.mock_torch_npu.current_device = MagicMock(return_value=0)
        cls.mock_torch_npu.set_device = MagicMock()
        cls.mock_torch_npu.empty_cache = MagicMock()

        cls.mock_torch_distributed = MagicMock()
        cls.mock_torch_distributed.barrier = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.compilation = MagicMock()
        cls.mock_vllm.config = MagicMock()
        cls.mock_vllm.forward_context = MagicMock()
        cls.mock_vllm.logger = MagicMock()
        cls.mock_vllm.utils = MagicMock()

        class MockCompilationCounter:
            def __init__(self):
                self.num_cudagraph_captured = 0

        cls.mock_compilation_counter = MockCompilationCounter()

        cls.mock_counter_module = MagicMock()
        cls.mock_counter_module.compilation_counter = cls.mock_compilation_counter

        cls.mock_vllm.compilation.counter = cls.mock_counter_module
        cls.mock_vllm.compilation.monitor = MagicMock()
        cls.mock_vllm.compilation.monitor.validate_cudagraph_capturing_enabled = MagicMock()

        class MockCUDAGraphMode:
            class Mode:
                def __init__(self, name):
                    self.name = name

            NONE = Mode("NONE")
            PIECEWISE = Mode("PIECEWISE")
            FULL = Mode("FULL")

        cls.mock_vllm.config.CUDAGraphMode = MockCUDAGraphMode
        cls.mock_vllm.forward_context.BatchDescriptor = MagicMock()
        cls.mock_vllm.forward_context.get_forward_context = MagicMock()
        cls.mock_vllm.logger.debug = MagicMock()
        cls.mock_vllm.logger.info_once = MagicMock()
        cls.mock_vllm.utils.weak_ref_tensors = MagicMock(return_value="weak_ref_output")

        cls.mock_logger_module = MagicMock()
        cls.mock_logger_module.logger = cls.mock_vllm.logger

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.compilation = MagicMock()
        cls.mock_vllm_ascend.compilation.acl_graph = MagicMock()
        cls.mock_vllm_ascend.compilation.acl_graph.ACLGraphEntry = MagicMock()
        cls.mock_vllm_ascend.compilation.acl_graph.ACLGraphWrapper = MagicMock()

        cls.mock_gc = MagicMock()
        cls.mock_gc.collect = MagicMock()

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch.distributed': cls.mock_torch_distributed,
            'vllm': cls.mock_vllm,
            'vllm.compilation': cls.mock_vllm.compilation,
            'vllm.compilation.counter': cls.mock_counter_module,
            'vllm.compilation.monitor': cls.mock_vllm.compilation.monitor,
            'vllm.config': cls.mock_vllm.config,
            'vllm.forward_context': cls.mock_vllm.forward_context,
            'vllm.logger': cls.mock_logger_module,
            'vllm.utils': cls.mock_vllm.utils,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.compilation': cls.mock_vllm_ascend.compilation,
            'vllm_ascend.compilation.acl_graph': cls.mock_vllm_ascend.compilation.acl_graph,
            'gc': cls.mock_gc,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))

        module_path = os.path.join(
            project_root,
            'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_acl_graph.py'
        )

        spec = importlib.util.spec_from_file_location(
            'patch_acl_graph',
            module_path
        )
        cls.patch_acl_graph = importlib.util.module_from_spec(spec)
        sys.modules['patch_acl_graph'] = cls.patch_acl_graph
        spec.loader.exec_module(cls.patch_acl_graph)

        cls.patch_acl_graph.logger = cls.mock_vllm.logger

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.mock_torch.reset_mock()
        self.mock_torch_npu.reset_mock()
        self.mock_vllm.reset_mock()
        self.mock_vllm_ascend.reset_mock()
        self.mock_gc.reset_mock()

        self.mock_compilation_counter.num_cudagraph_captured = 0

        self.mock_self = MagicMock()
        self.mock_self.runtime_mode = self.mock_vllm.config.CUDAGraphMode.PIECEWISE
        self.mock_self.concrete_aclgraph_entries = {}
        self.mock_self.aclgraph_options = MagicMock()
        self.mock_self.aclgraph_options.debug_log_enable = False
        self.mock_self.aclgraph_options.gc_disable = False
        self.mock_self.aclgraph_options.weak_ref_output = False
        self.mock_self.graph_pool = None
        self.mock_self.runnable = MagicMock(return_value="mock_output")

        self.mock_forward_context = MagicMock()
        self.mock_batch_descriptor = MagicMock()
        self.mock_forward_context.batch_descriptor = self.mock_batch_descriptor
        self.mock_forward_context.cudagraph_runtime_mode = self.mock_vllm.config.CUDAGraphMode.PIECEWISE
        self.mock_vllm.forward_context.get_forward_context.return_value = self.mock_forward_context

        self.mock_acl_graph_entry = MagicMock()
        self.mock_acl_graph_entry.aclgraph = None
        self.mock_acl_graph_entry.input_addresses = None
        self.mock_acl_graph_entry.output = None
        self.mock_vllm_ascend.compilation.acl_graph.ACLGraphEntry.return_value = self.mock_acl_graph_entry

        self.mock_graph = MagicMock()
        self.mock_torch_npu.NPUGraph.return_value = self.mock_graph

        self.mock_tensor = self.MockTensor()
        self.mock_tensor.data_ptr.return_value = 12345

    def test_call_with_none_mode(self):
        self.mock_forward_context.cudagraph_runtime_mode = self.mock_vllm.config.CUDAGraphMode.NONE

        result = self.patch_acl_graph.__call__(self.mock_self, "arg1", "arg2", kwarg1="value1")

        self.mock_self.runnable.assert_called_once_with("arg1", "arg2", kwarg1="value1")
        self.assertEqual(result, "mock_output")
        self.mock_vllm_ascend.compilation.acl_graph.ACLGraphEntry.assert_not_called()
        self.mock_torch_npu.NPUGraph.assert_not_called()

    def test_call_with_mismatched_mode(self):
        self.mock_forward_context.cudagraph_runtime_mode = self.mock_vllm.config.CUDAGraphMode.FULL

        result = self.patch_acl_graph.__call__(self.mock_self, "arg1", "arg2")

        self.mock_self.runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "mock_output")
        self.mock_vllm_ascend.compilation.acl_graph.ACLGraphEntry.assert_not_called()
        self.mock_torch_npu.NPUGraph.assert_not_called()

    def test_call_with_new_batch_descriptor(self):
        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.mock_vllm_ascend.compilation.acl_graph.ACLGraphEntry.assert_called_once_with(
            batch_descriptor=self.mock_batch_descriptor
        )

        self.mock_torch_npu.NPUGraph.assert_called_once()
        self.mock_graph.capture_begin.assert_called_once_with()
        self.mock_graph.capture_end.assert_called_once()

        self.mock_self.runnable.assert_called_once_with(self.mock_tensor)

        self.assertEqual(self.mock_acl_graph_entry.input_addresses, [12345])

        self.mock_vllm.utils.weak_ref_tensors.assert_any_call("mock_output")
        self.assertEqual(self.mock_acl_graph_entry.output, "weak_ref_output")

        self.assertEqual(self.mock_compilation_counter.num_cudagraph_captured, 1)

        self.assertEqual(result, "mock_output")

    def test_call_with_existing_aclgraph(self):
        self.mock_acl_graph_entry.aclgraph = self.mock_graph
        self.mock_acl_graph_entry.output = "cached_output"
        self.mock_acl_graph_entry.input_addresses = []
        self.mock_self.concrete_aclgraph_entries[self.mock_batch_descriptor] = self.mock_acl_graph_entry

        result = self.patch_acl_graph.__call__(self.mock_self, "arg1")

        self.mock_graph.replay.assert_called_once()

        self.mock_self.runnable.assert_not_called()

        self.assertEqual(result, "cached_output")

    def test_call_with_debug_mode_and_matching_addresses(self):
        self.mock_self.is_debugging_mode = True
        self.mock_acl_graph_entry.aclgraph = self.mock_graph
        self.mock_acl_graph_entry.output = "cached_output"
        self.mock_acl_graph_entry.input_addresses = [12345]
        self.mock_self.concrete_aclgraph_entries[self.mock_batch_descriptor] = self.mock_acl_graph_entry

        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.mock_graph.replay.assert_called_once()

        self.assertEqual(result, "cached_output")

    def test_call_with_debug_mode_and_mismatched_addresses(self):
        self.mock_self.is_debugging_mode = True
        self.mock_acl_graph_entry.aclgraph = self.mock_graph
        self.mock_acl_graph_entry.output = "cached_output"
        self.mock_acl_graph_entry.input_addresses = [54321]
        self.mock_self.concrete_aclgraph_entries[self.mock_batch_descriptor] = self.mock_acl_graph_entry

        with self.assertRaises(RuntimeError):
            self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

    def test_call_with_weak_ref_output(self):
        self.mock_self.aclgraph_options.weak_ref_output = True

        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.assertEqual(self.mock_vllm.utils.weak_ref_tensors.call_count, 2)

        self.assertEqual(result, "weak_ref_output")

    def test_call_with_gc_disable(self):
        self.mock_self.aclgraph_options.gc_disable = True

        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.mock_self.runnable.assert_called_once_with(self.mock_tensor)
        self.assertEqual(result, "mock_output")

    def test_call_with_graph_pool(self):
        self.mock_self.graph_pool = "mock_graph_pool"

        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.mock_graph.capture_begin.assert_called_once_with("mock_graph_pool")
        self.assertEqual(result, "mock_output")

    def test_call_with_debug_log_enabled(self):
        self.mock_self.aclgraph_options.debug_log_enable = True

        self.mock_self.concrete_aclgraph_entries.clear()

        result = self.patch_acl_graph.__call__(self.mock_self, self.mock_tensor)

        self.mock_vllm.logger.debug.assert_called_once()

        call_args = self.mock_vllm.logger.debug.call_args
        self.assertIn("Capturing a aclgraph on", str(call_args))
        self.assertEqual(result, "mock_output")


if __name__ == '__main__':
    unittest.main()
