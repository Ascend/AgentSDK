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

import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock


class TestNPUModeLProfiling(unittest.TestCase):
    """Test NPU model profiling functionality"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = Mock()
        cls.mock_torch_npu = Mock()

        cls.mock_torch.randn = Mock(return_value="mock_tensor")
        cls.mock_torch.randint = Mock(return_value="mock_input_ids")
        cls.mock_torch.arange = Mock(return_value="mock_positions")
        cls.mock_torch.unsqueeze = Mock(return_value="mock_unsqueezed")
        cls.mock_torch.npu = Mock()
        cls.mock_torch.npu.synchronize = Mock()

        cls._configure_torch_npu_profiler(cls.mock_torch_npu)

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch_npu': cls.mock_torch_npu,
        })
        cls.modules_patcher.start()

    @classmethod
    def _configure_torch_npu_profiler(cls, mock_torch_npu):
        mock_torch_npu.profiler = Mock()
        mock_torch_npu.profiler._ExperimentalConfig = Mock()
        mock_torch_npu.profiler.ProfilerLevel = Mock()
        mock_torch_npu.profiler.ProfilerLevel.Level2 = "level2"
        mock_torch_npu.profiler.AiCMetrics = Mock()
        mock_torch_npu.profiler.AiCMetrics.PipeUtilization = "pipe_utilization"
        mock_torch_npu.profiler.ExportType = Mock()
        mock_torch_npu.profiler.ExportType.Text = "text"
        mock_torch_npu.profiler.ProfilerActivity = Mock()
        mock_torch_npu.profiler.ProfilerActivity.NPU = "npu"
        mock_torch_npu.profiler.ProfilerActivity.CPU = "cpu"
        mock_torch_npu.profiler.profile = Mock()
        mock_torch_npu.profiler.tensorboard_trace_handler = Mock()
        mock_torch_npu.profiler.schedule = Mock()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'npu_model_profiling.py')
        spec = importlib.util.spec_from_file_location('npu_model_profiling', module_path)
        cls.npu_model_profiling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.npu_model_profiling)

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.original_profiling_path = os.environ.get("PROFILING_SAVE_PATH")
        os.environ["PROFILING_SAVE_PATH"] = "./test_prof"

        self.mock_torch_npu.profiler.profile.reset_mock()
        self.mock_torch_npu.profiler.tensorboard_trace_handler.reset_mock()
        self.mock_torch_npu.profiler.schedule.reset_mock()
        self.mock_torch_npu.profiler._ExperimentalConfig.reset_mock()

    def tearDown(self):
        if self.original_profiling_path is None:
            if "PROFILING_SAVE_PATH" in os.environ:
                del os.environ["PROFILING_SAVE_PATH"]
        else:
            os.environ["PROFILING_SAVE_PATH"] = self.original_profiling_path

    def _setup_profiler_mock(self):
        mock_profiler = MagicMock()
        self.mock_torch_npu.profiler.profile.return_value = mock_profiler
        self.mock_torch_npu.profiler.tensorboard_trace_handler = MagicMock()
        self.mock_torch_npu.profiler.schedule = MagicMock()
        self.mock_torch_npu.profiler._ExperimentalConfig = MagicMock()
        self.mock_torch_npu.profiler.ProfilerLevel.Level2 = "level2"
        self.mock_torch_npu.profiler.AiCMetrics.PipeUtilization = "pipe_utilization"
        self.mock_torch_npu.profiler.ExportType.Text = "text"
        self.mock_torch_npu.profiler.ProfilerActivity.NPU = "npu"
        self.mock_torch_npu.profiler.ProfilerActivity.CPU = "cpu"
        return mock_profiler

    def test_run_model_with_profiling_basic(self):
        """Test basic functionality: model execution and profiling"""
        mock_model = MagicMock()
        mock_hidden_states = "mock_hidden_states"
        mock_model.return_value = mock_hidden_states

        input_ids = "mock_input_ids"
        positions = "mock_positions"
        intermediate_tensors = {}
        inputs_embeds = "mock_inputs_embeds"
        stat_step = 5
        process_name = "test_process"

        mock_profiler = self._setup_profiler_mock()

        result = self.npu_model_profiling.run_model_with_profiling(
            mock_model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            stat_step,
            process_name
        )

        self.assertEqual(result, mock_hidden_states)

        mock_model.assert_called_once_with(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )

        self.mock_torch_npu.profiler.profile.assert_called_once()
        mock_profiler.start.assert_called_once()
        mock_profiler.step.assert_called_once()

    def test_process_name_cleaning(self):
        """Test process_name cleaning functionality"""
        mock_model = MagicMock()
        mock_model.return_value = "mock_hidden_states"

        input_ids = "mock_input_ids"
        positions = "mock_positions"
        intermediate_tensors = {}
        inputs_embeds = "mock_inputs_embeds"
        stat_step = 1
        process_name = "test-process@123#abc"

        self.npu_model_profiling.run_model_with_profiling(
            mock_model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            stat_step,
            process_name
        )

        expected_process_name = "test_process_123_abc_step_1"
        self.mock_torch_npu.profiler.tensorboard_trace_handler.assert_called_once()
        call_args = self.mock_torch_npu.profiler.tensorboard_trace_handler.call_args
        worker_name = call_args.kwargs.get('worker_name', '')
        self.assertEqual(worker_name, expected_process_name)

    def test_profiler_configuration(self):
        """Test profiler configuration parameters"""
        mock_model = MagicMock()
        mock_model.return_value = "mock_hidden_states"

        input_ids = "mock_input_ids"
        positions = "mock_positions"
        intermediate_tensors = {}
        inputs_embeds = "mock_inputs_embeds"
        stat_step = 1
        process_name = "test"

        self.npu_model_profiling.run_model_with_profiling(
            mock_model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            stat_step,
            process_name
        )

        self.mock_torch_npu.profiler._ExperimentalConfig.assert_called_once_with(
            profiler_level="level2",
            aic_metrics="pipe_utilization",
            export_type="text"
        )

        self.mock_torch_npu.profiler.profile.assert_called_once()
        profile_args = self.mock_torch_npu.profiler.profile.call_args
        self.assertEqual(profile_args.kwargs['activities'], ["npu", "cpu"])
        self.assertEqual(profile_args.kwargs['with_stack'], False)
        self.assertEqual(profile_args.kwargs['record_shapes'], True)
        self.assertEqual(profile_args.kwargs['profile_memory'], False)

        self.mock_torch_npu.profiler.schedule.assert_called_once_with(
            wait=0, warmup=0, active=1, repeat=1, skip_first=0
        )

    def test_model_with_intermediate_tensors(self):
        """Test model call with intermediate_tensors"""
        mock_model = MagicMock()
        mock_model.return_value = "mock_hidden_states"

        input_ids = "mock_input_ids"
        positions = "mock_positions"
        intermediate_tensors = {
            'layer_1': "mock_layer_1_tensor",
            'layer_2': "mock_layer_2_tensor"
        }
        inputs_embeds = "mock_inputs_embeds"
        stat_step = 1
        process_name = "test"

        self.npu_model_profiling.run_model_with_profiling(
            mock_model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            stat_step,
            process_name
        )

        mock_model.assert_called_once()
        call_args = mock_model.call_args
        self.assertEqual(call_args.kwargs['intermediate_tensors'], intermediate_tensors)


if __name__ == '__main__':
    unittest.main()
