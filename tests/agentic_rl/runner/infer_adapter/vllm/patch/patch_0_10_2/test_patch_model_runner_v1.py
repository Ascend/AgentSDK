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

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib
import importlib.util

os.environ['ENABLE_VLLM_STAT'] = 'true'
os.environ['PROFILING_FORWARD'] = '1'
os.environ['PROFILING_SAMPLE_PROB'] = '100'


class TestPatchModelRunnerV1(unittest.TestCase):
    """Test patch_model_runner_v1.py module"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for the entire test class"""
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment for the entire test class"""
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        """Setup mock objects for torch, numpy, vllm, vllm_ascend"""
        cls.mock_torch = MagicMock()

        def mock_inference_mode(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            else:
                def decorator(func):
                    return func
                return decorator

        cls.mock_torch.inference_mode = mock_inference_mode
        cls.mock_torch._dynamo = MagicMock()
        cls.mock_torch._dynamo.cache_size = MagicMock()
        cls.mock_torch.distributed = MagicMock()

        cls.mock_numpy = MagicMock()

        cls.mock_vllm = MagicMock()

        cls.mock_vllm_config_module = MagicMock()
        cls.mock_vllm_config_module.CompilationLevel = MagicMock()
        cls.mock_vllm_config_module.CUDAGraphMode = MagicMock()
        cls.mock_vllm_config_module.VllmConfig = MagicMock()

        cls.mock_parallel_state = MagicMock()
        cls.mock_dp_group = MagicMock()
        cls.mock_dp_group.cpu_group = MagicMock()
        cls.mock_dp_group.device_group = MagicMock()
        cls.mock_parallel_state.get_dp_group.return_value = cls.mock_dp_group
        cls.mock_pp_group = MagicMock()
        cls.mock_pp_group.is_first_rank = True
        cls.mock_pp_group.is_last_rank = True
        cls.mock_parallel_state.get_pp_group.return_value = cls.mock_pp_group
        cls.mock_parallel_state.get_tp_group.return_value = MagicMock()
        cls.mock_parallel_state.is_global_first_rank.return_value = True

        cls.mock_vllm_utils = MagicMock()
        cls.mock_vllm_utils.cdiv.return_value = 1

        cls.mock_vllm_sequence = MagicMock()

        cls.mock_vllm_outputs = MagicMock()
        cls.mock_vllm_outputs.EMPTY_MODEL_RUNNER_OUTPUT = MagicMock()
        cls.mock_vllm_outputs.ModelRunnerOutput = MagicMock()

        cls.mock_vllm_kv_connector = MagicMock()
        cls.mock_vllm_kv_connector.KVConnectorOutput = MagicMock()

        cls.mock_vllm_kv_transfer = MagicMock()
        cls.mock_vllm_kv_transfer.has_kv_transfer_group.return_value = False
        cls.mock_vllm_kv_transfer.get_kv_transfer_group.return_value = MagicMock()

        cls.mock_vllm_forward_context = MagicMock()
        cls.mock_vllm_forward_context.BatchDescriptor = MagicMock()

        cls.mock_vllm_logger = MagicMock()

        cls.mock_vllm_ascend = MagicMock()

        cls.mock_vllm_ascend_utils = MagicMock()
        cls.mock_vllm_ascend_utils.vllm_version_is.return_value = False
        cls.mock_vllm_ascend_utils.lmhead_tp_enable.return_value = False

        cls.mock_vllm_ascend_forward_context = MagicMock()
        cls.mock_vllm_ascend_forward_context.set_ascend_forward_context = MagicMock(return_value=MagicMock())

        cls.mock_vllm_ascend_attention = MagicMock()
        cls.mock_vllm_ascend_attention.AscendAttentionState = MagicMock()
        cls.mock_vllm_ascend_attention.AscendAttentionState.DecodeOnly = "DecodeOnly"
        cls.mock_vllm_ascend_attention.AscendAttentionState.Prefill = "Prefill"

        cls.mock_vllm_ascend_mtp_proposer = MagicMock()
        cls.mock_vllm_ascend_eagle_proposer = MagicMock()

        class MockNPUModelRunner:
            def __init__(self, *args, **kwargs):
                self.original_init_called = True

            def _sync_metadata_across_dp(self, *args, **kwargs):
                return (args[0], None, args[1], args[2])

            def execute_model(self, *args, **kwargs):
                return self.mock_vllm_outputs.EMPTY_MODEL_RUNNER_OUTPUT

            def dummy_run(self, *args, **kwargs):
                return self.mock_torch.tensor([1.0])

            def _update_states(self, *args, **kwargs):
                pass

            def _prepare_inputs(self, *args, **kwargs):
                pass

            def _generate_process_reqs_hidden_states(self, *args, **kwargs):
                pass

            def _generate_dummy_run_hidden_states(self, *args, **kwargs):
                return self.mock_torch.tensor([1.0])

            def _build_attention_metadata(self, *args, **kwargs):
                return MagicMock()

            def _select_moe_comm_method(self, *args, **kwargs):
                return "all_reduce"

            def maybe_dummy_run_with_lora(self, *args, **kwargs):
                return MagicMock()

        MockNPUModelRunner.mock_vllm_outputs = cls.mock_vllm_outputs
        MockNPUModelRunner.mock_torch = cls.mock_torch

        cls.mock_vllm_ascend_model_runner = MagicMock()
        cls.mock_vllm_ascend_model_runner.NPUModelRunner = MockNPUModelRunner

        cls.mock_vllm_execute_stat = MagicMock()
        cls.mock_vllm_execute_stat.StatTimeUtil = MagicMock
        cls.mock_vllm_execute_stat.vllm_output_statics = MagicMock()
        cls.mock_vllm_execute_stat.StatPhase = MagicMock()
        cls.mock_vllm_execute_stat.vllm_output_statics.set_cur_requestid_stepid = MagicMock()
        cls.mock_vllm_execute_stat.vllm_output_statics.add_stat = MagicMock()
        cls.mock_vllm_execute_stat.vllm_output_statics.set_stat = MagicMock()
        cls.mock_vllm_execute_stat.vllm_output_statics.set_step_finish_time = MagicMock()

        cls.mock_npu_model_profiling = MagicMock()
        cls.mock_npu_model_profiling.run_model_with_profiling = MagicMock(return_value=cls.mock_torch.tensor([1.0]))

        cls.mock_datetime = MagicMock()

        cls.mock_profile_execution = MagicMock()
        cls.mock_profile_execution.capture_async.return_value = MagicMock()
        cls.mock_profile_execution.pop_captured_sync.return_value = {}

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch._dynamo': cls.mock_torch._dynamo,
            'torch._dynamo.cache_size': cls.mock_torch._dynamo.cache_size,
            'torch.distributed': cls.mock_torch.distributed,
            'numpy': cls.mock_numpy,
            'vllm': cls.mock_vllm,
            'vllm.config': cls.mock_vllm_config_module,
            'vllm.distributed.parallel_state': cls.mock_parallel_state,
            'vllm.utils': cls.mock_vllm_utils,
            'vllm.sequence': cls.mock_vllm_sequence,
            'vllm.v1.outputs': cls.mock_vllm_outputs,
            'vllm.v1.worker.kv_connector_model_runner_mixin': cls.mock_vllm_kv_connector,
            'vllm.distributed.kv_transfer': cls.mock_vllm_kv_transfer,
            'vllm.forward_context': cls.mock_vllm_forward_context,
            'vllm.logger': cls.mock_vllm_logger,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.utils': cls.mock_vllm_ascend_utils,
            'vllm_ascend.ascend_forward_context': cls.mock_vllm_ascend_forward_context,
            'vllm_ascend.attention.attention_v1': cls.mock_vllm_ascend_attention,
            'vllm_ascend.worker.mtp_proposer_v1': cls.mock_vllm_ascend_mtp_proposer,
            'vllm_ascend.worker.eagle_proposer_v1': cls.mock_vllm_ascend_eagle_proposer,
            'vllm_ascend.worker.model_runner_v1': cls.mock_vllm_ascend_model_runner,
            'agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat': cls.mock_vllm_execute_stat,
            'agentic_rl.runner.infer_adapter.vllm.patch.comm.npu_model_profiling': cls.mock_npu_model_profiling,
            'datetime': cls.mock_datetime,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_model_runner_v1',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_model_runner_v1.py')
        )
        cls.patch_model_runner_v1 = importlib.util.module_from_spec(spec)
        sys.modules['patch_model_runner_v1'] = cls.patch_model_runner_v1
        spec.loader.exec_module(cls.patch_model_runner_v1)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_torch.distributed.reset_mock()
        self.mock_torch.distributed.barrier.reset_mock()
        self.mock_torch.distributed.all_reduce.reset_mock()
        self.mock_vllm_kv_transfer.has_kv_transfer_group.reset_mock()
        self.mock_vllm_kv_transfer.has_kv_transfer_group.return_value = False
        self.mock_vllm_ascend_utils.vllm_version_is.reset_mock()
        self.mock_vllm_ascend_utils.vllm_version_is.side_effect = None
        self.mock_vllm_ascend_utils.vllm_version_is.return_value = False

        self.mock_pp_group.is_last_rank = True
        self.mock_pp_group.is_first_rank = True

        self.mock_self = MagicMock()
        self.mock_vllm_config = MagicMock()
        self.mock_device = MagicMock()
        self.mock_scheduler_output = MagicMock()

        self.mock_self.dp_size = 1
        self.mock_self.dp_rank = 0
        self.mock_self.parallel_config = MagicMock()
        self.mock_self.parallel_config.tensor_parallel_size = 2
        self.mock_self.parallel_config.distributed_executor_backend = "not_external_launcher"
        self.mock_self.vllm_config = MagicMock()
        self.mock_self.model_config = MagicMock()
        self.mock_self.model_config.max_model_len = 1024
        self.mock_self.model = MagicMock()
        self.mock_self.model.return_value = self.mock_torch.tensor([1.0])
        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self.model.make_empty_intermediate_tensors = MagicMock(return_value={})
        self.mock_self.device = "npu:0"
        self.mock_self.input_batch = MagicMock()
        self.mock_self.input_batch.req_ids = ["req1", "req2"]
        self.mock_self.input_batch.num_reqs = 2
        self.mock_self.input_batch.num_tokens_no_spec = [1, 1]
        self.mock_self.input_batch.token_ids_cpu = MagicMock()
        self.mock_self.input_batch.num_tokens = [1, 1]
        self.mock_self.input_batch.req_id_to_index = {"req1": 0, "req2": 1}
        self.mock_self.input_batch.pooling_params = None
        self.mock_self.input_batch.sampling_metadata = MagicMock()
        self.mock_self.input_batch.generators = {}
        self.mock_self.sampler = MagicMock()
        self.mock_self.rejection_sampler = MagicMock()
        self.mock_self.requests = {"req1": MagicMock(), "req2": MagicMock()}
        self.mock_self.requests["req1"].output_token_ids = []
        self.mock_self.requests["req2"].output_token_ids = []
        self.mock_self.requests["req1"].num_computed_tokens = 1
        self.mock_self.requests["req2"].num_computed_tokens = 1
        self.mock_self.requests["req1"].num_tokens = 5
        self.mock_self.requests["req2"].num_tokens = 5

        mock_sampler_output = MagicMock()
        mock_sampler_output.logprobs_tensors = None
        mock_sampler_output.sampled_token_ids = self.mock_torch.tensor([[100], [200]])
        self.mock_self.sampler.return_value = mock_sampler_output

        self.mock_self.reserved_mc2_mask = None
        self.mock_self.moe_comm_method = "all_reduce"
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.with_prefill = False
        self.mock_self.is_kv_producer = False
        self.mock_self.use_aux_hidden_state_outputs = False
        self.mock_self.input_ids = self.mock_torch.tensor([1, 2, 3])
        self.mock_self.positions = self.mock_torch.tensor([1, 2, 3])
        self.mock_self.mrope_positions = self.mock_torch.tensor([[1, 2, 3]])
        self.mock_self.inputs_embeds = self.mock_torch.tensor([[1.0], [2.0], [3.0]])
        self.mock_self.scheduler_config = MagicMock()
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.intermediate_tensors = None
        self.mock_self.in_profile_run = False
        self.mock_self.speculative_config = None
        self.mock_self.use_spec_decode = False
        self.mock_self.drafter = None
        self.mock_self.lora_config = None
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.dtype = self.mock_torch.float32

        self.mock_scheduler_output.total_num_scheduled_tokens = 0
        self.mock_scheduler_output.num_scheduled_tokens = {"req1": 1, "req2": 1}
        self.mock_scheduler_output.grammar_bitmask = None

    def test_model_runner_init(self):
        """Test model_runner_init function"""
        self.patch_model_runner_v1.model_runner_init(self.mock_self, self.mock_vllm_config, self.mock_device)

        self.assertEqual(self.mock_self.mc2_tokens_capacity, 256)
        self.mock_torch.distributed.barrier.assert_called_once_with(group=self.mock_dp_group.cpu_group)
        self.assertEqual(self.mock_self.stat_step, 0)

    def test_sync_metadata_across_dp_single_dp(self):
        """Test sync_metadata_across_dp function with single DP rank"""
        self.mock_self.dp_size = 1

        result = self.patch_model_runner_v1.sync_metadata_across_dp(self.mock_self, 10, True, False)

        self.assertEqual(result, (10, None, True, False))
        self.mock_torch.distributed.all_reduce.assert_not_called()
        self.mock_torch.distributed.barrier.assert_not_called()

    def test_sync_metadata_across_dp_multiple_dp(self):
        """Test sync_metadata_across_dp function with multiple DP ranks"""
        self.mock_self.dp_size = 2
        self.mock_self.dp_rank = 0

        mock_tensor = MagicMock()
        self.mock_torch.tensor.return_value = mock_tensor
        self.mock_torch.cat.return_value = mock_tensor

        mock_max_result = MagicMock()
        mock_max_result.item.return_value = 20
        self.mock_torch.max.return_value = mock_max_result

        mock_tensor.__getitem__.return_value = mock_tensor
        mock_tensor.__bool__.return_value = True

        result = self.patch_model_runner_v1.sync_metadata_across_dp(self.mock_self, 10, True, False)

        self.assertEqual(result[0], 20)
        self.mock_torch.distributed.all_reduce.assert_called_once()
        self.mock_torch.distributed.barrier.assert_called_once_with(group=self.mock_dp_group.device_group)

    def test_execute_model_patch_empty_batch(self):
        """Test execute_model_patch function with empty batch"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 0
        self.mock_self._update_states = MagicMock()

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)

    def test_execute_model_patch_with_kv_transfer(self):
        """Test execute_model_patch function with empty batch and kv transfer"""
        self.mock_vllm_kv_transfer.has_kv_transfer_group.return_value = True

        self.mock_scheduler_output.total_num_scheduled_tokens = 0
        self.mock_self._update_states = MagicMock()
        self.mock_self.kv_connector_no_forward = MagicMock()

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self.kv_connector_no_forward.assert_called_once_with(self.mock_scheduler_output)

    def test_execute_model_patch_with_kv_connector_output(self):
        """Test execute_model_patch function with kv_connector_output"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 10
        self.mock_self._update_states = MagicMock()
        self.mock_self._prepare_inputs = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=MagicMock())
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(MagicMock(), MagicMock()))

        self.mock_parallel_state.get_pp_group().is_last_rank = True
        self.mock_self.parallel_config.distributed_executor_backend = "not_external_launcher"

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self.get_finished_kv_transfer.assert_called_once_with(self.mock_scheduler_output)

    def test_execute_model_patch_use_aux_hidden_states(self):
        """Test execute_model_patch function with use_aux_hidden_state_outputs=True"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 10
        self.mock_self._update_states = MagicMock()

        mock_attn_metadata = MagicMock()
        mock_attn_metadata.attn_state = MagicMock()
        mock_attn_metadata.num_actual_tokens = 10
        mock_attn_metadata.seq_lens = MagicMock()
        mock_attn_metadata.seq_lens.shape = [2]
        mock_attn_metadata.seq_lens.tolist.return_value = [5, 5]

        self.mock_self._prepare_inputs = MagicMock(return_value=(mock_attn_metadata, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()

        mock_hidden_states = MagicMock()
        mock_aux_hidden_states = MagicMock()
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=(mock_hidden_states, mock_aux_hidden_states))
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))

        self.mock_self.use_aux_hidden_state_outputs = True

        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.assertEqual(self.mock_self.use_aux_hidden_state_outputs, True)

    def test_dummy_run_with_stat(self):
        """Test dummy_run_with_stat function"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), True, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_uniform_decode(self):
        """Test dummy_run_with_stat function with uniform_decode=True"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), False, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 2
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=False, uniform_decode=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_uniform_decode_remainder(self):
        """Test dummy_run_with_stat function with uniform_decode=True and remainder"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(11, MagicMock(), False, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 2
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 6

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 11, with_prefill=False, uniform_decode=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_kv_producer(self):
        """Test dummy_run_with_stat function with is_kv_producer=True"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), False, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = True

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=False)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_multimodal(self):
        """Test dummy_run_with_stat function with multimodal model"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), True, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = True
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_mrope(self):
        """Test dummy_run_with_stat function with mrope positions"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), False, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = True
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()
        self.mock_self.model = MagicMock()
        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = True
        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=False)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_with_stat_non_first_pp_rank(self):
        """Test dummy_run_with_stat function with non-first PP rank"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), True, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()

        mock_empty_intermediate_tensors = {"test": MagicMock()}
        self.mock_self.model = MagicMock()
        self.mock_self.model.make_empty_intermediate_tensors.return_value = mock_empty_intermediate_tensors

        self.mock_self.intermediate_tensors = None

        self.mock_parallel_state.get_pp_group().is_first_rank = False
        self.mock_parallel_state.get_pp_group().is_last_rank = False

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()
        self.mock_self.model.make_empty_intermediate_tensors.assert_called_once()

    def test_dummy_run_with_stat_non_first_pp_rank_existing_tensors(self):
        """Test dummy_run_with_stat function with non-first PP rank and existing intermediate_tensors"""
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, MagicMock(), True, False))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self.maybe_dummy_run_with_lora = MagicMock(return_value=MagicMock())
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.is_kv_producer = False

        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.mrope_positions = MagicMock()
        self.mock_self.inputs_embeds = MagicMock()

        self.mock_self.model = MagicMock()

        self.mock_self.intermediate_tensors = {"test": MagicMock()}

        self.mock_parallel_state.get_pp_group().is_first_rank = False
        self.mock_parallel_state.get_pp_group().is_last_rank = False

        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.lora_config = None
        self.mock_self.dtype = self.mock_torch.float32
        self.mock_self.device = "npu:0"

        self.mock_vllm_utils.cdiv.return_value = 5

        self.patch_model_runner_v1.dummy_run_with_stat(self.mock_self, 10, with_prefill=True)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()
        self.mock_self.model.make_empty_intermediate_tensors.assert_not_called()

    def test_execute_model_patch_pooling_v010(self):
        """Test execute_model_patch function with pooling for v0.10.1 version"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 10
        self.mock_self._update_states = MagicMock()
        self.mock_self._prepare_inputs = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=MagicMock())
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))

        self.mock_self.input_batch.pooling_params = MagicMock()

        self.mock_vllm_ascend_utils.vllm_version_is.side_effect = lambda v: v in ["0.10.1.1", "0.10.1"]

        self.mock_parallel_state.get_pp_group().is_last_rank = True
        self.mock_self.parallel_config.distributed_executor_backend = "not_external_launcher"

        self.mock_self._pool_v010 = MagicMock()

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._pool_v010.assert_called_once()

    def test_execute_model_patch_pooling_non_v010(self):
        """Test execute_model_patch function with pooling for non-v0.10.1 version"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 10
        self.mock_self._update_states = MagicMock()
        self.mock_self._prepare_inputs = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=MagicMock())
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))

        self.mock_self.input_batch.pooling_params = MagicMock()

        self.mock_vllm_ascend_utils.vllm_version_is.return_value = False

        self.mock_parallel_state.get_pp_group().is_last_rank = True
        self.mock_self.parallel_config.distributed_executor_backend = "not_external_launcher"

        self.mock_self._pool = MagicMock()

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._pool.assert_called_once()

    def test_execute_model_patch_normal(self):
        """Test execute_model_patch function with normal execution"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_self._update_states = MagicMock()
        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            None,
            MagicMock(),
            None,
            None
        ))
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self._get_prompt_logprobs_dict = MagicMock(return_value={})
        self.mock_self.rejection_sampler.parse_output = MagicMock(return_value=[[100], [200]])
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"
        self.mock_self._draft_token_ids = None

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_self._prepare_inputs.assert_called_once()
        self.mock_self._generate_process_reqs_hidden_states.assert_called_once()
        self.mock_self.model.compute_logits.assert_called_once()
        self.mock_self.sampler.assert_called_once()

    def test_execute_model_patch_spec_decode(self):
        """Test execute_model_patch function with speculative decode"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_self._update_states = MagicMock()

        mock_spec_decode_metadata = MagicMock()
        mock_spec_decode_metadata.bonus_logits_indices = [0]
        mock_spec_decode_metadata.target_logits_indices = [1]
        mock_spec_decode_metadata.logits_indices = [0, 1]

        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            mock_spec_decode_metadata,
            MagicMock(),
            None,
            None
        ))
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=self.mock_torch.tensor([[1.0], [2.0]]))
        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0], [2.0]]))
        self.mock_self._get_prompt_logprobs_dict = MagicMock(return_value={})
        self.mock_self.rejection_sampler.parse_output = MagicMock(return_value=[[100], [200]])
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.rejection_sampler = MagicMock(return_value=self.mock_torch.tensor([400]))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"
        self.mock_self._draft_token_ids = None

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_self._prepare_inputs.assert_called_once()
        self.mock_self._generate_process_reqs_hidden_states.assert_called_once()
        self.mock_self.model.compute_logits.assert_called_once()
        self.mock_self.sampler.assert_called_once()
        self.mock_self.rejection_sampler.assert_called_once()

    def test_execute_model_patch_with_aux_hidden_states(self):
        """Test execute_model_patch function with aux hidden states"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_self._update_states = MagicMock()
        self.mock_self.use_aux_hidden_state_outputs = True

        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            None,
            MagicMock(),
            None,
            None
        ))

        mock_hidden_states = MagicMock()
        mock_aux_hidden_states = MagicMock()
        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=(mock_hidden_states, mock_aux_hidden_states))

        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self._get_prompt_logprobs_dict = MagicMock(return_value={})
        self.mock_self.rejection_sampler.parse_output = MagicMock(return_value=[[100], [200]])
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"
        self.mock_self._draft_token_ids = None

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_self._generate_process_reqs_hidden_states.assert_called_once()
        self.mock_self.model.compute_logits.assert_called_once_with(mock_hidden_states[MagicMock()], None)

    def test_execute_model_patch_with_broadcast_pp(self):
        """Test execute_model_patch function with broadcast pp output"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_self._update_states = MagicMock()

        self.mock_self.parallel_config.distributed_executor_backend = "external_launcher"
        self.mock_pp_group.ranks = [0, 1]

        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            None,
            MagicMock(),
            None,
            None
        ))

        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self._get_prompt_logprobs_dict = MagicMock(return_value={})
        self.mock_self.rejection_sampler.parse_output = MagicMock(return_value=[[100], [200]])
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"
        self.mock_self._draft_token_ids = None

        mock_broadcast_data = {"logits": self.mock_torch.tensor([[1.0]])}
        self.mock_pp_group.broadcast_tensor_dict.return_value = mock_broadcast_data

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_pp_group.broadcast_tensor_dict.assert_called_once()

    def test_execute_model_patch_with_pooling_params(self):
        """Test execute_model_patch function with pooling_params"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_self._update_states = MagicMock()

        self.mock_self.input_batch.pooling_params = MagicMock()
        self.mock_self._pool = MagicMock(return_value=self.mock_vllm_outputs.ModelRunnerOutput())

        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            None,
            MagicMock(),
            None,
            None
        ))

        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"

        self.mock_parallel_state.get_pp_group().is_last_rank = True

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_self._pool.assert_called_once()

    def test_execute_model_patch_with_grammar_bitmask(self):
        """Test execute_model_patch function with grammar bitmask"""
        self.mock_scheduler_output.total_num_scheduled_tokens = 2
        self.mock_scheduler_output.grammar_bitmask = MagicMock()
        self.mock_self._update_states = MagicMock()
        self.mock_self.apply_grammar_bitmask = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))

        self.mock_self._prepare_inputs = MagicMock(return_value=(
            MagicMock(attn_state="DecodeOnly", num_actual_tokens=2, seq_lens=self.mock_torch.tensor([1, 1])),
            MagicMock(),
            self.mock_numpy.array([1, 1]),
            2,
            None,
            None,
            MagicMock(),
            None,
            MagicMock(),
            None,
            None
        ))

        self.mock_self._generate_process_reqs_hidden_states = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self.model.compute_logits = MagicMock(return_value=self.mock_torch.tensor([[1.0]]))
        self.mock_self._get_prompt_logprobs_dict = MagicMock(return_value={})
        self.mock_self.rejection_sampler.parse_output = MagicMock(return_value=[[100], [200]])
        self.mock_self.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_setup_kv_connector = MagicMock()
        self.mock_self.maybe_wait_for_kv_save = MagicMock()
        self.mock_self.attn_state = "DecodeOnly"
        self.mock_self._draft_token_ids = None

        self.patch_model_runner_v1.execute_model_patch(self.mock_self, self.mock_scheduler_output)

        self.mock_self._update_states.assert_called_once_with(self.mock_scheduler_output)
        self.mock_self.apply_grammar_bitmask.assert_called_once_with(self.mock_scheduler_output, self.mock_torch.tensor([[1.0]]))

    def test_dummy_run_basic(self):
        """Test dummy_run function with basic configuration"""
        self.mock_self.dp_size = 1
        self.mock_self.with_prefill = False
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, None, False, False))
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_dummy_run_with_lora = MagicMock()
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.scheduler_config = MagicMock()
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.intermediate_tensors = None
        self.mock_self.model.make_empty_intermediate_tensors = MagicMock(return_value={})
        self.mock_self._generate_dummy_run_hidden_states = MagicMock(return_value=self.mock_torch.tensor([1.0]))

        self.patch_model_runner_v1.dummy_run(self.mock_self, 10)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._select_moe_comm_method.assert_called_once_with(10)
        self.mock_self._build_attention_metadata.assert_called_once()

    def test_dummy_run_kv_producer(self):
        """Test dummy_run function when node is a KV producer"""
        self.mock_self.dp_size = 1
        self.mock_self.with_prefill = False
        self.mock_self.is_kv_producer = True
        self.mock_self._sync_metadata_across_dp = MagicMock(return_value=(10, None, True, False))
        self.mock_self._build_attention_metadata = MagicMock()
        self.mock_self._select_moe_comm_method = MagicMock(return_value="all_reduce")
        self.mock_self.aclgraph_dispatcher = MagicMock()
        self.mock_self.aclgraph_dispatcher.dispatch = MagicMock(return_value=(None, MagicMock()))
        self.mock_self.maybe_dummy_run_with_lora = MagicMock()
        self.mock_self.is_multimodal_model = False
        self.mock_self.uses_mrope = False
        self.mock_self.input_ids = MagicMock()
        self.mock_self.positions = MagicMock()
        self.mock_self.scheduler_config = MagicMock()
        self.mock_self.scheduler_config.max_num_batched_tokens = 1024
        self.mock_self.scheduler_config.max_num_seqs = 10
        self.mock_self.decode_token_per_req = 1
        self.mock_self.uniform_decode_query_len = 1
        self.mock_self.intermediate_tensors = None
        self.mock_self.model.make_empty_intermediate_tensors = MagicMock(return_value={})
        self.mock_self._generate_dummy_run_hidden_states = MagicMock(return_value=self.mock_torch.tensor([1.0]))

        self.patch_model_runner_v1.dummy_run(self.mock_self, 10)

        self.mock_self._sync_metadata_across_dp.assert_called_once()
        self.mock_self._build_attention_metadata.assert_called_once()


if __name__ == '__main__':
    unittest.main()
