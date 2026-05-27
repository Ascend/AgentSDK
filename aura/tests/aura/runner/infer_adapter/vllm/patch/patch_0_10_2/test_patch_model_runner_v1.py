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
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


class DummyTensor:
    def __init__(self, value=None):
        self.value = value if value is not None else []
        self.shape = [1]

    def item(self):
        return self.value if not isinstance(self.value, list) else 1

    def tolist(self):
        return self.value if isinstance(self.value, list) else [self.value]

    def tolists(self):
        return self.tolist()

    def copy_(self, *args, **kwargs):
        return self

    def zero_(self):
        return self

    def fill_(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def flatten(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return DummyTensor()

    def __setitem__(self, key, value):
        pass

    def __len__(self):
        return len(self.value) if isinstance(self.value, list) else 1


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_model_runner_env():
    # ---- Fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.int32 = "int32"
    fake_torch.int64 = "int64"
    fake_torch.Tensor = MagicMock
    fake_torch.device = MagicMock
    fake_torch.tensor = MagicMock(side_effect=lambda *a, **k: DummyTensor())
    fake_torch.cat = MagicMock(side_effect=lambda *a, **k: DummyTensor())
    fake_torch.max = MagicMock(return_value=DummyTensor(10))
    fake_torch.zeros = MagicMock(side_effect=lambda *a, **k: DummyTensor())
    fake_torch.from_numpy = MagicMock(side_effect=lambda x: DummyTensor(x))
    fake_torch.index_select = MagicMock()

    def inference_mode_factory(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    fake_torch.inference_mode = inference_mode_factory

    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.functional = MagicMock()
    fake_torch.Generator = MagicMock

    fake_torch.distributed = types.ModuleType("torch.distributed")
    fake_torch.distributed.all_reduce = MagicMock()
    fake_torch.distributed.barrier = MagicMock()

    fake_torch_dynamo = types.ModuleType("torch._dynamo")
    fake_torch_dynamo.__path__ = []
    fake_torch_dynamo_cache_size = types.ModuleType("torch._dynamo.cache_size")
    fake_torch._dynamo = fake_torch_dynamo

    # ---- Fake numpy ----
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = MagicMock
    fake_numpy.int32 = "int32"
    fake_numpy.array = MagicMock(return_value=MagicMock())
    fake_numpy.empty = MagicMock(return_value=MagicMock())
    fake_numpy.cumsum = MagicMock(return_value=MagicMock())
    fake_numpy.repeat = MagicMock(return_value=MagicMock())
    fake_numpy.add = MagicMock()
    fake_numpy.where = MagicMock()
    fake_numpy.all = MagicMock(return_value=True)

    # ---- Fake vllm ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_config = types.ModuleType("vllm.config")
    class CUDAGraphMode:
        NONE = 0
        PIECEWISE = 1
        FULL = 2
    fake_vllm_config.CUDAGraphMode = CUDAGraphMode
    fake_vllm_config.VllmConfig = MagicMock
    fake_vllm_sequence = types.ModuleType("vllm.sequence")
    class FakeIntermediateTensors:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    fake_vllm_sequence.IntermediateTensors = FakeIntermediateTensors
    fake_vllm_v1 = types.ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_v1_outputs = types.ModuleType("vllm.v1.outputs")
    fake_vllm_v1_outputs.EMPTY_MODEL_RUNNER_OUTPUT = object()
    fake_vllm_v1_outputs.ModelRunnerOutput = MagicMock
    fake_vllm_v1_worker = types.ModuleType("vllm.v1.worker")
    fake_vllm_v1_worker.__path__ = []
    fake_vllm_v1_worker_kv = types.ModuleType("vllm.v1.worker.kv_connector_model_runner_mixin")
    fake_vllm_v1_worker_kv.KVConnectorOutput = MagicMock
    fake_vllm_v1_spec_decode = types.ModuleType("vllm.v1.spec_decode.metadata")
    fake_vllm_v1_spec_decode.SpecDecodeMetadata = MagicMock
    fake_vllm_distributed = types.ModuleType("vllm.distributed")
    fake_vllm_distributed.__path__ = []
    fake_vllm_distributed_parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    fake_vllm_distributed_parallel_state.get_dp_group = MagicMock()
    fake_vllm_distributed_parallel_state.get_pp_group = MagicMock()
    fake_vllm_distributed_parallel_state.get_tp_group = MagicMock()
    fake_vllm_distributed_kv_transfer = types.ModuleType("vllm.distributed.kv_transfer")
    fake_vllm_distributed_kv_transfer.get_kv_transfer_group = MagicMock()
    fake_vllm_distributed_kv_transfer.has_kv_transfer_group = MagicMock(return_value=False)
    fake_vllm_forward_context = types.ModuleType("vllm.forward_context")
    fake_vllm_forward_context.BatchDescriptor = MagicMock
    fake_vllm_sampling_params = types.ModuleType("vllm.sampling_params")
    class FakeSamplingType:
        RANDOM_SEED = 1
        GREEDY = 0
    fake_vllm_sampling_params.SamplingType = FakeSamplingType
    fake_vllm_model_executor = types.ModuleType("vllm.model_executor.models.interfaces_base")
    fake_vllm_model_executor.VllmModelForPooling = MagicMock
    fake_vllm_model_executor_layers = types.ModuleType("vllm.model_executor.layers.rotary_embedding")
    fake_vllm_model_executor_layers.MRotaryEmbedding = MagicMock
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.logger = MagicMock()
    fake_vllm_logger.init_logger = MagicMock(return_value=MagicMock())
    fake_vllm_utils = types.ModuleType("vllm.utils")
    fake_vllm_utils.cdiv = lambda a,b: (a+b-1)//b
    fake_vllm_utils.merge_async_iterators = MagicMock()

    # ---- Fake vllm_ascend ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_utils = types.ModuleType("vllm_ascend.utils")
    class FakeProfileExecuteDuration:
        def capture_async(self, *args, **kwargs):
            return DummyContext()
        def pop_captured_sync(self):
            return {"prepare": 1.0, "forward": 2.0}
    fake_vllm_ascend_utils.ProfileExecuteDuration = FakeProfileExecuteDuration
    fake_vllm_ascend_utils.lmhead_tp_enable = MagicMock(return_value=False)
    fake_vllm_ascend_utils.vllm_version_is = MagicMock(return_value=False)
    fake_vllm_ascend_worker = types.ModuleType("vllm_ascend.worker")
    fake_vllm_ascend_worker.__path__ = []
    fake_vllm_ascend_worker_model_runner_v1 = types.ModuleType("vllm_ascend.worker.model_runner_v1")
    class NPUModelRunner:
        execute_model = None
        _dummy_run = None
        _generate_process_reqs_hidden_states = None
        _prepare_inputs = None
        _update_states = None
    fake_vllm_ascend_worker_model_runner_v1.NPUModelRunner = NPUModelRunner
    fake_vllm_ascend_worker_npu_input_batch = types.ModuleType("vllm_ascend.worker.npu_input_batch")
    fake_vllm_ascend_worker_npu_input_batch.CachedRequestState = MagicMock
    fake_vllm_ascend_ascend_forward_context = types.ModuleType("vllm_ascend.ascend_forward_context")
    fake_vllm_ascend_ascend_forward_context.set_ascend_forward_context = lambda *a, **k: DummyContext()
    fake_vllm_ascend_attention = types.ModuleType("vllm_ascend.attention.attention_v1")
    class FakeAscendAttentionState:
        DecodeOnly = 1
        SpecDecoding = 2
    fake_vllm_ascend_attention.AscendAttentionState = FakeAscendAttentionState
    fake_vllm_ascend_attention.AscendMetadata = MagicMock
    fake_vllm_ascend_attention_mla = types.ModuleType("vllm_ascend.attention.mla_v1")
    fake_vllm_ascend_attention_mla.AscendMLAMetadata = MagicMock
    fake_vllm_ascend_torchair = types.ModuleType("vllm_ascend.torchair.torchair_attention")
    fake_vllm_ascend_torchair.AscendTorchairMetadata = MagicMock
    fake_vllm_ascend_torchair_mla = types.ModuleType("vllm_ascend.torchair.torchair_mla")
    fake_vllm_ascend_torchair_mla.AscendMLATorchairMetadata = MagicMock
    fake_vllm_ascend_attention_utils = types.ModuleType("vllm_ascend.attention.utils")
    fake_vllm_ascend_attention_utils.AscendCommonAttentionMetadata = MagicMock
    fake_vllm_ascend_worker_mtp_proposer = types.ModuleType("vllm_ascend.worker.mtp_proposer_v1")
    fake_vllm_ascend_worker_eagle_proposer = types.ModuleType("vllm_ascend.worker.eagle_proposer_v1")
    class FakeMtpProposer:
        pass
    class FakeEagleProposer:
        pass
    fake_vllm_ascend_worker_mtp_proposer.MtpProposer = FakeMtpProposer
    fake_vllm_ascend_worker_eagle_proposer.EagleProposer = FakeEagleProposer

    # ---- Fake comm modules ----
    fake_comm_exec_stat = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat")
    class StatTimeUtil:
        def __init__(self):
            self.last_time = 0.0
        def get_duration(self):
            return 0.1
    class StatPhase:
        prepare_input_time = "prepare_input"
        with_prefill = "with_prefill"
        attn_state = "attn_state"
        num_actual_tokens = "num_actual_tokens"
        batch_num = "batch_num"
        seq_lens = "seq_lens"
        aclgraph_dispatcher_time = "aclgraph_dispatcher"
        forward_time = "forward"
        kvconnectoroutput_time = "kvconnectoroutput"
        post_process_compute_logits_time = "post_compute_logits"
        post_samper_logits_slice_time = "logits_slice"
        post_process_sampler_time = "sampler"
        post_process_other_time = "other"
        post_process_time = "post_process"
        pop_captured_sync_time = "pop_captured"
        prepare_copy_bt_time = "copy_bt"
        prepare_get_tokens_time = "get_tokens"
        prepare_pad_tokens_time = "pad_tokens"
        prepare_sync_meta_time = "sync_meta"
        prepare_set_lora_time = "set_lora"
        prepare_pos_cpu_time = "pos_cpu"
        prepare_mrope_time = "mrope"
        prepare_pos_npu_time = "pos_npu"
        prepare_slot_map_time = "slot_map"
        prepare_atten_mask_time = "atten_mask"
        prepare_seq_len_time = "seq_len"
        prepare_attn_meta_time = "attn_meta"
        prepare_inputids_cpu_time = "inputids_cpu"
        prepare_copy_inputids_time = "copy_inputids"
        prepare_inputsembeds_time = "inputsembeds"
        prepare_slice_inputids_time = "slice_inputids"
        prepare_update_ids_and_pos_time = "update_ids_pos"
        prepare_inter_tensors_time = "inter_tensors"
        prepare_logits_indice_time = "logits_indice"
        prepare_specdeco_meta_time = "specdeco_meta"
        prepare_lmhead_logits_indices_time = "lmhead_logits_indices"
        prepare_remove_reqs_time = "remove_reqs"
        prepare_add_reqs_time = "add_reqs"
        prepare_update_states_time = "update_states"
        prepare_other_states_time = "other_states"
        is_profiling = "is_profiling"
        is_dummy_run = "is_dummy_run"
        post_samper_sample_topk_topp_apply_time = "apply"
        post_samper_sample_topk_topp_logits_log_softmax_time = "log_softmax"
        post_samper_sample_topk_topp_probs_softmax_time = "probs_softmax"
        post_samper_sample_topk_topp_random_sample_time = "random_sample"
    fake_comm_exec_stat.StatTimeUtil = StatTimeUtil
    fake_comm_exec_stat.vllm_output_statics = MagicMock()
    fake_comm_exec_stat.StatPhase = StatPhase

    fake_comm_profiling = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm.npu_model_profiling")
    fake_comm_profiling.run_model_with_profiling = MagicMock()

    # ---- Fake os ----
    fake_os = types.ModuleType("os")
    fake_os.getenv = MagicMock(return_value="False")
    fake_os.environ = MagicMock()
    fake_os.environ.get = MagicMock(return_value="0")

    # ---- Aura packages ----
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
    fake_0_10_2_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_10_2")
    fake_0_10_2_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_10_2")]
    fake_comm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm")
    fake_comm_pkg.__path__ = []

    fakes = {
        "torch": fake_torch,
        "torch.distributed": fake_torch.distributed,
        "torch._dynamo": fake_torch_dynamo,
        "torch._dynamo.cache_size": fake_torch_dynamo_cache_size,
        "torch.nn": fake_torch.nn,
        "numpy": fake_numpy,
        "vllm": fake_vllm,
        "vllm.config": fake_vllm_config,
        "vllm.sequence": fake_vllm_sequence,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.outputs": fake_vllm_v1_outputs,
        "vllm.v1.worker": fake_vllm_v1_worker,
        "vllm.v1.worker.kv_connector_model_runner_mixin": fake_vllm_v1_worker_kv,
        "vllm.v1.spec_decode.metadata": fake_vllm_v1_spec_decode,
        "vllm.distributed": fake_vllm_distributed,
        "vllm.distributed.parallel_state": fake_vllm_distributed_parallel_state,
        "vllm.distributed.kv_transfer": fake_vllm_distributed_kv_transfer,
        "vllm.forward_context": fake_vllm_forward_context,
        "vllm.sampling_params": fake_vllm_sampling_params,
        "vllm.model_executor.models.interfaces_base": fake_vllm_model_executor,
        "vllm.model_executor.layers.rotary_embedding": fake_vllm_model_executor_layers,
        "vllm.logger": fake_vllm_logger,
        "vllm.utils": fake_vllm_utils,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.utils": fake_vllm_ascend_utils,
        "vllm_ascend.worker": fake_vllm_ascend_worker,
        "vllm_ascend.worker.model_runner_v1": fake_vllm_ascend_worker_model_runner_v1,
        "vllm_ascend.worker.npu_input_batch": fake_vllm_ascend_worker_npu_input_batch,
        "vllm_ascend.ascend_forward_context": fake_vllm_ascend_ascend_forward_context,
        "vllm_ascend.attention.attention_v1": fake_vllm_ascend_attention,
        "vllm_ascend.attention.mla_v1": fake_vllm_ascend_attention_mla,
        "vllm_ascend.torchair.torchair_attention": fake_vllm_ascend_torchair,
        "vllm_ascend.torchair.torchair_mla": fake_vllm_ascend_torchair_mla,
        "vllm_ascend.attention.utils": fake_vllm_ascend_attention_utils,
        "vllm_ascend.worker.mtp_proposer_v1": fake_vllm_ascend_worker_mtp_proposer,
        "vllm_ascend.worker.eagle_proposer_v1": fake_vllm_ascend_worker_eagle_proposer,
        "aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat": fake_comm_exec_stat,
        "aura.runner.infer_adapter.vllm.patch.comm.npu_model_profiling": fake_comm_profiling,
        "os": fake_os,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_10_2": fake_0_10_2_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm": fake_comm_pkg,
    }

    fake_vllm.__path__ = []
    fake_vllm_ascend.__path__ = []

    yield {
        "fakes": fakes,
        "NPUModelRunner": NPUModelRunner,
        "CUDAGraphMode": CUDAGraphMode,
        "vllm_output_statics": fake_comm_exec_stat.vllm_output_statics,
        "StatPhase": StatPhase,
        "vllm_version_is": fake_vllm_ascend_utils.vllm_version_is,
        "os": fake_os,
    }


def import_module(fake_model_runner_env, env_getenv="False", env_prof="0", env_prob="100", version_01011=False):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_model_runner_v1"
    if module_name in sys.modules:
        del sys.modules[module_name]

    fake_model_runner_env["os"].getenv.return_value = env_getenv
    fake_model_runner_env["os"].environ.get.side_effect = lambda k, d: {
        "PROFILING_FORWARD": env_prof,
        "PROFILING_SAMPLE_PROB": env_prob,
    }.get(k, d)

    if version_01011:
        fake_model_runner_env["vllm_version_is"].side_effect = lambda x: x in ("0.10.1.1", "0.10.1")
    else:
        fake_model_runner_env["vllm_version_is"].side_effect = lambda x: False

    fakes = fake_model_runner_env["fakes"]
    with patch.dict(sys.modules, fakes):
        import aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_model_runner_v1 as mod
    return mod


def make_self_mock(fake_env):
    CUDAGraphMode = fake_env["CUDAGraphMode"]
    self = MagicMock()
    self.device = "npu:0"
    self.vllm_config = MagicMock()
    self.vllm_config.model_config = MagicMock()
    self.vllm_config.model_config.enforce_eager = False
    self.vllm_config.model_config.use_mla = False
    self.vllm_config.model_config.max_model_len = 4096
    self.vllm_config.kv_transfer_config = None
    self.parallel_config = MagicMock()
    self.parallel_config.distributed_executor_backend = "torchrun"
    self.scheduler_config = MagicMock()
    self.scheduler_config.max_num_seqs = 100
    self.scheduler_config.max_num_batched_tokens = 2048
    self.dp_size = 1
    self.dp_rank = 0
    self.stat_step = 0
    self.input_batch = MagicMock()
    self.input_batch.req_ids = []
    self.input_batch.req_id_to_index = {}
    self.input_batch.num_reqs = 0
    self.requests = {}
    self.model = MagicMock()
    self.sampler = MagicMock()
    self.rejection_sampler = MagicMock()
    self.speculative_config = None
    self.drafter = None
    self.attn_state = None
    self.use_aux_hidden_state_outputs = False
    self.is_kv_producer = False
    self.lora_config = None
    self.is_multimodal_model = False
    self.uses_mrope = False
    self.in_profile_run = False
    self.reserved_mc2_mask = None
    self.graph_pad_size = -1
    self.decode_token_per_req = 1
    self.uniform_decode_query_len = 1
    self.mc2_tokens_capacity = 256
    self.attn_metadata_builder = MagicMock()
    self.aclgraph_dispatcher = MagicMock()
    self.aclgraph_dispatcher.dispatch.return_value = (CUDAGraphMode.NONE, MagicMock())
    return self


class TestModelRunnerInit:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def test_init_calls_original_and_sets_fields(self):
        NPUModelRunner = self.env["NPUModelRunner"]
        original_init = self.mod.original_model_runner_init
        original_init_mock = MagicMock()
        self.mod.original_model_runner_init = original_init_mock

        self_mock = make_self_mock(self.env)
        vllm_config = MagicMock()
        device = MagicMock()
        self.mod.model_runner_init(self_mock, vllm_config, device)

        original_init_mock.assert_called_once_with(self_mock, vllm_config, device)
        assert self_mock.mc2_tokens_capacity == 256
        assert self_mock.stat_step == 0
        self.env["fakes"]["torch.distributed"].barrier.assert_called()


class TestSyncMetadataAcrossDP:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def test_dp_size_1(self):
        self_mock = make_self_mock(self.env)
        self_mock.dp_size = 1
        res = self.mod.sync_metadata_across_dp(self_mock, 10, True, True)
        assert res == (10, None, True, True)

    def test_dp_size_greater_than_1(self):
        self_mock = make_self_mock(self.env)
        self_mock.dp_size = 4
        self_mock.dp_rank = 2

        dp_group_mock = MagicMock()
        dp_group_mock.cpu_group = "cpu_group"
        self.env["fakes"]["vllm.distributed.parallel_state"].get_dp_group.return_value = dp_group_mock

        num_tokens_tensor_mock = MagicMock()
        packed_tensor_mock = MagicMock()
        flags_tensor_mock = [1, 0]

        packed_tensor_mock.__getitem__.side_effect = lambda idx: (
            num_tokens_tensor_mock if idx == slice(None, -2) else flags_tensor_mock
        )

        with patch.object(self.mod.torch, "cat", return_value=packed_tensor_mock), \
             patch.object(self.mod.torch, "max", return_value=DummyTensor(10)):
            res = self.mod.sync_metadata_across_dp(self_mock, 10, True, True)

        max_tokens, after_pad, wp, ebo = res
        assert max_tokens == 10
        assert after_pad is not None
        assert wp is True
        assert ebo is True


class TestExecuteModelPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_getenv="true")

    def test_no_scheduled_tokens_no_kv_transfer(self):
        self_mock = make_self_mock(self.env)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 0
        self.env["fakes"]["vllm.distributed.kv_transfer"].has_kv_transfer_group.return_value = False

        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result is self.env["fakes"]["vllm.v1.outputs"].EMPTY_MODEL_RUNNER_OUTPUT

    def test_no_scheduled_tokens_with_kv_transfer(self):
        self_mock = make_self_mock(self.env)
        self_mock.kv_connector_no_forward = MagicMock(return_value="kv_result")
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 0
        self.env["fakes"]["vllm.distributed.kv_transfer"].has_kv_transfer_group.return_value = True

        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result == "kv_result"

    def test_normal_execution_returns_model_runner_output(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.req_ids = ["req1"]
        self_mock.input_batch.num_reqs = 1
        self_mock.input_batch.sampling_metadata = MagicMock()
        self_mock.input_batch.token_ids_cpu = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.requests = {"req1": MagicMock()}
        self_mock.requests["req1"].num_computed_tokens = 0
        self_mock.requests["req1"].num_tokens = 1
        self_mock.requests["req1"].output_token_ids = []
        self_mock._prepare_inputs = MagicMock(return_value=(
            MagicMock(), MagicMock(), MagicMock(), 10, MagicMock(), 10,
            MagicMock(), None, MagicMock(), None, MagicMock()
        ))
        self_mock._generate_process_reqs_hidden_states = MagicMock(return_value=MagicMock())
        self_mock.model.compute_logits = MagicMock(return_value=MagicMock())
        self_mock.sampler = MagicMock()
        self_mock.sampler.return_value = MagicMock()
        self_mock.sampler.return_value.sampled_token_ids = MagicMock()
        self_mock.sampler.return_value.logprobs_tensors = None
        self_mock._get_prompt_logprobs_dict = MagicMock(return_value={})
        self_mock.apply_grammar_bitmask = MagicMock()
        self_mock._pool_v010 = MagicMock()
        self_mock._pool = MagicMock()
        self_mock.maybe_setup_kv_connector = MagicMock()
        self_mock.maybe_wait_for_kv_save = MagicMock()
        self_mock.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self_mock._select_moe_comm_method = MagicMock(return_value=None)
        self_mock.propose_draft_token_ids = MagicMock(return_value=[])
        self_mock._draft_token_ids = []
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        scheduler_output.grammar_bitmask = None
        scheduler_output.scheduled_spec_decode_tokens = {}
        self.env["fakes"]["vllm.distributed.kv_transfer"].has_kv_transfer_group.return_value = False
        self.env["fakes"]["vllm_ascend.utils"].lmhead_tp_enable.return_value = False

        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert isinstance(result, MagicMock)

    def test_execute_model_patch_apply_grammar_bitmask(self):
        self_mock = make_self_mock(self.env)

        self_mock.input_batch.req_ids = ["req1"]
        self_mock.input_batch.num_reqs = 1
        self_mock.input_batch.sampling_metadata = MagicMock()
        self_mock.input_batch.token_ids_cpu = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.pooling_params = None
        self_mock.attn_state = "decode"

        req = MagicMock()
        req.num_computed_tokens = 0
        req.num_tokens = 1
        req.output_token_ids = []
        self_mock.requests = {"req1": req}

        logits_mock = MagicMock()
        self_mock._prepare_inputs = MagicMock(return_value=(
            MagicMock(), MagicMock(), MagicMock(), 10, MagicMock(), 10,
            MagicMock(), None, MagicMock(), None, MagicMock()
        ))
        self_mock._generate_process_reqs_hidden_states = MagicMock(return_value=MagicMock())
        self_mock.model.compute_logits = MagicMock(return_value=logits_mock)
        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = MagicMock()
        sampler_output.logprobs_tensors = None
        self_mock.sampler = MagicMock(return_value=sampler_output)
        self_mock._get_prompt_logprobs_dict = MagicMock(return_value={})
        self_mock.apply_grammar_bitmask = MagicMock()
        self_mock._pool_v010 = MagicMock()
        self_mock._pool = MagicMock()
        self_mock.maybe_setup_kv_connector = MagicMock()
        self_mock.maybe_wait_for_kv_save = MagicMock()
        self_mock.get_finished_kv_transfer = MagicMock(return_value=(None, None))
        self_mock._select_moe_comm_method = MagicMock(return_value=None)
        self_mock.propose_draft_token_ids = MagicMock(return_value=[])
        self_mock._draft_token_ids = []

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        scheduler_output.grammar_bitmask = MagicMock()
        scheduler_output.scheduled_spec_decode_tokens = {}

        self.env["fakes"]["vllm.distributed.kv_transfer"].has_kv_transfer_group.return_value = False
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True

        self.mod.execute_model_patch(self_mock, scheduler_output)
        self_mock.apply_grammar_bitmask.assert_called()

    def test_execute_model_patch_with_kv_connector_output(self):
        self_mock = make_self_mock(self.env)

        self_mock.input_batch.req_ids = ["r1"]
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.input_batch.num_reqs = 1
        self_mock.input_batch.sampling_metadata = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.vocab_size = 100

        req_state = MagicMock()
        req_state.num_computed_tokens = 0
        req_state.num_tokens = 1
        req_state.output_token_ids = []

        self_mock.requests = {"r1": req_state}

        attn_meta = MagicMock()
        attn_meta.attn_state = "decode"
        attn_meta.num_actual_tokens = 1
        attn_meta.seq_lens = DummyTensor([1])

        logits_indices = DummyTensor([0])

        self_mock._prepare_inputs.return_value = (
            attn_meta,
            DummyTensor(),
            [1],
            1,
            DummyTensor(),
            1,
            logits_indices,
            None,
            DummyTensor(),
            None,
            None,
        )

        hidden_states = DummyTensor([1])
        self_mock._generate_process_reqs_hidden_states.return_value = hidden_states

        logits = DummyTensor([1])
        self_mock.model.compute_logits.return_value = logits

        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = DummyTensor([[10]])
        sampler_output.logprobs_tensors = None

        self_mock.sampler.return_value = sampler_output
        self_mock._get_prompt_logprobs_dict.return_value = {}

        self_mock._select_moe_comm_method.return_value = None

        self_mock.get_finished_kv_transfer.return_value = ("send", "recv")

        self_mock.maybe_setup_kv_connector = MagicMock()
        self_mock.maybe_wait_for_kv_save = MagicMock()

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"r1": 1}
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.grammar_bitmask = None

        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result is not None

    def test_execute_model_patch_spec_decode(self):
        self_mock = make_self_mock(self.env)

        self_mock.input_batch.req_ids = ["r1"]
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.input_batch.num_reqs = 1
        self_mock.input_batch.sampling_metadata = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.vocab_size = 100

        req_state = MagicMock()
        req_state.num_computed_tokens = 0
        req_state.num_tokens = 1
        req_state.output_token_ids = []

        self_mock.requests = {"r1": req_state}

        spec_meta = MagicMock()
        spec_meta.bonus_logits_indices = [0]
        spec_meta.target_logits_indices = [0]
        spec_meta.logits_indices = [0]

        attn_meta = MagicMock()
        attn_meta.attn_state = "spec"
        attn_meta.num_actual_tokens = 1
        attn_meta.seq_lens = DummyTensor([1])

        self_mock._prepare_inputs.return_value = (
            attn_meta,
            DummyTensor(),
            [1],
            1,
            DummyTensor(),
            1,
            DummyTensor([0]),
            spec_meta,
            DummyTensor(),
            None,
            None,
        )

        logits = DummyTensor([1])
        self_mock._generate_process_reqs_hidden_states.return_value = DummyTensor([1])
        self_mock.model.compute_logits.return_value = logits

        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = DummyTensor([[2]])
        sampler_output.logprobs_tensors = None

        self_mock.sampler.return_value = sampler_output
        self_mock.rejection_sampler.return_value = DummyTensor([[3]])
        self_mock.rejection_sampler.parse_output.return_value = [[3]]

        self_mock._get_prompt_logprobs_dict.return_value = {}
        self_mock._select_moe_comm_method.return_value = None
        self_mock.get_finished_kv_transfer.return_value = (None, None)

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"r1": 1}
        scheduler_output.scheduled_spec_decode_tokens = {"r1": [1]}
        scheduler_output.grammar_bitmask = None

        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result is not None


class TestPrepareInputsPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def test_prepare_inputs_invalid_total_tokens(self):
        self_mock = make_self_mock(self.env)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 0
        with pytest.raises(ValueError):
            self.mod._prepare_inputs_patch(self_mock, scheduler_output)

    def test_prepare_inputs_invalid_num_reqs(self):
        self_mock = make_self_mock(self.env)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        self_mock.input_batch.num_reqs = 0
        with pytest.raises(ValueError):
            self.mod._prepare_inputs_patch(self_mock, scheduler_output)


class TestDummyRun:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def test_dummy_run_invalid_aclgraph_mode(self):
        self_mock = make_self_mock(self.env)
        with pytest.raises(RuntimeError, match="aclgraph_runtime_mode must be"):
            self.mod.dummy_run(self_mock, 10, aclgraph_runtime_mode=999)

    def test_dummy_run_force_attention_raises(self):
        self_mock = make_self_mock(self.env)
        with pytest.raises(RuntimeError, match="Capturing attention"):
            self.mod.dummy_run(self_mock, 10, force_attention=True)

    def test_dummy_run_normal(self):
        self_mock = make_self_mock(self.env)

        self_mock._sync_metadata_across_dp = MagicMock(return_value=(1, DummyTensor(), False, False))
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor([1])
        self_mock.positions = DummyTensor([1])
        self_mock._generate_dummy_run_hidden_states = MagicMock(return_value=DummyTensor([1]))

        result = self.mod.dummy_run(self_mock, 1)
        assert result is not None

    def test_dummy_run_with_prefill_and_uniform_decode(self):
        self_mock = make_self_mock(self.env)
        self_mock.uniform_decode_query_len = 2
        self_mock._sync_metadata_across_dp = MagicMock(return_value=(4, DummyTensor(), True, False))
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor([1, 2, 3, 4])
        self_mock.positions = DummyTensor([1, 2, 3, 4])
        self_mock._generate_dummy_run_hidden_states = MagicMock(return_value=DummyTensor([4]))

        result = self.mod.dummy_run(self_mock, 4, with_prefill=True, uniform_decode=True)
        assert result is not None

    def test_dummy_run_num_tokens_exceeds_max(self):
        self_mock = make_self_mock(self.env)
        self_mock.scheduler_config.max_num_batched_tokens = 10
        # mock _sync_metadata_across_dp to avoid unpack error
        self_mock._sync_metadata_across_dp.return_value = (20, DummyTensor(), False, False)
        with pytest.raises(RuntimeError, match="cannot exceed max_num_batched_tokens"):
            self.mod.dummy_run(self_mock, 20)

    def test_dummy_run_aclgraph_mode_mismatch(self):
        self_mock = make_self_mock(self.env)
        self_mock.aclgraph_dispatcher.dispatch.return_value = (self.env["CUDAGraphMode"].PIECEWISE, MagicMock())
        # provide valid return for _sync_metadata_across_dp
        self_mock._sync_metadata_across_dp.return_value = (5, DummyTensor(), False, False)
        self_mock._select_moe_comm_method.return_value = None
        with pytest.raises(RuntimeError, match="Aclgraph runtime mode mismatch"):
            self.mod.dummy_run(self_mock, 5, aclgraph_runtime_mode=self.env["CUDAGraphMode"].FULL)


class TestUpdateStatesPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def test_update_states_remove_finished(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.requests = {"r1": MagicMock()}
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = ["r1"]
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        self.mod._update_states_patch(self_mock, scheduler_output)
        self_mock.input_batch.remove_request.assert_called()


class TestGenerateHiddenStatesPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_prof="1")

    def test_no_model_raises(self):
        self_mock = make_self_mock(self.env)
        self_mock.model = None
        with pytest.raises(RuntimeError, match="Model must be initialized"):
            self.mod._generate_process_reqs_hidden_states_patch(
                self_mock, None, False, None, None, None, None, None
            )

    def test_profiling_mode_called(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 0
        self.mod._generate_process_reqs_hidden_states_patch(
            self_mock, None, False, None, None, None, None, None
        )
        self.env["fakes"]["aura.runner.infer_adapter.vllm.patch.comm.npu_model_profiling"].run_model_with_profiling.assert_called_once()

    def test_regular_forward(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 1
        self.mod._generate_process_reqs_hidden_states_patch(
            self_mock, None, False, None, None, None, None, None
        )
        self_mock.model.assert_called_once()


class TestExecuteModelPatchBroadcastPP:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_getenv="true")

    def test_mid_pipeline_not_last_rank_no_broadcast(self):
        """Mid-pipeline (not last rank) with broadcast_pp_output=False returns hidden_states."""
        self_mock = make_self_mock(self.env)
        self_mock.parallel_config.distributed_executor_backend = "torchrun"  # not external_launcher
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = False
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.ranks = [0,1]
        hidden_states = MagicMock()
        self_mock._prepare_inputs.return_value = (MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
                                                  MagicMock(), None, MagicMock(), None, MagicMock())
        self_mock._generate_process_reqs_hidden_states.return_value = hidden_states
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result is hidden_states
        assert result.kv_connector_output is None

    def test_mid_pipeline_not_last_rank_with_broadcast_sends_tensor_dict(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = None
        self_mock.parallel_config.distributed_executor_backend = "external_launcher"
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = False
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.ranks = [0,1]
        hidden_states = MagicMock()
        hidden_states.tensors = {"t1": MagicMock()}
        IntermediateTensors = self.env["fakes"]["vllm.sequence"].IntermediateTensors
        hidden_states.__class__ = IntermediateTensors
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = hidden_states
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        self.mod.execute_model_patch(self_mock, scheduler_output)
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group().send_tensor_dict.assert_called_once()

    def test_mid_pipeline_hidden_states_not_intermediate_tensors_raises(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = None
        self_mock.parallel_config.distributed_executor_backend = "external_launcher"
        pp_group = self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value
        pp_group.is_last_rank = False
        pp_group.ranks = [0, 1]
        hidden_states = MagicMock()
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = hidden_states
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        with pytest.raises(RuntimeError, match="hidden_states must be IntermediateTensors"):
            self.mod.execute_model_patch(self_mock, scheduler_output)

    def test_last_rank_pooling_params_v010(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = {"task": "embedding"}
        self.env["fakes"]["vllm_ascend.utils"].vllm_version_is.side_effect = lambda x: x == "0.10.1.1"
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True
        self_mock.input_batch.pooling_params = {"task": "embedding"}
        hidden_states = MagicMock()
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = hidden_states
        self_mock._pool_v010.return_value = "pooled_v010"
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        result = self.mod.execute_model_patch(self_mock, scheduler_output)
        assert result == "pooled_v010"

    def test_speculative_config_calls_propose_draft_token_ids(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = None
        self_mock.speculative_config = MagicMock()
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True
        self_mock.input_batch.req_ids = ["r1"]
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.input_batch.num_reqs = 1
        req = MagicMock()
        req.num_computed_tokens = 0
        req.num_tokens = 10
        req.output_token_ids = []
        self_mock.requests = {"r1": req}
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = MagicMock()
        self_mock.model.compute_logits.return_value = MagicMock()
        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = DummyTensor([[1]])
        sampler_output.logprobs_tensors = None
        self_mock.sampler.return_value = sampler_output
        self_mock._get_prompt_logprobs_dict.return_value = {}
        self_mock.propose_draft_token_ids.return_value = [2,3]
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"r1": 1}
        scheduler_output.grammar_bitmask = None
        scheduler_output.scheduled_spec_decode_tokens = {}
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        self.mod.execute_model_patch(self_mock, scheduler_output)
        self_mock.propose_draft_token_ids.assert_called_once()

    def test_has_kv_transfer_group_calls_clear_connector_metadata(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = None
        self.env["fakes"]["vllm.distributed.kv_transfer"].has_kv_transfer_group.return_value = True
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True
        self_mock.input_batch.req_ids = ["r1"]
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.input_batch.num_reqs = 1
        req = MagicMock()
        req.num_computed_tokens = 0
        req.num_tokens = 10
        req.output_token_ids = []
        self_mock.requests = {"r1": req}
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = MagicMock()
        self_mock.model.compute_logits.return_value = MagicMock()
        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = DummyTensor([[1]])
        sampler_output.logprobs_tensors = None
        self_mock.sampler.return_value = sampler_output
        self_mock._get_prompt_logprobs_dict.return_value = {}
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"r1": 1}
        scheduler_output.grammar_bitmask = None
        scheduler_output.scheduled_spec_decode_tokens = {}
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        self.mod.execute_model_patch(self_mock, scheduler_output)
        self.env["fakes"]["vllm.distributed.kv_transfer"].get_kv_transfer_group().clear_connector_metadata.assert_called_once()


    def test_end_idx_exceeds_max_model_len_raises(self):
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.pooling_params = None
        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True
        self_mock.input_batch.req_ids = ["r1"]
        self_mock.input_batch.req_id_to_index = {"r1": 0}
        self_mock.input_batch.num_reqs = 1
        self_mock.input_batch.num_tokens_no_spec = [50]
        self_mock.input_batch.num_tokens = [50]
        self_mock.input_batch.token_ids_cpu = MagicMock()
        req = MagicMock()
        req.num_computed_tokens = 5
        req.num_tokens = 5
        req.output_token_ids = []
        self_mock.requests = {"r1": req}
        self_mock._prepare_inputs.return_value = (
            MagicMock(), MagicMock(), MagicMock(), 1, MagicMock(), 1,
            MagicMock(), None, MagicMock(), None, MagicMock()
        )
        self_mock._generate_process_reqs_hidden_states.return_value = MagicMock()
        self_mock.model.compute_logits.return_value = MagicMock()
        sampler_output = MagicMock()
        sampler_output.sampled_token_ids = DummyTensor([[1]])   # len=1
        sampler_output.logprobs_tensors = None
        self_mock.sampler.return_value = sampler_output
        self_mock._get_prompt_logprobs_dict.return_value = {}
        self_mock.model_config.max_model_len = 10               # 50 + 1 > 10
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.num_scheduled_tokens = {"r1": 1}
        scheduler_output.grammar_bitmask = None
        scheduler_output.scheduled_spec_decode_tokens = {}
        self_mock.get_finished_kv_transfer.return_value = (None, None)
        with pytest.raises(RuntimeError, match="Sampled token IDs exceed"):
            self.mod.execute_model_patch(self_mock, scheduler_output)


class TestDummyRunMoreBranches:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def make_dummy_mock(self, num_tokens=5):
        """Helper to create a self mock with minimal dummy_run setup."""
        self_mock = make_self_mock(self.env)
        self_mock._sync_metadata_across_dp.return_value = (num_tokens, DummyTensor(), False, False)
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor(list(range(num_tokens)))
        self_mock.positions = DummyTensor(list(range(num_tokens)))
        self_mock._generate_dummy_run_hidden_states.return_value = DummyTensor(list(range(num_tokens)))
        return self_mock

    def test_with_prefill_decode_num_reqs(self):
        """with_prefill=True gives num_reqs = num_tokens (no decode token per req)."""
        self_mock = self.make_dummy_mock(3)
        self_mock._sync_metadata_across_dp.return_value = (3, DummyTensor(), True, False)
        result = self.mod.dummy_run(self_mock, 3, with_prefill=True)
        assert result is not None

    def test_is_multimodal_model(self):
        """is_multimodal_model=True uses inputs_embeds."""
        self_mock = self.make_dummy_mock(3)
        self_mock.is_multimodal_model = True
        self_mock.inputs_embeds = DummyTensor([10,11,12])
        result = self.mod.dummy_run(self_mock, 3)
        assert result is not None

    def test_uses_mrope(self):
        """uses_mrope=True uses mrope_positions."""
        self_mock = self.make_dummy_mock(3)
        self_mock.uses_mrope = True
        self_mock.mrope_positions = DummyTensor([[1,2,3]])
        result = self.mod.dummy_run(self_mock, 3)
        assert result is not None

    def test_need_dummy_logits_true(self):
        """need_dummy_logits=True calls compute_logits during dummy run."""
        self_mock = self.make_dummy_mock(3)
        self_mock.in_profile_run = False
        self.env["fakes"]["vllm_ascend.utils"].lmhead_tp_enable.return_value = True
        result = self.mod.dummy_run(self_mock, 3)
        assert result is not None
        # compute_logits should have been called; can assert model.compute_logits
        self_mock.model.compute_logits.assert_called()

    def test_speculative_config_deepseek_mtp(self):
        self_mock = self.make_dummy_mock(3)
        self_mock.speculative_config = MagicMock()
        self_mock.speculative_config.method = "deepseek_mtp"
        drafter = MagicMock(spec=self.env["fakes"]["vllm_ascend.worker.mtp_proposer_v1"].MtpProposer)
        drafter.dummy_run = MagicMock()
        self_mock.drafter = drafter
        self.mod.dummy_run(self_mock, 3)
        drafter.dummy_run.assert_called_once()

    def test_speculative_config_deepseek_mtp_wrong_drafter_raises(self):
        self_mock = self.make_dummy_mock(3)
        self_mock.speculative_config = MagicMock()
        self_mock.speculative_config.method = "deepseek_mtp"
        self_mock.drafter = MagicMock()
        with pytest.raises(RuntimeError, match="drafter must be MtpProposer"):
            self.mod.dummy_run(self_mock, 3)


class TestGenerateDummyRunHiddenStatesPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_prof="1")  # profiling enabled

    def test_profiling_mode_called(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 0
        self_mock.model = MagicMock()
        self_mock.use_aux_hidden_state_outputs = False
        self_mock.use_spec_decode = False
        result = self.mod._generate_dummy_run_hidden_states_patch(
            self_mock, False, False, None, None, None, 5, None, None
        )
        self.env["fakes"]["aura.runner.infer_adapter.vllm.patch.comm.npu_model_profiling"].run_model_with_profiling.assert_called_once()

    def test_regular_forward(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 1
        self_mock.model = MagicMock()
        self_mock.use_aux_hidden_state_outputs = False
        self_mock.use_spec_decode = False
        result = self.mod._generate_dummy_run_hidden_states_patch(
            self_mock, False, False, None, None, None, 5, None, None
        )
        self_mock.model.assert_called_once()

    def test_aux_hidden_state_outputs(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 1
        self_mock.model.return_value = (MagicMock(), MagicMock())
        self_mock.use_aux_hidden_state_outputs = True
        self_mock.use_spec_decode = False
        result = self.mod._generate_dummy_run_hidden_states_patch(
            self_mock, False, False, None, None, None, 5, None, None
        )

    def test_use_spec_decode_eagle_proposer(self):
        self_mock = make_self_mock(self.env)
        self_mock.stat_step = 1
        self_mock.model.return_value = MagicMock()
        self_mock.use_aux_hidden_state_outputs = False
        self_mock.use_spec_decode = True
        drafter = MagicMock(spec=self.env["fakes"]["vllm_ascend.worker.eagle_proposer_v1"].EagleProposer)
        drafter.dummy_run = MagicMock()
        self_mock.drafter = drafter
        self.mod._generate_dummy_run_hidden_states_patch(
            self_mock, False, False, None, None, None, 5, None, None
        )
        drafter.dummy_run.assert_called_once_with(5)


class TestDummyRunWithStat:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_getenv="true")  # enable stat patch

    def test_dummy_run_with_stat_calls(self):
        self_mock = make_self_mock(self.env)
        self_mock._sync_metadata_across_dp.return_value = (3, DummyTensor(), False, False)
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor([1,2,3])
        self_mock.positions = DummyTensor([1,2,3])
        self_mock._generate_dummy_run_hidden_states.return_value = DummyTensor([1,2,3])
        self_mock.stat_step = 0
        result = self.mod.dummy_run_with_stat(self_mock, 3)
        assert result is not None
        # check that some stat calls were made
        self.env["vllm_output_statics"].set_stat.assert_called()
        self.env["vllm_output_statics"].add_stat.assert_called()


class TestDummyRunMorePaths:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env)

    def make_dummy_mock(self, num_tokens=5):
        self_mock = make_self_mock(self.env)
        self_mock._sync_metadata_across_dp.return_value = (num_tokens, DummyTensor(), False, False)
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor(list(range(num_tokens)))
        self_mock.positions = DummyTensor(list(range(num_tokens)))
        self_mock._generate_dummy_run_hidden_states.return_value = DummyTensor(list(range(num_tokens)))
        return self_mock

    def test_is_kv_producer(self):
        """is_kv_producer=True forces with_prefill=True."""
        self_mock = self.make_dummy_mock(3)
        self_mock.is_kv_producer = True
        # sync_metadata_across_dp should be called with with_prefill True (but our mock returns anyway)
        result = self.mod.dummy_run(self_mock, 3, with_prefill=False)
        assert result is not None


class TestDummyRunWithStatMore:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_getenv="true")  # stat patch enabled

    def make_stat_mock(self, num_tokens=3):
        self_mock = make_self_mock(self.env)
        self_mock._sync_metadata_across_dp.return_value = (num_tokens, DummyTensor(), False, False)
        self_mock._select_moe_comm_method.return_value = None
        self_mock._build_attention_metadata.return_value = MagicMock()
        self_mock.maybe_dummy_run_with_lora.return_value = DummyContext()
        self_mock.input_ids = DummyTensor(list(range(num_tokens)))
        self_mock.positions = DummyTensor(list(range(num_tokens)))
        self_mock._generate_dummy_run_hidden_states.return_value = DummyTensor(list(range(num_tokens)))
        return self_mock

    def test_uniform_decode(self):
        """uniform_decode=True path in dummy_run_with_stat."""
        self_mock = self.make_stat_mock(6)
        self_mock.uniform_decode_query_len = 2
        self_mock._sync_metadata_across_dp.return_value = (6, DummyTensor(), False, False)
        result = self.mod.dummy_run_with_stat(self_mock, 6, uniform_decode=True)
        assert result is not None

    def test_with_prefill(self):
        """with_prefill=True path in dummy_run_with_stat."""
        self_mock = self.make_stat_mock(4)
        self_mock._sync_metadata_across_dp.return_value = (4, DummyTensor(), True, False)
        result = self.mod.dummy_run_with_stat(self_mock, 4, with_prefill=True)
        assert result is not None

    def test_multimodal(self):
        """is_multimodal_model=True uses inputs_embeds."""
        self_mock = self.make_stat_mock(3)
        self_mock.is_multimodal_model = True
        self_mock.inputs_embeds = DummyTensor([10, 11, 12])
        result = self.mod.dummy_run_with_stat(self_mock, 3)
        assert result is not None

    def test_uses_mrope(self):
        """uses_mrope=True uses mrope_positions."""
        self_mock = self.make_stat_mock(3)
        self_mock.uses_mrope = True
        self_mock.mrope_positions = DummyTensor([[1, 2, 3]])
        result = self.mod.dummy_run_with_stat(self_mock, 3)
        assert result is not None

    def test_need_dummy_logits(self):
        """need_dummy_logits=True triggers compute_logits."""
        self_mock = self.make_stat_mock(3)
        self_mock.in_profile_run = False
        self.env["fakes"]["vllm_ascend.utils"].lmhead_tp_enable.return_value = True
        result = self.mod.dummy_run_with_stat(self_mock, 3)
        assert result is not None
        self_mock.model.compute_logits.assert_called()

    def test_speculative_deepseek_mtp(self):
        """speculative_config with deepseek_mtp calls drafter.dummy_run."""
        self_mock = self.make_stat_mock(3)
        self_mock.speculative_config = MagicMock()
        self_mock.speculative_config.method = "deepseek_mtp"
        drafter = MagicMock(spec=self.env["fakes"]["vllm_ascend.worker.mtp_proposer_v1"].MtpProposer)
        drafter.dummy_run = MagicMock()
        self_mock.drafter = drafter
        result = self.mod.dummy_run_with_stat(self_mock, 3)
        drafter.dummy_run.assert_called_once()

    def test_aclgraph_mode_mismatch(self):
        """aclgraph_runtime_mode mismatch raises."""
        self_mock = self.make_stat_mock(5)
        self_mock.aclgraph_dispatcher.dispatch.return_value = (self.env["CUDAGraphMode"].PIECEWISE, MagicMock())
        self_mock._sync_metadata_across_dp.return_value = (5, DummyTensor(), False, False)
        with pytest.raises(RuntimeError, match="Aclgraph runtime mode mismatch"):
            self.mod.dummy_run_with_stat(self_mock, 5, aclgraph_runtime_mode=self.env["CUDAGraphMode"].FULL)

    def test_num_tokens_exceeds_max(self):
        """num_tokens exceeds max_num_batched_tokens raises."""
        self_mock = self.make_stat_mock(20)
        self_mock.scheduler_config.max_num_batched_tokens = 10
        self_mock._sync_metadata_across_dp.return_value = (20, DummyTensor(), False, False)
        with pytest.raises(RuntimeError, match="cannot exceed max_num_batched_tokens"):
            self.mod.dummy_run_with_stat(self_mock, 20)


class TestUpdateStatesPatchMore:
    @pytest.fixture(autouse=True)
    def setup(self, fake_model_runner_env):
        self.env = fake_model_runner_env
        self.mod = import_module(self.env, env_getenv="true")  # stat patch enabled

    def test_encoder_cache_pop_v010(self):
        """vllm_version_is 0.10.1.1: encoder_cache.pop for finished_req_ids."""
        self_mock = make_self_mock(self.env)
        self_mock.encoder_cache = {"req1": MagicMock(), "req2": MagicMock()}
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()
        self_mock.input_batch.req_id_to_index = {}

        self.env["fakes"]["vllm_ascend.utils"].vllm_version_is.side_effect = lambda x: x in ("0.10.1.1", "0.10.1")

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = ["req1"]
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys.return_value = []
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        self.mod._update_states_patch(self_mock, scheduler_output)
        # req1 should be popped from encoder_cache (and requests)
        assert "req1" not in self_mock.encoder_cache

    def test_free_encoder_input_ids_v010(self):
        """vllm_version_is 0.10.1.1: free_encoder_input_ids removes cached outputs."""
        self_mock = make_self_mock(self.env)
        encoder_out = MagicMock()
        encoder_out.pop.return_value = None
        self_mock.encoder_cache = {"req1": encoder_out}
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()
        self_mock.input_batch.req_id_to_index = {}

        self.env["fakes"]["vllm_ascend.utils"].vllm_version_is.side_effect = lambda x: x in ("0.10.1.1", "0.10.1")

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = [("req1", "input1")]
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys.return_value = []
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        self.mod._update_states_patch(self_mock, scheduler_output)
        encoder_out.pop.assert_called_with("input1", None)

    def test_sampling_type_random_seed_creates_generator(self):
        """sampling_params.sampling_type == RANDOM_SEED creates a Generator."""
        self_mock = make_self_mock(self.env)
        self_mock.requests = {}
        self_mock.input_batch.add_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()
        self_mock.input_batch.req_id_to_index = {}
        self_mock.input_batch.remove_request = MagicMock()

        SamplingType = self.env["fakes"]["vllm.sampling_params"].SamplingType
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.sampling_params = MagicMock()
        new_req.sampling_params.sampling_type = SamplingType.RANDOM_SEED
        new_req.sampling_params.seed = 42
        new_req.pooling_params = None
        new_req.prompt_token_ids = [1,2,3]
        new_req.mm_kwargs = None
        new_req.mm_positions = None
        new_req.block_ids = None
        new_req.num_computed_tokens = 0
        new_req.lora_request = None
        new_req.mm_hashes = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys.return_value = []
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        self.mod._update_states_patch(self_mock, scheduler_output)
        # Check that a request with generator was added
        assert "r1" in self_mock.requests
        assert self_mock.requests["r1"].generator is not None

    def test_pooling_params_task_none_raises(self):
        """pooling_params with task=None raises ValueError."""
        self_mock = make_self_mock(self.env)
        self_mock.requests = {}
        self_mock.input_batch.add_request = MagicMock()
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()
        self_mock.input_batch.req_id_to_index = {}

        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.sampling_params = None
        new_req.pooling_params = MagicMock()
        new_req.pooling_params.task = None   # invalid
        new_req.prompt_token_ids = [1,2,3]
        new_req.mm_kwargs = None
        new_req.mm_positions = None
        new_req.block_ids = None
        new_req.num_computed_tokens = 0
        new_req.lora_request = None
        new_req.mm_hashes = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys.return_value = []
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = MagicMock()
        scheduler_output.scheduled_cached_reqs.req_ids = []

        with pytest.raises(ValueError, match="You did not set `task` in the API"):
            self.mod._update_states_patch(self_mock, scheduler_output)

    def test_not_last_rank_adds_new_tokens(self):
        """not is_last_rank: adds new_token_ids to output_token_ids."""
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.req_id_to_index = MagicMock()
        self_mock.input_batch.req_id_to_index.get = MagicMock(return_value=0)
        self_mock.input_batch.req_id_to_index.keys = MagicMock(return_value=set())

        self_mock.requests = {"r1": MagicMock()}
        self_mock.requests["r1"].num_tokens = 1
        self_mock.requests["r1"].num_computed_tokens = 0
        self_mock.requests["r1"].output_token_ids = MagicMock()
        self_mock.requests["r1"].block_ids = []
        self_mock.input_batch.num_computed_tokens_cpu = [0]
        self_mock.input_batch.token_ids_cpu = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.block_table = MagicMock()
        self_mock.input_batch.add_request = MagicMock()
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()

        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = False

        req_data = MagicMock()
        req_data.req_ids = ["r1"]
        req_data.num_computed_tokens = [2]
        req_data.new_block_ids = [None]
        req_data.resumed_from_preemption = [False]
        req_data.new_token_ids = [[201, 202]]

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys = MagicMock(return_value=set())
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = req_data
        scheduler_output.scheduled_spec_decode_tokens = {}

        self.mod._update_states_patch(self_mock, scheduler_output)
        self_mock.requests["r1"].output_token_ids.extend.assert_called()

    def test_spec_token_ids_appended(self):
        """spec_token_ids present appends to token_ids_cpu and increments num_tokens."""
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.req_id_to_index = MagicMock()
        self_mock.input_batch.req_id_to_index.get = MagicMock(return_value=0)
        self_mock.input_batch.req_id_to_index.keys = MagicMock(return_value=set())

        self_mock.requests = {"r1": MagicMock()}
        self_mock.requests["r1"].num_tokens = 2
        self_mock.requests["r1"].num_computed_tokens = 0
        self_mock.requests["r1"].output_token_ids = MagicMock()
        self_mock.requests["r1"].block_ids = []
        self_mock.input_batch.num_computed_tokens_cpu = [0]
        self_mock.input_batch.token_ids_cpu = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.block_table = MagicMock()
        self_mock.input_batch.add_request = MagicMock()
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()

        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True

        req_data = MagicMock()
        req_data.req_ids = ["r1"]
        req_data.num_computed_tokens = [2]
        req_data.new_block_ids = [None]
        req_data.resumed_from_preemption = [False]
        req_data.new_token_ids = [[]]

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys = MagicMock(return_value=set())
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = req_data
        scheduler_output.scheduled_spec_decode_tokens = {"r1": [301, 302]}

        self.mod._update_states_patch(self_mock, scheduler_output)
        assert self_mock.input_batch.num_tokens[0] == 2

    def test_resumed_from_preemption_requires_new_block_ids(self):
        """resumed_from_preemption=True with new_block_ids=None raises RuntimeError."""
        self_mock = make_self_mock(self.env)
        self_mock.input_batch.req_id_to_index = MagicMock()
        self_mock.input_batch.req_id_to_index.get = MagicMock(return_value=0)
        self_mock.input_batch.req_id_to_index.keys = MagicMock(return_value=set())

        self_mock.requests = {"r1": MagicMock()}
        self_mock.requests["r1"].num_tokens = 5
        self_mock.requests["r1"].num_computed_tokens = 0
        self_mock.requests["r1"].output_token_ids = MagicMock()
        self_mock.requests["r1"].block_ids = []
        self_mock.input_batch.num_computed_tokens_cpu = [0]
        self_mock.input_batch.token_ids_cpu = MagicMock()
        self_mock.input_batch.num_tokens_no_spec = [0]
        self_mock.input_batch.num_tokens = [0]
        self_mock.input_batch.block_table = MagicMock()
        self_mock.input_batch.add_request = MagicMock()
        self_mock.input_batch.remove_request = MagicMock()
        self_mock.input_batch.condense = MagicMock()
        self_mock.input_batch.refresh_metadata = MagicMock()

        self.env["fakes"]["vllm.distributed.parallel_state"].get_pp_group.return_value.is_last_rank = True

        req_data = MagicMock()
        req_data.req_ids = ["r1"]
        req_data.num_computed_tokens = [2]
        req_data.new_block_ids = [None]
        req_data.resumed_from_preemption = [True]
        req_data.new_token_ids = [[]]

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.free_encoder_input_ids = []
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = MagicMock()
        scheduler_output.num_scheduled_tokens.keys = MagicMock(return_value=set())
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = req_data
        scheduler_output.scheduled_spec_decode_tokens = {}

        with pytest.raises(RuntimeError, match="new_block_ids must not be None"):
            self.mod._update_states_patch(self_mock, scheduler_output)
