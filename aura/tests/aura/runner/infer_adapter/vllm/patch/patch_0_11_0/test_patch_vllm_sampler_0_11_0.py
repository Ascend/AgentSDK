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
import torch as real_torch

# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_vllm_sampler
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_sampler_env():
    # ---- Fake torch ----
    orig_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.int32 = "int32"
    fake_torch.int64 = "int64"
    fake_torch.Tensor = MagicMock
    fake_torch.where = MagicMock(return_value=MagicMock())

    # ---- Fake vllm ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_v1 = types.ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_v1_sample = types.ModuleType("vllm.v1.sample")
    fake_vllm_v1_sample.__path__ = []
    fake_vllm_v1_sample_sampler = types.ModuleType("vllm.v1.sample.sampler")
    class FakeSampler:
        def forward(self, *args, **kwargs):
            pass
        def sample(self, *args, **kwargs):
            pass
    fake_vllm_v1_sample_sampler.Sampler = FakeSampler

    fake_vllm_v1_outputs = types.ModuleType("vllm.v1.outputs")
    fake_vllm_v1_outputs.SamplerOutput = MagicMock

    fake_vllm_v1_sample_metadata = types.ModuleType("vllm.v1.sample.metadata")
    fake_vllm_v1_sample_metadata.SamplingMetadata = MagicMock

    # ---- Fake comm module ----
    fake_comm = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat")
    class StatTimeUtil:
        def get_duration(self):
            return 1.0
    class StatPhase:
        post_samper_compute_logprobs_time = "compute_logprobs"
        post_samper_logits_preproc_time = "logits_preproc"
        post_samper_processor_apply_time = "processor_apply"
        post_samper_apply_penalties_time = "apply_penalties"
        post_samper_sample_next_token_time = "sample_next_token"
        post_samper_sampled_long_time = "sampled_long"
        post_samper_gather_logprobs_time = "gather_logprobs"
        post_samper_sampled_int32_time = "sampled_int32"
        post_samper_sample_greedy_time = "greedy_time"
        post_samper_sample_apply_temperature_time = "apply_temp"
        post_samper_sample_processor_apply_again_time = "processor_again"
        post_samper_sample_topk_topp_time = "topk_topp"
        post_samper_sample_greedy_where_time = "greedy_where"
    fake_comm.StatTimeUtil = StatTimeUtil
    fake_comm.vllm_output_statics = MagicMock()
    fake_comm.StatPhase = StatPhase

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
    fake_0_11_0_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_11_0")
    fake_0_11_0_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_11_0")]
    fake_comm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm")
    fake_comm_pkg.__path__ = []

    all_fakes = {
        "torch": fake_torch,
        "vllm": fake_vllm,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.sample": fake_vllm_v1_sample,
        "vllm.v1.sample.sampler": fake_vllm_v1_sample_sampler,
        "vllm.v1.outputs": fake_vllm_v1_outputs,
        "vllm.v1.sample.metadata": fake_vllm_v1_sample_metadata,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm": fake_comm_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat": fake_comm,
    }
    for name, mod in all_fakes.items():
        sys.modules[name] = mod

    target_module = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_vllm_sampler"
    if target_module in sys.modules:
        del sys.modules[target_module]

    yield {
        "Sampler": FakeSampler,
        "vllm_output_statics": fake_comm.vllm_output_statics,
        "torch": fake_torch,
    }

    for name in list(all_fakes.keys()):
        if name in sys.modules:
            del sys.modules[name]
    if orig_torch is not None:
        sys.modules["torch"] = orig_torch
    else:
        sys.modules.pop("torch", None)

# ---------------------------------------------------------------------------
# Helper: import module with controlled environment
# ---------------------------------------------------------------------------
def import_module(env_getenv="False", env_environ_get="0"):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_vllm_sampler"
    if module_name in sys.modules:
        del sys.modules[module_name]
    with patch("os.getenv", return_value=env_getenv), patch("os.environ.get", return_value=env_environ_get):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_vllm_sampler as mod
        return mod


# ---------------------------------------------------------------------------
# Helper: create a mock Sampler instance
# ---------------------------------------------------------------------------
def make_sampler_mock():
    self = MagicMock()
    self.compute_logprobs = MagicMock(return_value=MagicMock())
    self.apply_allowed_token_ids = MagicMock(return_value=MagicMock())
    self.apply_bad_words = MagicMock(return_value=MagicMock())
    self.apply_penalties = MagicMock(return_value=MagicMock())
    self.sample = MagicMock(return_value=(MagicMock(), MagicMock()))
    self.gather_logprobs = MagicMock(return_value=MagicMock())
    self.logprobs_mode = "raw_logprobs"
    self.greedy_sample = MagicMock(return_value=MagicMock())
    self.apply_temperature = MagicMock(return_value=MagicMock())
    self.topk_topp_sampler = MagicMock(return_value=(MagicMock(), MagicMock()))
    return self


# ---------------------------------------------------------------------------
# Tests for forward_patch
# ---------------------------------------------------------------------------
class TestForwardPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_sampler_env):
        self.mod = import_module()
        self.forward_patch = self.mod.forward_patch

    def test_num_logprobs_none(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sampling_metadata = MagicMock()
        sampling_metadata.max_num_logprobs = None
        sampling_metadata.logitsprocs.non_argmax_invariant = []

        result = self.forward_patch(self_mock, logits, sampling_metadata)
        # gather_logprobs should not be called
        self_mock.gather_logprobs.assert_not_called()
        # result should be a SamplerOutput (MagicMock)
        assert isinstance(result, MagicMock)

    def test_num_logprobs_not_none_raw_logprobs(self):
        self_mock = make_sampler_mock()
        self_mock.logprobs_mode = "raw_logprobs"
        logits = MagicMock()
        sampling_metadata = MagicMock()
        sampling_metadata.max_num_logprobs = 5
        sampling_metadata.logitsprocs.non_argmax_invariant = []

        self.forward_patch(self_mock, logits, sampling_metadata)
        self_mock.compute_logprobs.assert_called_once_with(logits)
        self_mock.gather_logprobs.assert_called_once()

    def test_num_logprobs_not_none_raw_logits(self):
        self_mock = make_sampler_mock()
        self_mock.logprobs_mode = "raw_logits"
        logits = MagicMock()
        logits.clone.return_value = "cloned"
        sampling_metadata = MagicMock()
        sampling_metadata.max_num_logprobs = 3
        sampling_metadata.logitsprocs.non_argmax_invariant = []

        self.forward_patch(self_mock, logits, sampling_metadata)
        logits.clone.assert_called_once()
        self_mock.gather_logprobs.assert_called_once()
        self_mock.compute_logprobs.assert_not_called()

    def test_processed_logprobs_replaces_raw(self):
        self_mock = make_sampler_mock()
        processed_mock = MagicMock()
        self_mock.sample.return_value = (MagicMock(), processed_mock)
        logits = MagicMock()
        sampling_metadata = MagicMock()
        sampling_metadata.max_num_logprobs = 3
        sampling_metadata.logitsprocs.non_argmax_invariant = []

        self.forward_patch(self_mock, logits, sampling_metadata)
        # gather_logprobs called with processed_mock as first positional arg
        self_mock.gather_logprobs.assert_called_once()
        call_args = self_mock.gather_logprobs.call_args[0]
        assert call_args[0] is processed_mock

    def test_all_steps_and_returns_sampler_output(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sampling_metadata = MagicMock()
        sampling_metadata.max_num_logprobs = 2
        sampling_metadata.logitsprocs.non_argmax_invariant = [MagicMock()]
        sampling_metadata.logitsprocs.non_argmax_invariant[0].apply = MagicMock(return_value=MagicMock())

        result = self.forward_patch(self_mock, logits, sampling_metadata)
        self_mock.apply_allowed_token_ids.assert_called_once()
        self_mock.apply_bad_words.assert_called_once()
        self_mock.apply_penalties.assert_called_once()
        self_mock.sample.assert_called_once()
        self_mock.gather_logprobs.assert_called_once()
        assert isinstance(result, MagicMock)


# ---------------------------------------------------------------------------
# Tests for sample_patch
# ---------------------------------------------------------------------------
class TestSamplePatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_sampler_env):
        self.mod = import_module()
        self.sample_patch = self.mod.sample_patch

    def test_all_greedy_and_random_raises(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = True
        sm.all_random = True
        with pytest.raises(ValueError, match="cannot both be True"):
            self.sample_patch(self_mock, logits, sm)

    def test_all_random(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sm = MagicMock()
        sm.all_random = True
        sm.all_greedy = False
        sm.temperature = 0.8
        sm.max_num_logprobs = None
        sm.logitsprocs.argmax_invariant = []
        sm.generators = "gen"
        sm.top_k = 5
        sm.top_p = 0.9

        self.sample_patch(self_mock, logits, sm)
        self_mock.greedy_sample.assert_not_called()
        self_mock.apply_temperature.assert_called_once()
        self_mock.topk_topp_sampler.assert_called_once()

    def test_all_greedy(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = True
        sm.all_random = False
        sm.max_num_logprobs = None
        sampled, probs = self.sample_patch(self_mock, logits, sm)
        self_mock.greedy_sample.assert_called_once()
        assert probs is None

    def test_all_greedy_with_logprobs(self):
        self_mock = make_sampler_mock()
        self_mock.logprobs_mode = "processed_logits"
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = True
        sm.all_random = False
        sm.max_num_logprobs = 3
        sampled, probs = self.sample_patch(self_mock, logits, sm)
        self_mock.greedy_sample.assert_called_once()
        assert probs is logits

    def test_temperature_none_raises(self):
        self_mock = make_sampler_mock()
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = False
        sm.all_random = False
        sm.temperature = None
        sm.logitsprocs.argmax_invariant = []
        with pytest.raises(ValueError, match="temperature cannot be None"):
            self.sample_patch(self_mock, logits, sm)

    def test_normal_path_with_greedy_sampled(self, fake_sampler_env):
        self_mock = make_sampler_mock()
        greedy_mock = MagicMock()
        self_mock.greedy_sample.return_value = greedy_mock
        random_mock = MagicMock()
        self_mock.topk_topp_sampler.return_value = (random_mock, MagicMock())
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = False
        sm.all_random = False
        sm.temperature = 1.0
        sm.max_num_logprobs = None
        sm.generators = "gen"
        sm.top_k = 10
        sm.top_p = 0.8
        sm.logitsprocs.argmax_invariant = []
        sampled, _ = self.sample_patch(self_mock, logits, sm)
        fake_torch = fake_sampler_env["torch"]
        fake_torch.where.assert_called_once()
        assert sampled is fake_torch.where.return_value

    def test_normal_path_greedy_none(self):
        self_mock = make_sampler_mock()
        random_mock = MagicMock()
        self_mock.topk_topp_sampler.return_value = (random_mock, MagicMock())
        self_mock.greedy_sample.return_value = None
        logits = MagicMock()
        sm = MagicMock()
        sm.all_greedy = False
        sm.all_random = False
        sm.temperature = 1.0
        sm.max_num_logprobs = None
        sm.generators = "g"
        sm.top_k = 1
        sm.top_p = 0.5
        sm.logitsprocs.argmax_invariant = []
        sampled, _ = self.sample_patch(self_mock, logits, sm)
        assert sampled is random_mock


# ---------------------------------------------------------------------------
# Tests for module-level patching condition
# ---------------------------------------------------------------------------
class TestModulePatchCondition:
    def test_patch_applied(self, fake_sampler_env):
        mod = import_module(env_getenv="true", env_environ_get="1")
        Sampler = fake_sampler_env["Sampler"]
        assert Sampler.forward is mod.forward_patch
        assert Sampler.sample is mod.sample_patch

    def test_patch_not_applied_stats_disabled(self, fake_sampler_env):
        mod = import_module(env_getenv="false", env_environ_get="1")
        Sampler = fake_sampler_env["Sampler"]
        assert Sampler.forward is not mod.forward_patch
        assert Sampler.sample is not mod.sample_patch

    def test_patch_not_applied_level_zero(self, fake_sampler_env):
        mod = import_module(env_getenv="true", env_environ_get="0")
        Sampler = fake_sampler_env["Sampler"]
        assert Sampler.forward is not mod.forward_patch
