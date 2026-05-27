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


# ---------------------------------------------------------------------------
# Fixture: build fake module data, never touch sys.modules
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_sampler_env():
    # ---- Fake vllm_ascend.sample.sampler ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend_sample = types.ModuleType("vllm_ascend.sample")
    fake_vllm_ascend_sampler = types.ModuleType("vllm_ascend.sample.sampler")
    class AscendTopKTopPSampler:
        def forward_native(self, *args, **kwargs):
            pass
    fake_vllm_ascend_sampler.AscendTopKTopPSampler = AscendTopKTopPSampler

    # ---- Fake vllm.v1.sample.ops.topk_topp_sampler ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_v1_sample_ops = types.ModuleType("vllm.v1.sample.ops")
    fake_topk_topp = types.ModuleType("vllm.v1.sample.ops.topk_topp_sampler")
    random_sample_mock = MagicMock(return_value="sampled_output")
    fake_topk_topp.random_sample = random_sample_mock

    # ---- Fake vllm_ascend.utils ----
    fake_vllm_ascend_utils = types.ModuleType("vllm_ascend.utils")
    vllm_version_is_mock = MagicMock()
    fake_vllm_ascend_utils.vllm_version_is = vllm_version_is_mock

    # ---- Fake vllm.config ----
    fake_vllm_config = types.ModuleType("vllm.config")
    class LogprobsMode:
        PROCESSED_LOGITS = 1
        PROCESSED_LOGPROBS = 2
        RAW_LOGPROBS = 3
        RAW_LOGITS = 4
    fake_vllm_config.LogprobsMode = LogprobsMode

    # ---- Fake comm.stat module ----
    fake_comm = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat")
    class StatTimeUtil:
        def get_duration(self):
            return 0.1
    class StatPhase:
        post_samper_sample_topk_topp_apply_time = "apply"
        post_samper_sample_topk_topp_logits_log_softmax_time = "log_softmax"
        post_samper_sample_topk_topp_probs_softmax_time = "probs_softmax"
        post_samper_sample_topk_topp_random_sample_time = "random_sample"
    fake_comm.StatTimeUtil = StatTimeUtil
    fake_comm.vllm_output_statics = MagicMock()
    fake_comm.StatPhase = StatPhase

    # ---- Fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.Tensor = MagicMock

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

    # All fake modules
    fakes = {
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.sample": fake_vllm_ascend_sample,
        "vllm_ascend.sample.sampler": fake_vllm_ascend_sampler,
        "vllm": fake_vllm,
        "vllm.v1.sample.ops": fake_vllm_v1_sample_ops,
        "vllm.v1.sample.ops.topk_topp_sampler": fake_topk_topp,
        "vllm_ascend.utils": fake_vllm_ascend_utils,
        "vllm.config": fake_vllm_config,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_10_2": fake_0_10_2_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm": fake_comm_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat": fake_comm,
        "torch": fake_torch,
    }

    yield {
        "fakes": fakes,
        "AscendTopKTopPSampler": AscendTopKTopPSampler,
        "random_sample": random_sample_mock,
        "vllm_version_is": vllm_version_is_mock,
        "LogprobsMode": LogprobsMode,
        "vllm_output_statics": fake_comm.vllm_output_statics,
        "StatPhase": StatPhase,
    }
    # No manual cleanup – patch.dict handles restore automatically


# ---------------------------------------------------------------------------
# Helper: import module under test with temporary fake environment
# ---------------------------------------------------------------------------
def import_module(fake_sampler_env, env_getenv="False", version_is_01011=False):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_sampler"
    if module_name in sys.modules:
        del sys.modules[module_name]

    if version_is_01011:
        fake_sampler_env["vllm_version_is"].side_effect = lambda x: x in ("0.10.1.1", "0.10.1")
    else:
        fake_sampler_env["vllm_version_is"].side_effect = lambda x: False

    fakes = fake_sampler_env["fakes"]
    with patch.dict(sys.modules, fakes), \
         patch("os.getenv", return_value=env_getenv):
        import aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_sampler as mod
    return mod


# ---------------------------------------------------------------------------
# Helper: create a mock AscendTopKTopPSampler instance
# ---------------------------------------------------------------------------
def make_self_mock(logprobs_mode=None):
    self = MagicMock()
    self._apply_top_k_top_p = MagicMock(return_value=MagicMock())
    if logprobs_mode is not None:
        self.logprobs_mode = logprobs_mode
    return self


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestForwardNativePatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_sampler_env):
        self.env = fake_sampler_env

    def test_non_01011_logprobs_processed_logits(self):
        """logprobs_mode=PROCESSED_LOGITS returns logits_to_return=logits and a tuple."""
        mod = import_module(self.env, version_is_01011=False)
        self_mock = make_self_mock(logprobs_mode=self.env["LogprobsMode"].PROCESSED_LOGITS)
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        result = mod.forward_native_patch(self_mock, logits, "gen", 3, 0.9)

        self.env["random_sample"].assert_called_once_with("probs", "gen")
        assert result == (self.env["random_sample"].return_value, logits)

    def test_non_01011_logprobs_processed_logprobs(self):
        """logprobs_mode=PROCESSED_LOGPROBS calls log_softmax and returns its result."""
        mod = import_module(self.env, version_is_01011=False)
        self_mock = make_self_mock(logprobs_mode=self.env["LogprobsMode"].PROCESSED_LOGPROBS)
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"
        logits.log_softmax.return_value = "logsoft"

        result = mod.forward_native_patch(self_mock, logits, "g", 5, 0.5)

        logits.log_softmax.assert_called_once_with(dim=-1, dtype="float32")
        assert result == (self.env["random_sample"].return_value, "logsoft")

    def test_non_01011_logprobs_other_mode(self):
        """Other logprobs_mode returns logits_to_return=None."""
        mod = import_module(self.env, version_is_01011=False)
        self_mock = make_self_mock(logprobs_mode=999)
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        result = mod.forward_native_patch(self_mock, logits, "g", 1, 1)
        assert result == (self.env["random_sample"].return_value, None)

    def test_01011_version_no_logprobs(self):
        """0.10.1.1/0.10.1 version ignores logprobs_mode, output is random_sample result directly."""
        mod = import_module(self.env, version_is_01011=True)
        self_mock = make_self_mock()
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        result = mod.forward_native_patch(self_mock, logits, "g", 2, 0.8)

        assert result == self.env["random_sample"].return_value
        self.env["random_sample"].assert_called_once_with("probs", "g")

    def test_01011_version_logprobs_ignored(self):
        """Even with logprobs_mode set, 0.10.1.1/0.10.1 does not produce logits_to_return."""
        mod = import_module(self.env, version_is_01011=True)
        self_mock = make_self_mock(logprobs_mode=self.env["LogprobsMode"].PROCESSED_LOGITS)
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        result = mod.forward_native_patch(self_mock, logits, "g", 2, 0.8)
        assert result == self.env["random_sample"].return_value

    def test_stat_calls_order_non_01011(self):
        """Verify add_stat phases order for non-0.10.1.1 version."""
        mod = import_module(self.env, version_is_01011=False)
        self_mock = make_self_mock(logprobs_mode=self.env["LogprobsMode"].PROCESSED_LOGITS)
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        mod.forward_native_patch(self_mock, logits, "g", 3, 0.9)

        calls = self.env["vllm_output_statics"].add_stat.call_args_list
        phases = [call[0][0] for call in calls]
        assert phases == ["apply", "log_softmax", "probs_softmax", "random_sample"]

    def test_stat_calls_order_01011(self):
        """Verify add_stat phases order for 0.10.1.1 version (includes unconditional log_softmax)."""
        mod = import_module(self.env, version_is_01011=True)
        self_mock = make_self_mock()
        logits = self_mock._apply_top_k_top_p.return_value
        logits.softmax.return_value = "probs"

        mod.forward_native_patch(self_mock, logits, "g", 3, 0.9)

        calls = self.env["vllm_output_statics"].add_stat.call_args_list
        phases = [call[0][0] for call in calls]
        assert phases == ["apply", "log_softmax", "probs_softmax", "random_sample"]


class TestModulePatchCondition:
    def test_patch_applied_when_env_true(self, fake_sampler_env):
        """ENABLE_VLLM_STAT=true replaces AscendTopKTopPSampler.forward_native."""
        mod = import_module(fake_sampler_env, env_getenv="true", version_is_01011=False)
        Sampler = fake_sampler_env["AscendTopKTopPSampler"]
        assert Sampler.forward_native is mod.forward_native_patch

    def test_patch_not_applied_when_env_false(self, fake_sampler_env):
        """ENABLE_VLLM_STAT=false keeps the original forward_native."""
        mod = import_module(fake_sampler_env, env_getenv="false", version_is_01011=False)
        Sampler = fake_sampler_env["AscendTopKTopPSampler"]
        assert Sampler.forward_native is not mod.forward_native_patch
