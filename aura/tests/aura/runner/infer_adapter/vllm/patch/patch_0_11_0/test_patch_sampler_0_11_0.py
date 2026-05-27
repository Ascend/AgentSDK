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
# Fixture: fake module tree for patch_sampler
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_sampler_env():
    # ---- Fake StatPhase with exact attribute names ----
    class StatPhase:
        post_samper_sample_topk_topp_apply_time = "apply_time"
        post_samper_sample_topk_topp_logits_log_softmax_time = "log_softmax_time"
        post_samper_sample_topk_topp_probs_softmax_time = "probs_softmax_time"
        post_samper_sample_topk_topp_random_sample_time = "random_sample_time"

    # ---- Fake StatTimeUtil ----
    class StatTimeUtil:
        def get_duration(self):
            return 0.123

    # ---- Fake vllm_output_statics ----
    fake_output_statics = MagicMock()
    fake_output_statics.add_stat = MagicMock()

    # ---- Fake vllm_execute_stat module ----
    fake_exec_stat = types.ModuleType(
        "aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat"
    )
    fake_exec_stat.StatTimeUtil = StatTimeUtil
    fake_exec_stat.vllm_output_statics = fake_output_statics
    fake_exec_stat.StatPhase = StatPhase

    # ---- Fake vllm_ascend.sample.sampler ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_sample = types.ModuleType("vllm_ascend.sample")
    fake_vllm_ascend_sample.__path__ = []
    fake_vllm_ascend_sampler = types.ModuleType("vllm_ascend.sample.sampler")

    class AscendTopKTopPSampler:
        def forward_native(self, *args, **kwargs):
            return "original"
        def _apply_top_k_top_p(self, logits, k, p):
            return logits

    fake_vllm_ascend_sampler.AscendTopKTopPSampler = AscendTopKTopPSampler

    # ---- Fake vllm.v1.sample.ops.topk_topp_sampler ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_v1 = types.ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_v1_sample = types.ModuleType("vllm.v1.sample")
    fake_vllm_v1_sample.__path__ = []
    fake_vllm_v1_sample_ops = types.ModuleType("vllm.v1.sample.ops")
    fake_vllm_v1_sample_ops.__path__ = []
    fake_topk_topp_sampler = types.ModuleType("vllm.v1.sample.ops.topk_topp_sampler")
    # random_sample returns a tuple, consistent with real behavior
    fake_topk_topp_sampler.random_sample = MagicMock(return_value=("sampled_output",))

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
    fake_comm_pkg.__path__ = []  # critical to avoid fallback to real filesystem

    all_fakes = {
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.sample": fake_vllm_ascend_sample,
        "vllm_ascend.sample.sampler": fake_vllm_ascend_sampler,
        "vllm": fake_vllm,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.sample": fake_vllm_v1_sample,
        "vllm.v1.sample.ops": fake_vllm_v1_sample_ops,
        "vllm.v1.sample.ops.topk_topp_sampler": fake_topk_topp_sampler,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm": fake_comm_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat": fake_exec_stat,
    }
    for name, mod in all_fakes.items():
        sys.modules[name] = mod

    original_sampler = fake_vllm_ascend_sampler.AscendTopKTopPSampler

    yield {
        "stat_time_util_cls": StatTimeUtil,
        "output_statics": fake_output_statics,
        "stat_phase": StatPhase,
        "sampler_class": original_sampler,
        "random_sample": fake_topk_topp_sampler.random_sample,
        "fake_exec_stat": fake_exec_stat,
    }

    # Cleanup
    for name in list(all_fakes.keys()):
        if name in sys.modules:
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Helper: import the module with controlled environment variables
# ---------------------------------------------------------------------------
def import_sampler_module(env_getenv=None, env_environ_get=None):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_sampler"
    if module_name in sys.modules:
        del sys.modules[module_name]

    if env_getenv is None:
        env_getenv = "False"
    if env_environ_get is None:
        env_environ_get = "0"

    with patch("os.getenv", return_value=env_getenv), patch(
        "os.environ.get", return_value=env_environ_get
    ):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_sampler as mod
        return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestForwardNativePatch:
    def test_processed_logits_mode(self, fake_sampler_env):
        mod = import_sampler_module()

        self_mock = MagicMock()
        self_mock.logprobs_mode = "processed_logits"
        logits = MagicMock()
        logits.softmax = MagicMock(return_value="probs")
        self_mock._apply_top_k_top_p.return_value = logits

        result = mod.forward_native_patch(self_mock, "logits_in", "gen", 3, 0.9)

        self_mock._apply_top_k_top_p.assert_called_once_with("logits_in", 3, 0.9)
        logits.softmax.assert_called_once_with(dim=-1, dtype=torch.float32)
        fake_sampler_env["random_sample"].assert_called_once_with("probs", "gen")
        assert fake_sampler_env["output_statics"].add_stat.call_count == 4

        output, logits_returned = result
        # random_sample returns a tuple, and forward_native_patch wraps it as (output, logits_to_return)
        assert output == ("sampled_output",)
        assert logits_returned is logits

    def test_processed_logprobs_mode(self, fake_sampler_env):
        mod = import_sampler_module()

        self_mock = MagicMock()
        self_mock.logprobs_mode = "processed_logprobs"
        logits = MagicMock()
        logits.softmax.return_value = "probs"
        logits.log_softmax.return_value = "logprobs_result"
        self_mock._apply_top_k_top_p.return_value = logits

        _, logits_returned = mod.forward_native_patch(self_mock, "x", "g", 5, 0.5)
        logits.log_softmax.assert_called_once_with(dim=-1, dtype=torch.float32)
        assert logits_returned == "logprobs_result"

    def test_other_mode_returns_none(self, fake_sampler_env):
        mod = import_sampler_module()

        self_mock = MagicMock()
        self_mock.logprobs_mode = "none"
        logits = MagicMock()
        logits.softmax.return_value = "probs"
        self_mock._apply_top_k_top_p.return_value = logits

        _, logits_returned = mod.forward_native_patch(self_mock, "x", "g", 1, 1)
        assert logits_returned is None

    def test_stat_calls_order(self, fake_sampler_env):
        mod = import_sampler_module()

        self_mock = MagicMock()
        self_mock.logprobs_mode = "processed_logits"
        logits = MagicMock()
        logits.softmax.return_value = "probs"
        self_mock._apply_top_k_top_p.return_value = logits

        mod.forward_native_patch(self_mock, "logits", "gen", 3, 0.9)

        calls = fake_sampler_env["output_statics"].add_stat.call_args_list
        phases = [call[0][0] for call in calls]
        assert phases == [
            "apply_time",
            "log_softmax_time",
            "probs_softmax_time",
            "random_sample_time",
        ]


class TestModulePatchCondition:
    def test_patch_applied_when_stats_enabled_and_level_1(self, fake_sampler_env):
        mod = import_sampler_module(env_getenv="true", env_environ_get="1")
        sampler = fake_sampler_env["sampler_class"]
        assert sampler.forward_native is mod.forward_native_patch

    def test_patch_not_applied_when_stats_disabled(self, fake_sampler_env):
        mod = import_sampler_module(env_getenv="false", env_environ_get="1")
        sampler = fake_sampler_env["sampler_class"]
        assert sampler.forward_native is not mod.forward_native_patch
        assert callable(sampler.forward_native)

    def test_patch_not_applied_when_level_not_1(self, fake_sampler_env):
        mod = import_sampler_module(env_getenv="true", env_environ_get="0")
        sampler = fake_sampler_env["sampler_class"]
        assert sampler.forward_native is not mod.forward_native_patch
        assert callable(sampler.forward_native)
