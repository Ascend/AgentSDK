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
# Fixture: fake module tree for patch_model_runner_v1
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_model_runner_env():
    # ---- Fake vllm modules ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_config = types.ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = MagicMock
    fake_vllm_sequence = types.ModuleType("vllm.sequence")
    fake_vllm_sequence.IntermediateTensors = MagicMock
    fake_vllm_v1 = types.ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_v1_outputs = types.ModuleType("vllm.v1.outputs")
    fake_vllm_v1_outputs.AsyncModelRunnerOutput = MagicMock
    fake_vllm_v1_outputs.ModelRunnerOutput = MagicMock
    fake_vllm_v1_core = types.ModuleType("vllm.v1.core")
    fake_vllm_v1_core.__path__ = []
    fake_vllm_v1_core_sched = types.ModuleType("vllm.v1.core.sched")
    fake_vllm_v1_core_sched.__path__ = []
    fake_vllm_v1_core_sched_output = types.ModuleType("vllm.v1.core.sched.output")
    # SchedulerOutput only used for type hint
    fake_vllm_distributed = types.ModuleType("vllm.distributed")
    fake_vllm_distributed.__path__ = []
    fake_vllm_distributed_parallel_state = types.ModuleType(
        "vllm.distributed.parallel_state"
    )
    fake_vllm_distributed_parallel_state.get_dp_group = MagicMock()
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.logger = MagicMock()

    # ---- Fake vllm_ascend modules ----
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_worker = types.ModuleType("vllm_ascend.worker")
    fake_vllm_ascend_worker.__path__ = []
    fake_vllm_ascend_worker_model_runner_v1 = types.ModuleType(
        "vllm_ascend.worker.model_runner_v1"
    )

    # Create fake NPUModelRunner class with mock methods
    class NPUModelRunner:
        def __init__(self, *args, **kwargs):
            pass

        def execute_model(self, *args, **kwargs):
            return "exec_output"

        def _generate_process_reqs_hidden_states(self, *args, **kwargs):
            return "hidden_states"

    # Assign the class
    fake_vllm_ascend_worker_model_runner_v1.NPUModelRunner = NPUModelRunner

    # ---- Fake aura base haco_tool ----
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_accuracy = types.ModuleType("aura.base.accuracy")
    fake_aura_base_accuracy.__path__ = []
    fake_aura_base_accuracy_haco_tool = types.ModuleType(
        "aura.base.accuracy.haco_tool"
    )
    # enable_haco will be called with logger, make it return False by default
    fake_aura_base_accuracy_haco_tool.enable_haco = MagicMock(return_value=False)
    fake_aura_base_accuracy_haco_tool.vllm_model_runner_update_haco = MagicMock()

    # ---- aura packages ----
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType(
        "aura.runner.infer_adapter"
    )
    fake_aura_runner_infer_adapter.__path__ = [
        os.path.join(base_path, "runner/infer_adapter")
    ]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [
        os.path.join(base_path, "runner/infer_adapter/vllm/patch")
    ]
    fake_0_11_0_pkg = types.ModuleType(
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0"
    )
    fake_0_11_0_pkg.__path__ = [
        os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_11_0")
    ]

    all_fakes = {
        "vllm": fake_vllm,
        "vllm.config": fake_vllm_config,
        "vllm.sequence": fake_vllm_sequence,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.outputs": fake_vllm_v1_outputs,
        "vllm.v1.core": fake_vllm_v1_core,
        "vllm.v1.core.sched": fake_vllm_v1_core_sched,
        "vllm.v1.core.sched.output": fake_vllm_v1_core_sched_output,
        "vllm.distributed": fake_vllm_distributed,
        "vllm.distributed.parallel_state": fake_vllm_distributed_parallel_state,
        "vllm.logger": fake_vllm_logger,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.worker": fake_vllm_ascend_worker,
        "vllm_ascend.worker.model_runner_v1": fake_vllm_ascend_worker_model_runner_v1,
        "aura.base": fake_aura_base,
        "aura.base.accuracy": fake_aura_base_accuracy,
        "aura.base.accuracy.haco_tool": fake_aura_base_accuracy_haco_tool,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
    }
    for name, mod in all_fakes.items():
        sys.modules[name] = mod

    # Force reloading the module under test each time
    target_module = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_model_runner_v1"
    if target_module in sys.modules:
        del sys.modules[target_module]

    yield {
        "fake_npu_runner": fake_vllm_ascend_worker_model_runner_v1.NPUModelRunner,
        "get_dp_group": fake_vllm_distributed_parallel_state.get_dp_group,
        "haco_tool": fake_aura_base_accuracy_haco_tool,
        "logger": fake_vllm_logger.logger,
    }

    # Cleanup
    for name in list(all_fakes.keys()):
        if name in sys.modules:
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Tests for model_runner_init
# ---------------------------------------------------------------------------
def test_model_runner_init(fake_model_runner_env):
    import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_model_runner_v1 as patch_mod

    # The original __init__ should be the fake NPUModelRunner.__init__
    original_init = patch_mod.original_model_runner_init
    original_init_mock = MagicMock()
    # Replace the reference temporarily to assert call
    patch_mod.original_model_runner_init = original_init_mock

    fake_self = MagicMock()
    fake_config = MagicMock()
    fake_device = MagicMock()

    patch_mod.model_runner_init(fake_self, fake_config, fake_device)

    original_init_mock.assert_called_once_with(fake_self, fake_config, fake_device)
    assert fake_self.sentinel is None


# ---------------------------------------------------------------------------
# Tests for model_runner_model_execute
# ---------------------------------------------------------------------------
class TestModelRunnerExecute:
    @pytest.fixture
    def exec_helpers(self, fake_model_runner_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_model_runner_v1 as patch_mod
        return patch_mod

    def test_execute_when_sentinel_none_and_has_haco(self, exec_helpers, fake_model_runner_env):
        mod = exec_helpers
        # Set HAS_HACO True
        with patch.object(mod, "HAS_HACO", True):
            fake_self = MagicMock()
            fake_self.sentinel = None
            fake_self.model = "model"
            fake_sched_output = MagicMock()
            fake_intermediate = MagicMock()

            # Make original execute return a specific value
            original_exec_mock = MagicMock(return_value="result")
            mod.original_model_execute = original_exec_mock

            result = mod.model_runner_model_execute(fake_self, fake_sched_output, fake_intermediate)

            # Should have called update_haco
            fake_model_runner_env["haco_tool"].vllm_model_runner_update_haco.assert_called_once_with("model")
            assert fake_self.sentinel is fake_model_runner_env["haco_tool"].vllm_model_runner_update_haco.return_value
            original_exec_mock.assert_called_once_with(fake_self, fake_sched_output, fake_intermediate)
            assert result == "result"

    def test_execute_when_sentinel_not_none(self, exec_helpers, fake_model_runner_env):
        mod = exec_helpers
        with patch.object(mod, "HAS_HACO", True):
            fake_self = MagicMock()
            fake_self.sentinel = "already_set"
            original_exec_mock = MagicMock(return_value="result")
            mod.original_model_execute = original_exec_mock

            result = mod.model_runner_model_execute(fake_self, MagicMock())

            # update_haco should NOT be called
            fake_model_runner_env["haco_tool"].vllm_model_runner_update_haco.assert_not_called()
            original_exec_mock.assert_called_once()
            assert result == "result"

    def test_execute_when_has_haco_false(self, exec_helpers, fake_model_runner_env):
        mod = exec_helpers
        # HAS_HACO is already False by default
        fake_self = MagicMock()
        fake_self.sentinel = None
        original_exec_mock = MagicMock(return_value="result")
        mod.original_model_execute = original_exec_mock

        result = mod.model_runner_model_execute(fake_self, MagicMock())

        fake_model_runner_env["haco_tool"].vllm_model_runner_update_haco.assert_not_called()
        original_exec_mock.assert_called_once()
        assert result == "result"


# ---------------------------------------------------------------------------
# Tests for model_runner_generate_process_reqs_hidden_states
# ---------------------------------------------------------------------------
class TestGenerateHiddenStates:
    @pytest.fixture
    def gen_helpers(self, fake_model_runner_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_model_runner_v1 as patch_mod
        return patch_mod

    def test_with_haco(self, gen_helpers):
        mod = gen_helpers
        with patch.object(mod, "HAS_HACO", True):
            fake_self = MagicMock()
            fake_self.sentinel = MagicMock()
            fake_self.input_batch.req_ids = [1, 2]
            args = (
                "attn_metadata", "with_prefill", "num_tokens",
                "input_ids", "positions", "intermediate", "inputs_embeds"
            )

            original_gen_mock = MagicMock(return_value="hidden")
            mod.original_generate_process_reqs_hidden_states = original_gen_mock

            result = mod.model_runner_generate_process_reqs_hidden_states(
                fake_self, *args
            )

            fake_self.sentinel.record_input_id.assert_called_once_with([1, 2])
            original_gen_mock.assert_called_once_with(fake_self, *args)
            fake_self.sentinel.inference_step.assert_called_once()
            assert result == "hidden"

    def test_without_haco(self, gen_helpers):
        mod = gen_helpers
        # HAS_HACO False
        fake_self = MagicMock()
        original_gen_mock = MagicMock(return_value="hidden")
        mod.original_generate_process_reqs_hidden_states = original_gen_mock

        result = mod.model_runner_generate_process_reqs_hidden_states(
            fake_self, None, None, None, None, None, None, None
        )

        # record and inference_step not called
        if fake_self.sentinel:
            fake_self.sentinel.record_input_id.assert_not_called()
            fake_self.sentinel.inference_step.assert_not_called()
        original_gen_mock.assert_called_once()
        assert result == "hidden"


# ---------------------------------------------------------------------------
# Tests for sync_metadata_across_dp
# ---------------------------------------------------------------------------
class TestSyncMetadataAcrossDP:
    @pytest.fixture
    def sync_helpers(self, fake_model_runner_env):
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_model_runner_v1 as patch_mod
        return patch_mod

    def test_dp_size_1(self, sync_helpers):
        mod = sync_helpers
        fake_self = MagicMock()
        fake_self.dp_size = 1
        num_tokens, after_pad, wp, ebo = mod.sync_metadata_across_dp(
            fake_self, 10, True, True
        )
        assert num_tokens == 10
        assert after_pad is None
        assert wp is True
        assert ebo is True

    def test_dp_size_greater_than_1(self, sync_helpers, fake_model_runner_env):
        mod = sync_helpers
        fake_self = MagicMock()
        fake_self.dp_size = 4
        fake_self.dp_rank = 1

        # Mock get_dp_group to return a mock with cpu_group
        dp_group_mock = MagicMock()
        dp_group_mock.cpu_group = "cpu_group"
        fake_model_runner_env["get_dp_group"].return_value = dp_group_mock

        with patch("torch.tensor") as mock_tensor, \
             patch("torch.cat") as mock_cat, \
             patch("torch.max") as mock_max, \
             patch("torch.distributed.all_reduce") as mock_all_reduce, \
             patch("torch.distributed.barrier") as mock_barrier:

            # Configure tensor returns
            num_tokens_tensor = MagicMock()
            flags_tensor = MagicMock()
            packed_tensor = MagicMock()
            num_tokens_across_dp = MagicMock()
            # tensor for num_tokens_after_padding
            pad_tensor = MagicMock()

            mock_tensor.side_effect = [
                num_tokens_tensor,
                flags_tensor,
                pad_tensor,
            ]
            mock_cat.return_value = packed_tensor

            # packed_tensor[:-2] -> num_tokens_across_dp
            packed_tensor.__getitem__ = MagicMock()
            # first call returns num_tokens_across_dp, second call returns synced_flags (slice)
            packed_tensor.__getitem__.side_effect = lambda idx: (
                num_tokens_across_dp if idx == slice(None, -2) else flags_tensor
            )

            # torch.max return a mock with .item() returning max tokens
            max_val_mock = MagicMock()
            max_val_mock.item.return_value = 42
            mock_max.return_value = max_val_mock

            result = mod.sync_metadata_across_dp(
                fake_self, 10, True, True
            )

            # Verify all_reduce and barrier called with correct group
            mock_all_reduce.assert_called_once_with(packed_tensor, group="cpu_group")
            mock_barrier.assert_called_once_with(group="cpu_group")

            # Check return values
            max_tokens, after_pad, global_with_prefill, global_enable_dbo = result
            assert max_tokens == 42
            assert after_pad == pad_tensor
            assert global_with_prefill == True
