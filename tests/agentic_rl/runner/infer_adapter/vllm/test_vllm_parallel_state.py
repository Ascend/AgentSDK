#!/usr/bin/env python3
# coding=utf-8
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

import torch
import os
import sys
import types
import unittest
import importlib
from unittest.mock import MagicMock, patch

# ============================================================
# Fake module builder to isolate tests from real dependencies
# ============================================================

def _build_fake_modules():
    """Create and return a dict of fake modules needed for testing."""
    fake_modules = {}
    fake_modules["vertexai"] = types.ModuleType("vertexai")
    fake_modules["openai"] = types.ModuleType("openai")

    # Mock internal utility functions
    fake_utils = types.ModuleType("agentic_rl.base.utils.utils")
    fake_utils.get_cluster_info = lambda: ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    fake_modules["agentic_rl.base.utils.utils"] = fake_utils

    # Mock vLLM environment variables
    fake_envs = types.ModuleType("vllm.envs")
    for attr in ["VLLM_DP_RANK", "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT", "VLLM_PORT"]:
        setattr(fake_envs, attr, None)
    fake_modules["vllm.envs"] = fake_envs

    fake_config_mod = types.ModuleType("vllm.config")
    fake_cfg = MagicMock()
    fake_cfg.parallel_config.data_parallel_size = 1
    fake_config_mod.get_current_vllm_config = lambda: fake_cfg
    fake_modules["vllm.config"] = fake_config_mod

    # Mock vLLM distributed parallel state
    fake_ps = types.ModuleType("vllm.distributed.parallel_state")
    fake_ps._TP = fake_ps._PP = fake_ps._EP = fake_ps._DP = None

    fake_wg = MagicMock()
    fake_wg.local_rank = 0
    fake_wg.device_group = "fake_device_group"
    fake_ps.get_world_group = lambda: fake_wg
    fake_ps.get_pp_group = lambda: MagicMock()
    fake_ps.init_distributed_environment = lambda *a, **k: None

    def init_group(group_ranks, local_rank, backend, **kwargs):
        grp = MagicMock()
        grp.rank_in_group = 0
        return grp

    fake_ps.init_model_parallel_group = init_group

    fake_modules["vllm.distributed.parallel_state"] = fake_ps
    fake_modules["vllm.distributed"] = types.ModuleType("vllm.distributed")
    fake_modules["vllm.distributed"].parallel_state = fake_ps
    fake_modules["vllm"] = types.ModuleType("vllm")
    fake_modules["vllm"].distributed = fake_modules["vllm.distributed"]
    fake_modules["vllm"].envs = fake_envs
    fake_modules["vllm"].config = fake_config_mod

    # Mock vllm_ascend distributed state
    fake_ascend_ps = types.ModuleType("vllm_ascend.distributed.parallel_state")
    fake_ascend_ps.MC2 = None
    fake_modules["vllm_ascend.distributed.parallel_state"] = fake_ascend_ps
    fake_modules["vllm_ascend.distributed"] = types.ModuleType("vllm_ascend.distributed")
    fake_modules["vllm_ascend.distributed"].parallel_state = fake_ascend_ps
    fake_modules["vllm_ascend"] = types.ModuleType("vllm_ascend")
    fake_modules["vllm_ascend"].distributed = fake_modules["vllm_ascend.distributed"]

    return fake_modules


# ============================================================
# Base Test Class with module patching and state reset
# ============================================================

class BaseVllmTest(unittest.TestCase):
    MOD_PATH = "agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state"

    def setUp(self):
        """Install fake modules and reload the module under test."""
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()
        if self.MOD_PATH in sys.modules:
            del sys.modules[self.MOD_PATH]
        self.ps_mod = importlib.import_module(self.MOD_PATH)

        # Reset global state variables in the target module
        self.ps_mod._TP_GROUP_RANKS = None
        self.ps_mod._TP = None
        self.ps_mod._PP = None
        self.ps_mod._EP = None
        self.ps_mod._DP = None

        # Also reset internal vllm.parallel_state globals
        self.ps_mod.ps._TP = None
        self.ps_mod.ps._PP = None
        self.ps_mod.ps._EP = None
        self.ps_mod.ps._DP = None

    def tearDown(self):
        """Stop module patcher."""
        self.module_patcher.stop()


# ============================================================
# Test cases for vllm_parallel_state module
# ============================================================

class TestVllmParallelStateFull(BaseVllmTest):

    # --------------------------------------------------------
    # 1. Basic getter/setter for TP group ranks
    # --------------------------------------------------------
    def test_tp_group_ranks_access(self):
        """Test get_vllm_tp_group_ranks returns correct stored value."""
        self.assertIsNone(self.ps_mod.get_vllm_tp_group_ranks())
        test_ranks = [[0, 1], [2, 3]]
        self.ps_mod._TP_GROUP_RANKS = test_ranks
        self.assertEqual(self.ps_mod.get_vllm_tp_group_ranks(), test_ranks)

    # --------------------------------------------------------
    # 2. initialize_parallel_state: missing WORLD_SIZE raises error
    # --------------------------------------------------------
    def test_init_parallel_state_env_missing(self):
        """When WORLD_SIZE is missing, ValueError should be raised."""
        with patch.dict(os.environ, {}, clear=False):
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]
            with self.assertRaisesRegex(ValueError, "world_size is set to -1"):
                self.ps_mod.initialize_parallel_state()

    def test_initialize_parallel_state_world_size_missing_raises_and_set_env(self):
        """Missing WORLD_SIZE leads to ValueError and sets TORCH_NCCL_AVOID_RECORD_STREAMS."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                self.ps_mod.initialize_parallel_state()

            self.assertIn("world_size is set to -1", str(ctx.exception))
            self.assertEqual(os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS"), "1")

    # --------------------------------------------------------
    # 2.1 initialize_parallel_state normal paths
    # --------------------------------------------------------
    def test_initialize_parallel_state_single_process_calls_initialize_model_parallel(self):
        """Single process world_size=1 should call initialize_model_parallel."""
        with patch.dict(os.environ, {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }, clear=True):
            with patch.object(self.ps_mod, "init_distributed_environment") as mock_init_env, \
                 patch.object(self.ps_mod, "initialize_model_parallel") as mock_init_mp, \
                 patch.object(self.ps_mod.torch.distributed, "get_world_size", return_value=1):

                self.ps_mod.initialize_parallel_state()

                mock_init_env.assert_called_once_with(1, 0, "env://", 0, "hccl")
                mock_init_mp.assert_called_once_with(1, 1, "hccl")

    def test_initialize_parallel_state_multi_process_calls_initialize_model_parallel_for_vllm(self):
        """Multi-process scenario should call initialize_model_parallel_for_vllm."""
        with patch.dict(os.environ, {
            "WORLD_SIZE": "4",
            "RANK": "1",
            "LOCAL_RANK": "0",
        }, clear=True):
            with patch.object(self.ps_mod, "init_distributed_environment") as mock_init_env, \
                 patch.object(self.ps_mod, "initialize_model_parallel_for_vllm") as mock_init_vllm, \
                 patch.object(self.ps_mod.torch.distributed, "get_world_size", return_value=4):

                self.ps_mod.initialize_parallel_state(
                    infer_tensor_model_parallel_size=2,
                    train_tensor_model_parallel_size=4,
                    infer_pipeline_model_parallel_size=1,
                    train_pipeline_model_parallel_size=2,
                    infer_expert_tensor_parallel_size=2,
                    train_expert_model_parallel_size=4,
                    infer_expert_model_parallel_size=2,
                    train_context_model_parallel_size=1,
                )

                mock_init_env.assert_called_once_with(4, 1, "env://", 0, "hccl")
                mock_init_vllm.assert_called_once()

    # --------------------------------------------------------
    # 3. initialize_model_parallel_for_vllm error branches
    # --------------------------------------------------------
    def test_init_vllm_error_not_distributed(self):
        """If torch.distributed not initialized, raise ValueError."""
        with patch.object(torch.distributed, "is_initialized", return_value=False):
            with self.assertRaisesRegex(ValueError, "torch.distributed is not initialized"):
                self.ps_mod.initialize_model_parallel_for_vllm(2)

    def test_init_vllm_error_invalid_type(self):
        """Non-integer argument should raise TypeError."""
        with patch.object(torch.distributed, "is_initialized", return_value=True):
            with self.assertRaisesRegex(TypeError, "must be an integer"):
                self.ps_mod.initialize_model_parallel_for_vllm("not_an_int")

    def test_init_vllm_error_already_initialized(self):
        """If TP group already exists, raise ValueError."""
        with patch.object(torch.distributed, "is_initialized", return_value=True):
            import vllm.distributed.parallel_state as ps
            ps._TP = MagicMock()
            with self.assertRaisesRegex(ValueError, "tensor model parallel group is already initialized"):
                self.ps_mod.initialize_model_parallel_for_vllm(2)

    def test_init_vllm_error_invalid_split(self):
        """Test non-divisible case: world_size=8, train_tp=2, infer_tp=3 -> split invalid."""
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=8), \
             patch.object(torch.distributed, "get_backend", return_value="hccl"):

            with self.assertRaisesRegex(ValueError, "Can't split train tp size"):
                self.ps_mod.initialize_model_parallel_for_vllm(
                    infer_tensor_model_parallel_size=3,
                    train_tensor_model_parallel_size=2
                )

    def test_init_vllm_error_invalid_gather(self):
        """Test 'Can't gather train tp size' branch: world_size=8, train_tp=4, infer_tp=3."""
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=8), \
             patch.object(torch.distributed, "get_backend", return_value="hccl"), \
             patch.object(torch.distributed, "get_rank", return_value=0), \
             patch.object(self.ps_mod, "get_world_group") as mock_wg:

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            mock_wg.return_value = fake_wg

            with self.assertRaisesRegex(ValueError, "Can't gather train tp size"):
                self.ps_mod.initialize_model_parallel_for_vllm(
                    infer_tensor_model_parallel_size=3,
                    train_tensor_model_parallel_size=4,
                    infer_pipeline_model_parallel_size=1,
                    train_pipeline_model_parallel_size=1,
                    infer_expert_model_parallel_size=1,
                    train_expert_model_parallel_size=1,
                    train_context_model_parallel_size=1,
                    rebulid_EP_group=False,
                )

    # --------------------------------------------------------
    # 4. rebuild_EP_group=True path
    # --------------------------------------------------------
    def test_init_vllm_rebuild_ep_group(self):
        """Test rebuilding expert parallel groups (rebulid_EP_group=True)."""
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
            patch.object(torch.distributed, "get_world_size", return_value=8), \
            patch.object(torch.distributed, "get_backend", return_value="hccl"), \
            patch.object(torch.distributed, "get_rank", return_value=0), \
            patch.object(self.ps_mod, "get_world_group") as mock_wg, \
            patch.object(self.ps_mod, "init_model_parallel_group") as mock_init:

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            mock_wg.return_value = fake_wg

            # Create many mock groups to satisfy side effects
            mock_rets = [MagicMock(rank_in_group=0) for _ in range(20)]
            mock_init.side_effect = mock_rets

            with patch.dict(os.environ, {"MASTER_PORT": "29500"}, clear=False):
                self.ps_mod.initialize_model_parallel_for_vllm(
                    infer_tensor_model_parallel_size=2,
                    train_tensor_model_parallel_size=2,
                    infer_expert_model_parallel_size=2,
                    train_expert_model_parallel_size=2,
                    rebulid_EP_group=True
                )

            self.assertEqual(len(self.ps_mod._TP_GROUP_RANKS), 4)
            self.assertEqual(self.ps_mod._TP_GROUP_RANKS[0], [0, 1])
            self.assertGreater(mock_init.call_count, 4)

    def test_init_vllm_rebuild_ep_group_env_set(self):
        """rebuild_EP_group=True should set environment variables for data parallel."""
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=8), \
             patch.object(torch.distributed, "get_backend", return_value="hccl"), \
             patch.object(torch.distributed, "get_rank", return_value=1), \
             patch.object(self.ps_mod, "get_world_group") as mock_wg, \
             patch.object(self.ps_mod, "init_model_parallel_group") as mock_init, \
             patch.object(self.ps_mod, "get_cluster_info") as mock_cluster:

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            mock_wg.return_value = fake_wg

            fake_dp_group = MagicMock()
            fake_dp_group.rank_in_group = 0

            mock_init.side_effect = [
                MagicMock(name="TP_GROUP"),
                MagicMock(name="PP_GROUP"),
                MagicMock(name="EP_GROUP"),
                MagicMock(name="MC2_GROUP"),
                fake_dp_group,
            ]

            mock_cluster.return_value = [
                "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4",
                "10.0.0.5", "10.0.0.6", "10.0.0.7", "10.0.0.8",
            ]

            with patch.dict(os.environ, {"MASTER_PORT": "29500"}, clear=True):
                self.ps_mod.initialize_model_parallel_for_vllm(
                    infer_tensor_model_parallel_size=2,
                    train_tensor_model_parallel_size=2,
                    infer_pipeline_model_parallel_size=1,
                    train_pipeline_model_parallel_size=1,
                    infer_expert_model_parallel_size=2,
                    train_expert_model_parallel_size=1,
                    train_context_model_parallel_size=1,
                    rebulid_EP_group=True,
                )

                self.assertIn("VLLM_DP_RANK", os.environ)
                self.assertIn("VLLM_DP_MASTER_PORT", os.environ)
                self.assertIn("VLLM_DP_MASTER_IP", os.environ)
                self.assertIn("VLLM_PORT", os.environ)

            self.assertEqual(mock_init.call_count, 5)

    # --------------------------------------------------------
    # 5. initialize_model_parallel logic tests
    # --------------------------------------------------------
    def test_initialize_model_parallel_tp_pp_not_match_still_works(self):
        """Even if tp*pp != world_size, no assertion is raised (code allows it)."""
        with patch.object(self.ps_mod.torch.distributed, "is_initialized", return_value=True), \
            patch.object(self.ps_mod.torch.distributed, "get_world_size", return_value=8), \
            patch.object(self.ps_mod.torch.distributed, "get_backend", return_value="hccl"), \
            patch.object(self.ps_mod.ps, "get_world_group") as mock_wg, \
            patch.object(self.ps_mod, "init_model_parallel_group") as mock_init:

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            fake_wg.device_group = "fake_device_group"
            mock_wg.return_value = fake_wg

            mock_init.return_value = MagicMock()

            # tp=4, pp=4, product=16 != world_size=8 -> should still run
            self.ps_mod.initialize_model_parallel(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=4,
                backend=None
            )

            self.assertEqual(mock_init.call_count, 2)

    def test_initialize_model_parallel_already_init_pp(self):
        """If PP group already exists, raise ValueError."""
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
            patch.object(torch.distributed, "get_world_size", return_value=4), \
            patch.object(torch.distributed, "get_backend", return_value="hccl"):

            self.ps_mod._PP = MagicMock()
            with self.assertRaisesRegex(ValueError, "pipeline model parallel group is already initialized"):
                self.ps_mod.initialize_model_parallel(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2
                )

    def test_initialize_model_parallel_distributed_not_initialized(self):
        """When torch.distributed not initialized, raise ValueError."""
        with patch.object(self.ps_mod.torch.distributed, "is_initialized", return_value=False):
            with self.assertRaisesRegex(ValueError, "torch.distributed is not initialized"):
                self.ps_mod.initialize_model_parallel()

    def test_initialize_model_parallel_normal_path(self):
        """Normal initialization of TP and PP groups."""
        with patch.object(self.ps_mod.torch.distributed, "is_initialized", return_value=True), \
             patch.object(self.ps_mod.torch.distributed, "get_world_size", return_value=4), \
             patch.object(self.ps_mod, "init_model_parallel_group") as mock_init_group, \
             patch.object(self.ps_mod, "get_world_group") as mock_world_group:

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            fake_wg.device_group = "dg"
            mock_world_group.return_value = fake_wg

            self.ps_mod.initialize_model_parallel(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                backend="hccl",
            )

            self.assertEqual(mock_init_group.call_count, 2)

    # --------------------------------------------------------
    # 6. initialize_model_parallel_for_vllm normal path
    # --------------------------------------------------------
    def test_initialize_model_parallel_for_vllm_normal_path(self):
        """Full normal execution with rebulid_EP_group=False."""
        with patch.object(self.ps_mod.torch.distributed, "is_initialized", return_value=True), \
             patch.object(self.ps_mod.torch.distributed, "get_world_size", return_value=4), \
             patch.object(self.ps_mod.torch.distributed, "get_backend", return_value="hccl"), \
             patch.object(self.ps_mod.torch.distributed, "get_rank", return_value=0), \
             patch.object(self.ps_mod, "get_world_group") as mock_world_group, \
             patch.object(self.ps_mod, "init_model_parallel_group") as mock_init_group, \
             patch.object(self.ps_mod, "get_cluster_info", return_value=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]):

            fake_wg = MagicMock()
            fake_wg.local_rank = 0
            mock_world_group.return_value = fake_wg

            fake_dp_group = MagicMock()
            fake_dp_group.rank_in_group = 0

            mock_init_group.side_effect = [
                MagicMock(name="TP_GROUP"),
                MagicMock(name="PP_GROUP"),
                MagicMock(name="EP_GROUP"),
                MagicMock(name="MC2_GROUP"),
                fake_dp_group,
            ]

            with patch.dict(os.environ, {"MASTER_PORT": "29500"}, clear=True):
                self.ps_mod.initialize_model_parallel_for_vllm(
                    infer_tensor_model_parallel_size=2,
                    train_tensor_model_parallel_size=2,
                    infer_pipeline_model_parallel_size=1,
                    train_pipeline_model_parallel_size=1,
                    infer_expert_tensor_parallel_size=1,
                    train_expert_model_parallel_size=1,
                    infer_expert_model_parallel_size=1,
                    train_context_model_parallel_size=1,
                    rebulid_EP_group=False,
                )

                self.assertEqual(self.ps_mod._TP_GROUP_RANKS, [[0, 1], [2, 3]])
                self.assertEqual(os.environ["VLLM_DP_RANK"], "0")
                self.assertEqual(os.environ["VLLM_DP_MASTER_PORT"], "29501")
                self.assertEqual(os.environ["VLLM_PORT"], "29501")
                self.assertEqual(os.environ["VLLM_DP_MASTER_IP"], "10.0.0.1")

            self.assertGreaterEqual(mock_init_group.call_count, 5)


if __name__ == "__main__":
    unittest.main()