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

import sys
import types
import unittest
import importlib
from unittest.mock import patch, MagicMock


def _ensure_torch_pytree_api():
    """Add missing register_pytree_node to torch.utils._pytree if absent."""
    try:
        import torch
    except Exception:
        return

    # Ensure torch.utils exists
    if not hasattr(torch, "utils"):
        torch.utils = types.SimpleNamespace()

    # Ensure torch.utils._pytree exists
    if not hasattr(torch.utils, "_pytree") or torch.utils._pytree is None:
        torch.utils._pytree = types.SimpleNamespace()

    # Patch missing function
    if not hasattr(torch.utils._pytree, "register_pytree_node"):
        def _dummy_register_pytree_node(*args, **kwargs):
            return None
        torch.utils._pytree.register_pytree_node = _dummy_register_pytree_node


_ensure_torch_pytree_api()


# ============================================================
# Fake vllm module tree installer (avoid real vllm import)
# ============================================================

def _install_fake_vllm_modules():
    """
    Create a complete fake vLLM module tree to satisfy imports:
        vllm
        vllm.model_executor
        vllm.model_executor.models
        vllm.model_executor.models.ModelRegistry
        vllm.model_executor.layers.linear
        vllm.model_executor.layers.fused_moe.layer
        vllm.model_executor.layers.vocab_parallel_embedding
    """
    # Fake ModelRegistry with supported architectures
    fake_models_mod = types.ModuleType("vllm.model_executor.models")

    class FakeModelRegistry:
        @staticmethod
        def get_supported_archs():
            return ["llama", "qwen", "deepseek"]

    fake_models_mod.ModelRegistry = FakeModelRegistry

    # Fake linear layer classes (each has weight_loader attribute to be patched)
    fake_linear_mod = types.ModuleType("vllm.model_executor.layers.linear")

    class ColumnParallelLinear:
        weight_loader = None

    class MergedColumnParallelLinear:
        weight_loader = None

    class QKVParallelLinear:
        weight_loader = None

    class RowParallelLinear:
        weight_loader = None

    class ReplicatedLinear:
        weight_loader = None

    fake_linear_mod.ColumnParallelLinear = ColumnParallelLinear
    fake_linear_mod.MergedColumnParallelLinear = MergedColumnParallelLinear
    fake_linear_mod.QKVParallelLinear = QKVParallelLinear
    fake_linear_mod.RowParallelLinear = RowParallelLinear
    fake_linear_mod.ReplicatedLinear = ReplicatedLinear

    # Fake fused MoE layer
    fake_fused_layer_mod = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")

    class FusedMoE:
        weight_loader = None

    fake_fused_layer_mod.FusedMoE = FusedMoE

    # Fake vocab embedding classes
    fake_vocab_mod = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")

    class VocabParallelEmbedding:
        weight_loader = None

    class ParallelLMHead:
        weight_loader = None

    fake_vocab_mod.VocabParallelEmbedding = VocabParallelEmbedding
    fake_vocab_mod.ParallelLMHead = ParallelLMHead

    # Assemble the package hierarchy
    fake_layers_mod = types.ModuleType("vllm.model_executor.layers")
    fake_layers_mod.linear = fake_linear_mod

    fake_fused_moe_mod = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fake_fused_moe_mod.layer = fake_fused_layer_mod

    fake_model_executor_mod = types.ModuleType("vllm.model_executor")
    fake_model_executor_mod.models = fake_models_mod
    fake_model_executor_mod.layers = fake_layers_mod

    fake_vllm_mod = types.ModuleType("vllm")
    fake_vllm_mod.model_executor = fake_model_executor_mod

    # Register all fake modules into sys.modules
    sys.modules["vllm"] = fake_vllm_mod
    sys.modules["vllm.model_executor"] = fake_model_executor_mod
    sys.modules["vllm.model_executor.models"] = fake_models_mod

    sys.modules["vllm.model_executor.layers"] = fake_layers_mod
    sys.modules["vllm.model_executor.layers.linear"] = fake_linear_mod

    sys.modules["vllm.model_executor.layers.fused_moe"] = fake_fused_moe_mod
    sys.modules["vllm.model_executor.layers.fused_moe.layer"] = fake_fused_layer_mod

    sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"] = fake_vocab_mod


_install_fake_vllm_modules()


def _reload_target_module():
    """
    Reload the module under test after fakes are installed.
    Returns the reloaded module.
    """
    mod_name = "agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ============================================================
# Unit tests for VllmMegatronWeightLoaders
# ============================================================

class TestVllmMegatronWeightLoaders(unittest.TestCase):

    def test_init_register_model_loader_called(self):
        """
        Constructor should call register_model_loader for each known model architecture.
        """
        from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader

        with patch.object(BaseMegatronWeightLoader, "register_model_loader") as mock_register:
            mod = _reload_target_module()
            loader_cls = mod.VllmMegatronWeightLoaders

            loader = loader_cls()
            self.assertIsNotNone(loader)

            self.assertGreaterEqual(mock_register.call_count, 1)

            # Verify each expected architecture is registered
            expected_calls = [
                ("LlamaForCausalLM", BaseMegatronWeightLoader.llama_megatron_core_weight_loader),
                ("Qwen2ForCausalLM", BaseMegatronWeightLoader.qwen_megatron_weight_loader),
                ("DeepseekV3ForCausalLM", BaseMegatronWeightLoader.deepseek_megatron_weight_loader),
                ("Qwen2_5_VLForConditionalGeneration", BaseMegatronWeightLoader.qwen_vl_megatron_weight_loader),
            ]
            for arch, loader_func in expected_calls:
                mock_register.assert_any_call(arch, loader_func)

    def test_get_supported_architectures(self):
        """
        get_supported_architectures should delegate to vLLM's ModelRegistry.
        """
        mod = _reload_target_module()
        loader = mod.VllmMegatronWeightLoaders()

        archs = loader.get_supported_architectures()
        self.assertEqual(archs, ["llama", "qwen", "deepseek"])

    def test_update_megatron_weight_loader(self):
        """
        update_megatron_weight_loader should assign BaseMegatronWeightLoader.parallel_weight_loader
        to the weight_loader attribute of each vLLM layer class.
        """
        from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader

        mod = _reload_target_module()
        loader = mod.VllmMegatronWeightLoaders()

        loader.update_megatron_weight_loader()

        # Import fake vLLM classes after they are installed
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
            ReplicatedLinear,
        )
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
            ParallelLMHead,
        )

        # All these classes should have weight_loader set to the same function
        target_loader = BaseMegatronWeightLoader.parallel_weight_loader
        self.assertEqual(ColumnParallelLinear.weight_loader, target_loader)
        self.assertEqual(MergedColumnParallelLinear.weight_loader, target_loader)
        self.assertEqual(QKVParallelLinear.weight_loader, target_loader)
        self.assertEqual(RowParallelLinear.weight_loader, target_loader)
        self.assertEqual(ReplicatedLinear.weight_loader, target_loader)
        self.assertEqual(FusedMoE.weight_loader, target_loader)
        self.assertEqual(VocabParallelEmbedding.weight_loader, target_loader)
        self.assertEqual(ParallelLMHead.weight_loader, target_loader)

    def test_register_model_loader_custom(self):
        """
        Instance method register_model_loader should forward to the base class.
        """
        mod = _reload_target_module()
        loader = mod.VllmMegatronWeightLoaders()

        def custom_loader(model, state_dict, prefix, tensor_parallel_size, pipeline_parallel_size):
            return None

        with patch.object(loader, "register_model_loader") as mock_register:
            loader.register_model_loader("CustomModel", custom_loader)
            mock_register.assert_called_once_with("CustomModel", custom_loader)


if __name__ == "__main__":
    unittest.main()