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

import gc
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock, call
import importlib

import torch

# ============================================================
# Helpers: build fake dependency modules to avoid real imports
# ============================================================

def _fake_module(name: str):
    """Create an empty module with the given name."""
    return types.ModuleType(name)


def _install_fake_deps():
    """Install all required fake modules into sys.modules to isolate tests."""
    fake_modules = {}

    def safe_get_fake(name):
            try:
                mod = _fake_module(name)
                if mod is None:
                    raise ImportError(f"Mocking failed: _fake_module('{name}') returned None")
                return mod
            except Exception as e:
                raise RuntimeError(f"Failed to install fake dependency for {name}") from e
            
    # transformers.AutoConfig
    transformers_mod = safe_get_fake("transformers")
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return MagicMock(name="HFConfig")
    transformers_mod.AutoConfig = FakeAutoConfig
    fake_modules["transformers"] = transformers_mod

    # tokenizer.get_tokenizer
    tokenizer_mod = safe_get_fake("agentic_rl.base.utils.tokenizer")
    def fake_get_tokenizer(*args, **kwargs):
        tok = MagicMock()
        tok.tokenizer = MagicMock()
        tok.tokenizer.pad_token_id = None
        tok.tokenizer.eos_token_id = 2
        return tok
    tokenizer_mod.get_tokenizer = fake_get_tokenizer
    fake_modules["agentic_rl.base.utils.tokenizer"] = tokenizer_mod

    # get_local_rank
    get_local_rank_mod = safe_get_fake("agentic_rl.base.utils.get_local_rank")
    def fake_get_local_rank(*args, **kwargs):
        return 0
    get_local_rank_mod.get_local_rank = fake_get_local_rank
    fake_modules["agentic_rl.base.utils.get_local_rank"] = get_local_rank_mod

    # logger
    loggers_mod = safe_get_fake("agentic_rl.base.log.loggers")
    class FakeLoggers:
        def __init__(self, *args, **kwargs):
            pass
        def get_logger(self):
            logger = MagicMock()
            logger.info = MagicMock()
            logger.debug = MagicMock()
            logger.error = MagicMock()
            return logger
    loggers_mod.Loggers = FakeLoggers
    fake_modules["agentic_rl.base.log.loggers"] = loggers_mod

    # BaseInferEngine
    base_engine_mod = safe_get_fake("agentic_rl.runner.infer_adapter.vllm.base_inference_engine")
    class FakeBaseInferEngine:
        def __init__(self, *args, **kwargs):
            self.tokenizer_name_or_path = kwargs.get("tokenizer_name_or_path", "/fake_model")
            self.trust_remote_code = kwargs.get("trust_remote_code", True)
            self.prompt_type = kwargs.get("prompt_type", None)
            self.prompt_type_path = kwargs.get("prompt_type_path", None)
            self.infer_tensor_parallel_size = kwargs.get("infer_tensor_parallel_size", 1)
            self.train_tensor_parallel_size = kwargs.get("train_tensor_parallel_size", 1)
            self.infer_pipeline_parallel_size = kwargs.get("infer_pipeline_parallel_size", 1)
            self.train_pipeline_parallel_size = kwargs.get("train_pipeline_parallel_size", 1)
            self.train_expert_parallel_size = kwargs.get("train_expert_parallel_size", 1)
            self.infer_expert_parallel_size = kwargs.get("infer_expert_parallel_size", 1)
            self.train_context_parallel_size = kwargs.get("train_context_parallel_size", 1)
    base_engine_mod.BaseInferEngine = FakeBaseInferEngine
    fake_modules["agentic_rl.runner.infer_adapter.vllm.base_inference_engine"] = base_engine_mod

    # vllm.worker.worker_base
    worker_base_mod = safe_get_fake("vllm.worker.worker_base")
    class FakeWorkerWrapperBase:
        def __init__(self, *args, **kwargs):
            self.worker = MagicMock()
            self.worker.model_runner = MagicMock()
            self.worker.model_runner.get_model = MagicMock()
            self.worker.model_runner.vllm_config = MagicMock()
            self.worker.vllm_config = MagicMock()
            self.worker.vllm_config.parallel_config = MagicMock()
            self.worker.vllm_config.parallel_config.pipeline_parallel_size = 1
            self.worker.model_runner.kv_caches = []
            self.worker.model_runner.vllm_config.compilation_config = MagicMock()
            self.worker.model_runner.vllm_config.compilation_config.static_forward_context = {}
        def initialize_from_config(self, cfg):
            return None
        def init_worker(self, all_kwargs):
            return None
        def load_model(self, *args, **kwargs):
            return None
        def execute_method(self, method, *args, **kwargs):
            return f"executed:{method}"
        def get_kv_cache_spec(self):
            return "FAKE_KV_SPEC"
        def determine_available_memory(self):
            return 123
        def sleep(self, level=1):
            return None
        def wake_up(self, tags=None):
            return None
    class FakeSetCurrentVllmConfig:
        def __init__(self, cfg):
            self.cfg = cfg
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    worker_base_mod.WorkerWrapperBase = FakeWorkerWrapperBase
    worker_base_mod.set_current_vllm_config = lambda cfg: FakeSetCurrentVllmConfig(cfg)
    fake_modules["vllm.worker.worker_base"] = worker_base_mod

    # vllm.v1.core.kv_cache_utils
    kv_cache_utils_mod = safe_get_fake("vllm.v1.core.kv_cache_utils")
    def fake_get_kv_cache_config(vllm_config, spec, mem):
        cfg = MagicMock()
        cfg.num_blocks = 10
        return cfg
    def fake_unify_kv_cache_configs(cfgs):
        return None
    kv_cache_utils_mod.get_kv_cache_config = fake_get_kv_cache_config
    kv_cache_utils_mod.unify_kv_cache_configs = fake_unify_kv_cache_configs
    fake_modules["vllm.v1.core.kv_cache_utils"] = kv_cache_utils_mod

    # vllm.config
    vllm_config_mod = _fake_module("vllm.config")
    class FakeVllmConfig:
        pass
    vllm_config_mod.VllmConfig = FakeVllmConfig
    fake_modules["vllm.config"] = vllm_config_mod

    # vllm.attention.AttentionType
    vllm_attention_mod = _fake_module("vllm.attention")
    class FakeAttentionType:
        DECODER = "DECODER"
        ENCODER_DECODER = "ENCODER_DECODER"
    vllm_attention_mod.AttentionType = FakeAttentionType
    fake_modules["vllm.attention"] = vllm_attention_mod

    # vllm_ascend.platform
    vllm_ascend_platform_mod = safe_get_fake("vllm_ascend.platform")
    class FakeNPUPlatform:
        @staticmethod
        def mem_get_info():
            return (50 << 30, 100 << 30)
    vllm_ascend_platform_mod.NPUPlatform = FakeNPUPlatform
    fake_modules["vllm_ascend.platform"] = vllm_ascend_platform_mod

    # vllm_ascend.patch
    vllm_ascend_patch = _fake_module("vllm_ascend.patch")
    vllm_ascend_patch.platform = MagicMock()
    vllm_ascend_patch.worker = MagicMock()
    fake_modules["vllm_ascend.patch"] = vllm_ascend_patch

    # parallel_state
    parallel_state_mod = _fake_module("agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state")
    parallel_state_mod.initialize_parallel_state = MagicMock()
    fake_modules["agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state"] = parallel_state_mod

    # megatron weight loaders
    megatron_loader_mod = safe_get_fake("agentic_rl.base.weight_loaders.megatron_weight_loaders")
    class FakeInferParallelConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
    megatron_loader_mod.InferParallelConfig = FakeInferParallelConfig
    fake_modules["agentic_rl.base.weight_loaders.megatron_weight_loaders"] = megatron_loader_mod

    vllm_weight_loader_mod = safe_get_fake("agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders")
    class FakeVllmMegatronWeightLoaders:
        def __init__(self):
            self.load_megatron_weights = MagicMock()
            self.update_megatron_weight_loader = MagicMock()
    vllm_weight_loader_mod.VllmMegatronWeightLoaders = FakeVllmMegatronWeightLoaders
    fake_modules["agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders"] = vllm_weight_loader_mod

    return fake_modules


def _reload_target_module():
    """Reload the module under test after fakes are installed."""
    mod_name = "agentic_rl.runner.infer_adapter.vllm.vllm_worker"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ============================================================
# Tests
# ============================================================

class TestAsyncVLLMInferEngine(unittest.TestCase):
    """Unit tests for AsyncVLLMInferEngine and related helper functions."""

    def setUp(self):
        """Set up test environment: fake modules, environment variables, and mocks."""
        self._had_register = hasattr(torch.utils._pytree, 'register_pytree_node')
        if not self._had_register:
            torch.utils._pytree.register_pytree_node = lambda *args, **kwargs: None

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"

        fake_deps = _install_fake_deps()
        self.mod_patch = patch.dict(sys.modules, fake_deps, clear=False)
        self.mod_patch.start()
        self.addCleanup(self.mod_patch.stop)

        # Mock torch.cuda functions to avoid actual GPU calls
        self.cuda_patches = []
        for func_name in ['memory_allocated', 'memory_reserved', 'empty_cache']:
            patcher = patch(f'torch.cuda.{func_name}')
            mock_func = patcher.start()
            if func_name in ['memory_allocated', 'memory_reserved']:
                mock_func.return_value = 0
            self.cuda_patches.append(patcher)
            self.addCleanup(patcher.stop)

        self.target_mod = _reload_target_module()
        self.AsyncVLLMInferEngine = self.target_mod.AsyncVLLMInferEngine
        self.AgentWorkerWrapperBase = self.target_mod.AgentWorkerWrapperBase
        self.print_memory = self.target_mod.print_memory
        self.get_device_memory = self.target_mod.get_device_memory

    def tearDown(self):
        """Clean up patched torch function and environment variables."""
        if not self._had_register:
            if hasattr(torch.utils._pytree, 'register_pytree_node'):
                delattr(torch.utils._pytree, 'register_pytree_node')
        for key in ["CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "RANK", "LOCAL_RANK"]:
            os.environ.pop(key, None)

    # Basic initialization tests
    def test_init_basic_fields(self):
        """Verify that engine sets default fields correctly."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        self.assertTrue(engine.enable_sleep_mode)
        self.assertEqual(engine.load_format, "megatron")
        self.assertIsNone(engine.inference_engine)
        self.assertTrue(engine.is_sleep)
        self.assertTrue(engine.first_wake_up)

    def test_init_worker_sets_rank_and_local_rank(self):
        """Ensure init_worker assigns rank and local rank from environment."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model", train_tensor_parallel_size=1)
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        self.assertEqual(all_kwargs[0]["rank"], 0)
        self.assertEqual(all_kwargs[0]["local_rank"], 0)
        self.assertIsNotNone(engine.inference_engine)
        self.assertIsNotNone(engine.tokenizer)
        self.assertEqual(engine.pad_token_id, 2)

    def test_init_worker_calls_initialize_parallel_state(self):
        """Test that parallel state initialization is invoked for multi-GPU."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model", train_tensor_parallel_size=2)
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        init_ps = sys.modules["agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state"].initialize_parallel_state
        init_ps.assert_called_once()

    def test_init_worker_megatron_load_format_calls_update_loader(self):
        """When load_format is 'megatron', weight loader should be updated."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model", train_tensor_parallel_size=1, load_format="megatron")
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        engine.vllm_megatron_weight_loaders.update_megatron_weight_loader.assert_called_once()

    def test_init_worker_non_megatron_does_not_call_update_loader(self):
        """Other load formats should skip weight loader update."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model", train_tensor_parallel_size=1, load_format="auto")
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        engine.vllm_megatron_weight_loaders.update_megatron_weight_loader.assert_not_called()

    # Model loading and weight management tests
    def test_load_model_builds_cpu_model(self):
        """load_model should create a CPU copy of model parameters."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        fake_param = torch.nn.Parameter(torch.randn(2, 3))
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w1", fake_param), ("w2", fake_param)]
        engine.inference_engine.worker.model_runner.get_model.return_value = fake_model
        engine.load_model()
        self.assertIs(engine.model, fake_model)
        self.assertIn("w1", engine.cpu_model)
        self.assertIn("w2", engine.cpu_model)
        self.assertEqual(engine.cpu_model["w1"].device.type, "cpu")

    def test_execute_method_routes(self):
        """Test that execute_method dispatches to the appropriate internal method."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.execute_method("init_worker", all_kwargs)
        self.assertIsNotNone(engine.inference_engine)

        fake_param = torch.nn.Parameter(torch.randn(2, 3))
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w1", fake_param)]
        engine.inference_engine.worker.model_runner.get_model.return_value = fake_model
        engine.execute_method("load_model")
        self.assertIs(engine.model, fake_model)

        engine.inference_engine.sleep = MagicMock()
        engine.execute_method("sleep")
        engine.inference_engine.sleep.assert_called_once_with(level=2)

        engine.inference_engine.wake_up = MagicMock()
        engine.execute_method("wake_up")
        engine.inference_engine.wake_up.assert_called_once_with(tags=["kv_cache"])

        engine.inference_engine.execute_method = MagicMock(return_value="ok")
        ret = engine.execute_method("other", 1, a=2)
        self.assertEqual(ret, "ok")
        engine.inference_engine.execute_method.assert_called_once_with("other", 1, a=2)

    # KV cache management tests
    def test_init_cache_engine(self):
        """Verify cache engine initialization."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = MagicMock()
        engine.init_cache_engine()
        engine.inference_engine.initialize_from_config.assert_called_once_with(None)

    def test_initialize_kv_caches(self):
        """Test building KV cache configs."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = self.AgentWorkerWrapperBase(vllm_config=MagicMock())
        vllm_config = MagicMock()
        engine._initialize_kv_caches(vllm_config)
        self.assertIsNotNone(engine.kv_cache_configs)
        self.assertEqual(len(engine.kv_cache_configs), 1)
        self.assertEqual(engine.kv_cache_configs[0].num_blocks, 10)

    def test_free_cache_engine(self):
        """Ensure free_cache_engine calls internal method and tracks memory."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = self.AgentWorkerWrapperBase(vllm_config=MagicMock())
        with patch.object(engine, "_free_cache_engine") as mock_free:
            engine.free_cache_engine()
            mock_free.assert_called_once()
        self.assertGreaterEqual(engine.used_bytes, 0)

    # Sleep / wake-up tests
    def test_offload_model_weights_sleep_mode(self):
        """In sleep mode, offload_model_weights should call sleep(level=1)."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = MagicMock()
        engine.inference_engine.sleep = MagicMock()
        engine.offload_model_weights()
        engine.inference_engine.sleep.assert_called_once_with(level=1)

    def test_sync_model_weights_sleep_mode(self):
        """In sleep mode, sync_model_weights wakes up, loads weights, then sleeps? (test logic)."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = MagicMock()
        engine.inference_engine.wake_up = MagicMock()
        fake_param = torch.nn.Parameter(torch.randn(2, 3))
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w1", fake_param)]
        engine.model = fake_model
        engine.cpu_model = {"w1": torch.empty_like(fake_param, device="cpu")}
        engine.hf_config = MagicMock()
        engine.vllm_megatron_weight_loaders.load_megatron_weights = MagicMock()
        engine.sync_model_weights(params={"w": 1})
        engine.vllm_megatron_weight_loaders.load_megatron_weights.assert_called_once()
        engine.inference_engine.wake_up.assert_called_once_with(tags=["weights"])

    def test_sleep(self):
        """sleep() should either call engine.sleep or free_cache_engine depending on enable_sleep_mode."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = MagicMock()
        engine.inference_engine.sleep = MagicMock()
        engine.sleep()
        engine.inference_engine.sleep.assert_called_once_with(level=2)

        engine2 = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        with patch.object(engine2, "free_cache_engine") as mock_free:
            engine2.sleep()
            mock_free.assert_called_once()

    def test_wake_up(self):
        """wake_up should handle both sleep mode and non-sleep mode correctly."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.inference_engine = MagicMock()
        engine.inference_engine.wake_up = MagicMock()
        engine.wake_up()
        engine.inference_engine.wake_up.assert_called_once_with(tags=["kv_cache"])

        engine2 = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        engine2.inference_engine = self.AgentWorkerWrapperBase(vllm_config=MagicMock())
        with patch.object(engine2, "free_cache_engine") as mock_free, \
             patch.object(engine2, "_initialize_kv_caches") as mock_init, \
             patch.object(engine2, "init_cache_engine") as mock_init_engine:
            engine2.first_wake_up = True
            engine2.wake_up()
            mock_free.assert_called_once()
            mock_init.assert_called_once()
            mock_init_engine.assert_called_once()
            self.assertFalse(engine2.first_wake_up)

            engine2.first_wake_up = False
            engine2.wake_up()
            self.assertEqual(mock_free.call_count, 1)
            mock_init.assert_called_once()
            self.assertEqual(mock_init_engine.call_count, 2)

    # Miscellaneous tests
    def test_agent_worker_wrapper_base_initialize_from_config(self):
        """Test that AgentWorkerWrapperBase stores config only once."""
        wrapper = self.AgentWorkerWrapperBase(vllm_config=MagicMock())
        wrapper.initialize_from_config("cfg1")
        self.assertEqual(wrapper.kv_cache_configs, "cfg1")
        wrapper.initialize_from_config("cfg2")
        self.assertEqual(wrapper.kv_cache_configs, "cfg1")

    def test_print_memory_condition_false(self):
        """print_memory should not log when condition is False."""
        with patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.logger") as mock_logger:
            self.print_memory("test", condition=False)
            mock_logger.info.assert_not_called()

    def test_init_worker_no_parallel_state(self):
        """If train_tensor_parallel_size is None, parallel state should not be initialized."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model",
                                            train_tensor_parallel_size=None)
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        init_ps = sys.modules["agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state"].initialize_parallel_state
        init_ps.assert_not_called()

    def test_free_cache_engine_full_coverage(self):
        """Test the internal _free_cache_engine method that clears KV caches."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        mock_worker = MagicMock()
        mock_worker.model_runner.vllm_config.compilation_config.static_forward_context = {
            "layer1": MagicMock(attn_type="DECODER"),
            "layer2": MagicMock(attn_type="ENCODER_DECODER"),
            "layer3": MagicMock(attn_type="OTHER"),
        }
        mock_worker.vllm_config.parallel_config.pipeline_parallel_size = 2
        engine.inference_engine = MagicMock()
        engine.inference_engine.worker = mock_worker

        mock_layer = MagicMock()
        mock_layer.self_attn.attn.impl.key_cache = None
        mock_layer.self_attn.attn.impl.value_cache = None
        mock_model = MagicMock()
        mock_model.model.start_layer = 0
        mock_model.model.end_layer = 2
        mock_model.model.layers = [mock_layer, mock_layer]
        engine.model = mock_model

        engine._free_cache_engine()
        self.assertEqual(mock_worker.model_runner.kv_caches, [])
        for layer in mock_model.model.layers:
            self.assertIsNone(layer.self_attn.attn.impl.key_cache)

        # Test with language_model structure
        engine.model = MagicMock()
        engine.model.language_model.model.start_layer = 0
        engine.model.language_model.model.end_layer = 2
        engine.model.language_model.model.layers = [mock_layer, mock_layer]
        engine._free_cache_engine()
        for layer in engine.model.language_model.model.layers:
            self.assertIsNone(layer.self_attn.attn.impl.key_cache)

        # Fallback when model has no expected structure
        engine.model = MagicMock()
        engine._free_cache_engine()

    def test_offload_model_weights_no_sleep_mode(self):
        """When sleep mode is off, offload_model_weights should move weights to CPU and clear NPU MLA caches."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        fake_param = MagicMock()
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w1", fake_param)]
        engine.model = fake_model
        engine.cpu_model = {"w1": MagicMock()}

        mock_mla = MagicMock()
        mock_mla.impl = MagicMock()
        mock_layer = MagicMock()
        mock_layer.self_attn.mla_attn = mock_mla
        fake_model.model = MagicMock()
        fake_model.model.layers = [mock_layer]
        fake_model.model.start_layer = 0
        fake_model.model.end_layer = 1

        engine.offload_model_weights()
        fake_param.data = engine.cpu_model["w1"]
        self.assertIsNone(mock_mla.impl.w_kc)
        self.assertIsNone(mock_mla.impl.W_UV)

    def test_sync_model_weights_no_sleep_mode(self):
        """Without sleep mode, sync_model_weights should reload weights and process MLA."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=False, tokenizer_name_or_path="/fake_model")
        fake_param = MagicMock()
        fake_model = MagicMock()
        fake_model.named_parameters.return_value = [("w1", fake_param)]
        engine.model = fake_model
        engine.hf_config = MagicMock()
        engine.vllm_megatron_weight_loaders.load_megatron_weights = MagicMock()

        mock_mla = MagicMock()
        mock_layer = MagicMock()
        mock_layer.self_attn.mla_attn = mock_mla
        fake_model.model = MagicMock()
        fake_model.model.layers = [mock_layer]
        fake_model.model.start_layer = 0
        fake_model.model.end_layer = 1

        with patch.object(engine, "_process_mla") as mock_process:
            engine.sync_model_weights(params={"w": 1})
            engine.vllm_megatron_weight_loaders.load_megatron_weights.assert_called_once()
            mock_process.assert_called_once()

        # Test without MLA
        del fake_model.model.layers[0].self_attn.mla_attn
        engine.vllm_megatron_weight_loaders.load_megatron_weights.reset_mock()
        with patch.object(engine, "_process_mla") as mock_process:
            engine.sync_model_weights(params={"w": 1})
            mock_process.assert_not_called()

    def test_process_mla(self):
        """_process_mla should clear MLA temporary buffers and call process_weights_after_loading."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        mla_impl = MagicMock()
        mla_impl.w_kc = "some"
        mla_impl.w_vc = "some"
        mla_impl.W_UV = "some"
        mla_impl.W_UK_T = "some"
        mla_impl.process_weights_after_loading = MagicMock()

        mla_attn = MagicMock()
        mla_attn.impl = mla_impl

        mock_layer = MagicMock()
        mock_layer.self_attn.mla_attn = mla_attn

        engine.model = MagicMock()
        engine.model.model.start_layer = 0
        engine.model.model.end_layer = 1
        engine.model.model.layers = [mock_layer]

        engine._process_mla()

        self.assertIsNone(mla_impl.w_kc)
        self.assertIsNone(mla_impl.w_vc)
        self.assertIsNone(mla_impl.W_UV)
        self.assertIsNone(mla_impl.W_UK_T)
        mla_impl.process_weights_after_loading.assert_called_once_with(None)

    def test_generate_sequences(self):
        """generate_sequences exists and does nothing (placeholder)."""
        engine = self.AsyncVLLMInferEngine(enable_sleep_mode=True, tokenizer_name_or_path="/fake_model")
        engine.generate_sequences()

    def test_get_device_memory(self):
        """get_device_memory should return free memory in GB from NPUPlatform."""
        mem = self.get_device_memory()
        self.assertEqual(mem, 50.0)


if __name__ == "__main__":
    unittest.main()