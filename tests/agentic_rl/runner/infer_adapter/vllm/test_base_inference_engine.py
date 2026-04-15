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

import unittest
from agentic_rl.runner.infer_adapter.vllm.base_inference_engine import BaseInferEngine


# -----------------------------
# Helper: concrete BaseInferEngine implementation
# -----------------------------
class DummyInferEngine(BaseInferEngine):
    def init_cache_engine(self):
        return "cache_inited"

    def free_cache_engine(self):
        return "cache_freed"

    def offload_model_weights(self):
        return "weights_offloaded"

    def sync_model_weights(self, params, load_format="megatron"):
        # record args for UT verification
        self._last_sync_params = params
        self._last_sync_format = load_format
        return "weights_synced"

    def generate_sequences(
        self,
        prompts=None,
        sampling_params=None,
        prompt_token_ids=None,
        use_tqdm=None,
        **kwargs
    ):
        # record args for UT verification
        self._last_prompts = prompts
        self._last_sampling_params = sampling_params
        self._last_prompt_token_ids = prompt_token_ids
        self._last_use_tqdm = use_tqdm
        self._last_kwargs = kwargs

        return {"sequences": ["test_output"], "num_prompts": 0 if prompts is None else len(prompts)}


# -----------------------------
# Helper: incomplete implementation (should fail instantiation)
# -----------------------------
class IncompleteInferEngine(BaseInferEngine):
    def init_cache_engine(self):
        return None
    # free_cache_engine missing

    def offload_model_weights(self):
        return None

    def sync_model_weights(self, params, load_format="megatron"):
        return None

    def generate_sequences(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=None, **kwargs):
        return None


# -----------------------------
# BaseInferEngine UT
# -----------------------------
class TestBaseInferEngine(unittest.TestCase):

    def test_init_with_defaults(self):
        """Test BaseInferEngine initialization with default parameters."""
        engine = DummyInferEngine(
            tokenizer_name_or_path="test_tokenizer",
            train_tensor_parallel_size=1,
            train_pipeline_parallel_size=1,
        )

        # required args
        self.assertEqual(engine.tokenizer_name_or_path, "test_tokenizer")
        self.assertEqual(engine.train_tensor_parallel_size, 1)
        self.assertEqual(engine.train_pipeline_parallel_size, 1)

        # optional defaults
        self.assertIsNone(engine.prompt_type)
        self.assertIsNone(engine.prompt_type_path)
        self.assertEqual(engine.train_expert_parallel_size, 1)
        self.assertEqual(engine.train_context_parallel_size, 1)

        self.assertEqual(engine.infer_tensor_parallel_size, 8)
        self.assertEqual(engine.infer_pipeline_parallel_size, 1)
        self.assertEqual(engine.infer_expert_parallel_size, 1)

        self.assertEqual(engine.max_num_seqs, 1)
        self.assertEqual(engine.max_model_len, 2048)
        self.assertEqual(engine.dtype, "bfloat16")
        self.assertEqual(engine.gpu_memory_utilization, 0.5)
        self.assertTrue(engine.trust_remote_code)
        self.assertEqual(engine.infer_backend, "vllm")

    def test_init_with_custom_params(self):
        """Test BaseInferEngine initialization with custom parameters."""
        engine = DummyInferEngine(
            tokenizer_name_or_path="custom_tokenizer",
            train_tensor_parallel_size=2,
            train_pipeline_parallel_size=4,
            prompt_type="chat",
            prompt_type_path="/path/to/prompt",
            train_expert_parallel_size=2,
            train_context_parallel_size=2,
            infer_tensor_parallel_size=4,
            infer_pipeline_parallel_size=2,
            infer_expert_parallel_size=2,
            max_num_seqs=10,
            max_model_len=4096,
            dtype="float16",
            gpu_memory_utilization=0.8,
            trust_remote_code=False,
            infer_backend="custom_backend",
        )

        self.assertEqual(engine.tokenizer_name_or_path, "custom_tokenizer")
        self.assertEqual(engine.prompt_type, "chat")
        self.assertEqual(engine.prompt_type_path, "/path/to/prompt")

        self.assertEqual(engine.train_tensor_parallel_size, 2)
        self.assertEqual(engine.train_pipeline_parallel_size, 4)
        self.assertEqual(engine.train_expert_parallel_size, 2)
        self.assertEqual(engine.train_context_parallel_size, 2)

        self.assertEqual(engine.infer_tensor_parallel_size, 4)
        self.assertEqual(engine.infer_pipeline_parallel_size, 2)
        self.assertEqual(engine.infer_expert_parallel_size, 2)

        self.assertEqual(engine.max_num_seqs, 10)
        self.assertEqual(engine.max_model_len, 4096)
        self.assertEqual(engine.dtype, "float16")
        self.assertEqual(engine.gpu_memory_utilization, 0.8)
        self.assertFalse(engine.trust_remote_code)
        self.assertEqual(engine.infer_backend, "custom_backend")

    def test_init_required_args_missing(self):
        """Test that missing required args raises TypeError."""
        with self.assertRaises(TypeError):
            DummyInferEngine(
                tokenizer_name_or_path="test_tokenizer",
                train_tensor_parallel_size=1,
                # train_pipeline_parallel_size missing
            )

        with self.assertRaises(TypeError):
            DummyInferEngine(
                # tokenizer_name_or_path missing
                train_tensor_parallel_size=1,
                train_pipeline_parallel_size=1,
            )

    def test_base_class_cannot_be_instantiated(self):
        """Test that BaseInferEngine cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseInferEngine(
                tokenizer_name_or_path="test",
                train_tensor_parallel_size=1,
                train_pipeline_parallel_size=1,
            )

    def test_incomplete_subclass_cannot_be_instantiated(self):
        """Test that subclass missing abstract methods cannot be instantiated."""
        with self.assertRaises(TypeError):
            IncompleteInferEngine(
                tokenizer_name_or_path="test",
                train_tensor_parallel_size=1,
                train_pipeline_parallel_size=1,
            )

    def test_implemented_methods_basic_behavior(self):
        """Test that DummyInferEngine methods work and record args correctly."""
        engine = DummyInferEngine(
            tokenizer_name_or_path="tok",
            train_tensor_parallel_size=1,
            train_pipeline_parallel_size=1,
        )

        self.assertEqual(engine.init_cache_engine(), "cache_inited")
        self.assertEqual(engine.free_cache_engine(), "cache_freed")
        self.assertEqual(engine.offload_model_weights(), "weights_offloaded")

        ret = engine.sync_model_weights({"a": 1})
        self.assertEqual(ret, "weights_synced")
        self.assertEqual(engine._last_sync_params, {"a": 1})
        self.assertEqual(engine._last_sync_format, "megatron")

        ret = engine.sync_model_weights({"b": 2}, load_format="hf")
        self.assertEqual(ret, "weights_synced")
        self.assertEqual(engine._last_sync_params, {"b": 2})
        self.assertEqual(engine._last_sync_format, "hf")

    def test_generate_sequences_arg_recording(self):
        """Test generate_sequences records args and supports kwargs."""
        engine = DummyInferEngine(
            tokenizer_name_or_path="tok",
            train_tensor_parallel_size=1,
            train_pipeline_parallel_size=1,
        )

        result = engine.generate_sequences(
            prompts=["p1", "p2"],
            sampling_params={"temp": 0.7},
            prompt_token_ids=[[1, 2], [3, 4]],
            use_tqdm=False,
            custom_flag=True,
            top_p=0.9,
        )

        self.assertEqual(result["sequences"], ["test_output"])
        self.assertEqual(result["num_prompts"], 2)

        self.assertEqual(engine._last_prompts, ["p1", "p2"])
        self.assertEqual(engine._last_sampling_params, {"temp": 0.7})
        self.assertEqual(engine._last_prompt_token_ids, [[1, 2], [3, 4]])
        self.assertFalse(engine._last_use_tqdm)
        self.assertTrue(engine._last_kwargs["custom_flag"])
        self.assertEqual(engine._last_kwargs["top_p"], 0.9)

    def test_multiple_init_variants_subtest(self):
        """Test multiple init variants with subTest to increase coverage clarity."""
        test_cases = [
            {
                "name": "default_infer",
                "kwargs": dict(
                    tokenizer_name_or_path="tok1",
                    train_tensor_parallel_size=1,
                    train_pipeline_parallel_size=1,
                ),
                "expected_backend": "vllm",
                "expected_dtype": "bfloat16",
            },
            {
                "name": "custom_infer",
                "kwargs": dict(
                    tokenizer_name_or_path="tok2",
                    train_tensor_parallel_size=2,
                    train_pipeline_parallel_size=2,
                    infer_backend="custom",
                    dtype="float16",
                    max_model_len=1024,
                ),
                "expected_backend": "custom",
                "expected_dtype": "float16",
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                engine = DummyInferEngine(**case["kwargs"])
                self.assertEqual(engine.infer_backend, case["expected_backend"])
                self.assertEqual(engine.dtype, case["expected_dtype"])


if __name__ == "__main__":
    unittest.main()
