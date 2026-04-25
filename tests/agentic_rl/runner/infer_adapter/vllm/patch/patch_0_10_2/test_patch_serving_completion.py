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
import asyncio
import sys
import os
import importlib
import importlib.util


class TestPatchServingCompletion(unittest.TestCase):
    """Test patch_serving_completion.py module"""

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
        """Setup mock objects for vllm and vllm_ascend"""
        mock_vllm = MagicMock()
        mock_vllm_config = MagicMock()
        mock_vllm_entrypoints = MagicMock()
        mock_vllm_entrypoints_openai = MagicMock()
        mock_vllm_entrypoints_openai_protocol = MagicMock()
        mock_vllm_entrypoints_openai_serving_engine = MagicMock()
        mock_vllm_logger = MagicMock()
        mock_vllm_entrypoints_utils = MagicMock()
        mock_vllm_inputs = MagicMock()
        mock_vllm_inputs_data = MagicMock()
        mock_vllm_outputs = MagicMock()
        mock_vllm_sampling_params = MagicMock()
        mock_vllm_utils = MagicMock()
        mock_vllm_entrypoints_openai_serving_completion = MagicMock()
        mock_vllm_entrypoints_openai_serving_chat = MagicMock()
        mock_vllm_transformers_utils = MagicMock()
        mock_vllm_transformers_utils_tokenizer = MagicMock()
        mock_vllm_transformers_utils_tokenizers = MagicMock()
        mock_jinja2 = MagicMock()
        mock_fastapi = MagicMock()
        mock_typing_extensions = MagicMock()

        mock_vllm_ascend = MagicMock()
        mock_vllm_ascend_patch = MagicMock()
        mock_vllm_ascend_patch_platform = MagicMock()
        mock_vllm_ascend_patch_worker = MagicMock()

        mock_vllm_entrypoints_openai_serving_completion.OpenAIServingCompletion = MagicMock()
        mock_vllm_entrypoints_openai_serving_chat.OpenAIServingChat = MagicMock()
        mock_vllm_logger.init_logger.return_value = MagicMock()
        mock_vllm_utils.merge_async_iterators = MagicMock()
        mock_vllm_inputs_data.is_embeds_prompt = MagicMock(return_value=False)
        mock_vllm_inputs_data.is_tokens_prompt = MagicMock(return_value=True)
        mock_vllm_entrypoints_openai_serving_engine.is_text_tokens_prompt = MagicMock(return_value=True)

        mock_typing_extensions.assert_never = MagicMock()

        mock_vllm_sampling_params.SamplingParams = type('SamplingParams', (), {})
        mock_vllm_sampling_params.BeamSearchParams = type('BeamSearchParams', (), {})

        mock_vllm_entrypoints_utils.get_max_tokens = MagicMock(return_value=100)

        mock_jinja2.TemplateError = type('TemplateError', (Exception,), {})

        mock_vllm_transformers_utils_tokenizer.MistralTokenizer = type('MistralTokenizer', (), {})

        mock_vllm_transformers_utils_tokenizers.maybe_serialize_tool_calls = MagicMock()
        mock_vllm_transformers_utils_tokenizers.truncate_tool_call_ids = MagicMock()
        mock_vllm_transformers_utils_tokenizers.validate_request_params = MagicMock()

        cls.mock_vllm_sampling_params = mock_vllm_sampling_params
        cls.mock_vllm_utils = mock_vllm_utils
        cls.mock_vllm_entrypoints_utils = mock_vllm_entrypoints_utils

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm': mock_vllm,
            'vllm.config': mock_vllm_config,
            'vllm.entrypoints': mock_vllm_entrypoints,
            'vllm.entrypoints.openai': mock_vllm_entrypoints_openai,
            'vllm.entrypoints.openai.protocol': mock_vllm_entrypoints_openai_protocol,
            'vllm.entrypoints.openai.serving_engine': mock_vllm_entrypoints_openai_serving_engine,
            'vllm.logger': mock_vllm_logger,
            'vllm.entrypoints.utils': mock_vllm_entrypoints_utils,
            'vllm.inputs': mock_vllm_inputs,
            'vllm.inputs.data': mock_vllm_inputs_data,
            'vllm.outputs': mock_vllm_outputs,
            'vllm.sampling_params': mock_vllm_sampling_params,
            'vllm.utils': mock_vllm_utils,
            'vllm.entrypoints.openai.serving_completion': mock_vllm_entrypoints_openai_serving_completion,
            'vllm.entrypoints.openai.serving_chat': mock_vllm_entrypoints_openai_serving_chat,
            'vllm.transformers_utils': mock_vllm_transformers_utils,
            'vllm.transformers_utils.tokenizer': mock_vllm_transformers_utils_tokenizer,
            'vllm.transformers_utils.tokenizers': mock_vllm_transformers_utils_tokenizers,
            'jinja2': mock_jinja2,
            'fastapi': mock_fastapi,
            'typing_extensions': mock_typing_extensions,
            'vllm_ascend': mock_vllm_ascend,
            'vllm_ascend.patch': mock_vllm_ascend_patch,
            'vllm_ascend.patch.platform': mock_vllm_ascend_patch_platform,
            'vllm_ascend.patch.worker': mock_vllm_ascend_patch_worker,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_serving_completion',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_serving_completion.py')
        )
        cls.patch_serving_completion = importlib.util.module_from_spec(spec)
        sys.modules['patch_serving_completion'] = cls.patch_serving_completion
        spec.loader.exec_module(cls.patch_serving_completion)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_completion_request = MagicMock()
        self.mock_completion_request.suffix = None
        self.mock_completion_request.echo = False
        self.mock_completion_request.prompt_embeds = None
        self.mock_completion_request.request_id = "test-req-123"
        self.mock_completion_request.stream = False
        self.mock_completion_request.use_beam_search = False
        self.mock_completion_request.best_of = None
        self.mock_completion_request.n = 1
        self.mock_completion_request.add_special_tokens = True
        self.mock_completion_request.priority = 0

        self.mock_sampling_params = MagicMock()
        self.mock_completion_request.to_sampling_params.return_value = self.mock_sampling_params
        self.mock_completion_request.to_beam_search_params.return_value = self.mock_sampling_params

        self.mock_chat_request = MagicMock()
        self.mock_chat_request.model = "test-model"
        self.mock_chat_request.request_id = "test-chat-req-123"
        self.mock_chat_request.stream = False
        self.mock_chat_request.use_beam_search = False
        self.mock_chat_request.tool_choice = "none"
        self.mock_chat_request.tools = None
        self.mock_chat_request.add_generation_prompt = True
        self.mock_chat_request.continue_final_message = False
        self.mock_chat_request.documents = None
        self.mock_chat_request.chat_template = None
        self.mock_chat_request.chat_template_content_format = None
        self.mock_chat_request.chat_template_kwargs = None
        self.mock_chat_request.add_special_tokens = True
        self.mock_chat_request.priority = 0

        self.mock_chat_request.to_sampling_params.return_value = self.mock_sampling_params
        self.mock_chat_request.to_beam_search_params.return_value = self.mock_sampling_params

        self.mock_raw_request = MagicMock()
        self.mock_raw_request.headers = {}
        self.mock_raw_request.state = MagicMock()

        self.mock_engine_client = MagicMock()
        self.mock_engine_client.errored = False

        self.mock_tokenizer = MagicMock()

        self.mock_completion_response = MagicMock()
        self.mock_chat_response = MagicMock()

        self.mock_request_output = MagicMock()
        self.mock_request_output.prompt = None
        self.mock_request_output.prompt_token_ids = [1, 2, 3]
        self.mock_request_output.outputs = [MagicMock()]

        self.mock_model_config = MagicMock()
        self.mock_model_config.skip_tokenizer_init = False
        self.mock_model_config.logits_processor_pattern = None

        self.mock_serving_completion = MagicMock()
        self.mock_serving_completion.engine_client = self.mock_engine_client
        self.mock_serving_completion.model_config = self.mock_model_config
        self.mock_serving_completion.default_sampling_params = {}
        self.mock_serving_completion.max_model_len = 2048
        self.mock_serving_completion.enable_force_include_usage = False
        self.mock_serving_completion.tool_parser = None
        self.mock_serving_completion.enable_auto_tools = False
        self.mock_serving_completion.exclude_tools_when_tool_choice_none = True
        self.mock_serving_completion.use_harmony = False
        self.mock_serving_completion.chat_template = None
        self.mock_serving_completion.chat_template_content_format = None

        self.mock_serving_chat = MagicMock()
        self.mock_serving_chat.engine_client = self.mock_engine_client
        self.mock_serving_chat.model_config = self.mock_model_config
        self.mock_serving_chat.default_sampling_params = {}
        self.mock_serving_chat.max_model_len = 2048
        self.mock_serving_chat.enable_force_include_usage = False
        self.mock_serving_chat.tool_parser = None
        self.mock_serving_chat.enable_auto_tools = False
        self.mock_serving_chat.exclude_tools_when_tool_choice_none = True
        self.mock_serving_chat.use_harmony = False
        self.mock_serving_chat.chat_template = None
        self.mock_serving_chat.chat_template_content_format = None

        self.mock_vllm_utils.merge_async_iterators.reset_mock()
        self.mock_vllm_entrypoints_utils.get_max_tokens.reset_mock()

    def test_create_completion_patch_with_error_response(self):
        """Test create_completion_patch with error response from _check_model"""
        mock_error_response = MagicMock()

        async def mock_check_model(*args, **kwargs):
            return mock_error_response

        self.mock_serving_completion._check_model = mock_check_model

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
            )

            self.assertEqual(result, mock_error_response)
        finally:
            loop.close()

    def test_create_completion_patch_with_suffix(self):
        """Test create_completion_patch with suffix parameter (should return error)"""
        self.mock_completion_request.suffix = "test suffix"

        async def mock_check_model(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion.create_error_response = MagicMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
            )

            self.mock_serving_completion.create_error_response.assert_called_once_with(
                "suffix is not currently supported"
            )
        finally:
            loop.close()

    def test_create_completion_patch_with_echo_and_embeds(self):
        """Test create_completion_patch with echo and prompt_embeds (should return error)"""
        self.mock_completion_request.echo = True
        self.mock_completion_request.prompt_embeds = [1.0, 2.0, 3.0]

        async def mock_check_model(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion.create_error_response = MagicMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
            )

            self.mock_serving_completion.create_error_response.assert_called_once_with(
                "Echo is unsupported with prompt embeds."
            )
        finally:
            loop.close()

    def test_create_completion_patch_with_engine_error(self):
        """Test create_completion_patch when engine_client is in error state"""
        self.mock_engine_client.errored = True
        self.mock_engine_client.dead_error = RuntimeError("Engine died")

        async def mock_check_model(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(RuntimeError) as context:
                loop.run_until_complete(
                    self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
                )

            self.assertEqual(context.exception, self.mock_engine_client.dead_error)
        finally:
            loop.close()

    def test_create_completion_patch_skip_tokenizer_init(self):
        """Test create_completion_patch with skip_tokenizer_init=True"""
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        self.mock_model_config.skip_tokenizer_init = True

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers

        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)

        self.mock_engine_client.generate.side_effect = mock_generate

        self.mock_vllm_utils.merge_async_iterators.side_effect = lambda *generators: generators[0] if generators else MagicMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
            )

            self.mock_engine_client.get_tokenizer.assert_not_called()
        finally:
            loop.close()

    def test_create_completion_patch_with_streaming(self):
        """Test create_completion_patch with streaming=True"""
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        self.mock_completion_request.stream = True
        self.mock_completion_request.best_of = None

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        async def mock_completion_stream_generator(*args, **kwargs):
            yield "data: {}\n\n"

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers
        self.mock_serving_completion.completion_stream_generator = mock_completion_stream_generator

        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)

        self.mock_engine_client.generate.side_effect = mock_generate

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
            )

            self.assertTrue(asyncio.iscoroutinefunction(result.__anext__))
        finally:
            loop.close()

    def test_create_completion_patch_with_preprocessing_error(self):
        """Test create_completion_patch with ValueError during preprocessing"""
        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_preprocess_completion(*args, **kwargs):
            raise ValueError("Invalid prompt")

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion.create_error_response = MagicMock(return_value="Error response")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
            )

            self.mock_serving_completion.create_error_response.assert_called_once()
        finally:
            loop.close()

    def test_create_completion_patch_with_value_error(self):
        """Test create_completion_patch with ValueError during generator creation"""
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers
        self.mock_serving_completion.create_error_response = MagicMock(return_value="Error response")

        mock_sampling_params = MagicMock()
        self.mock_completion_request.to_sampling_params.return_value = mock_sampling_params

        self.mock_serving_completion._log_inputs.side_effect = ValueError("Test error")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.patch_serving_completion.create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
            )

            self.mock_serving_completion.create_error_response.assert_called_once()
        finally:
            loop.close()

    def test_create_completion_patch_without_dp_rank(self):
        """Test create_completion_patch without X-Dp-Rank header"""
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_get_tokenizer(*args, **kwargs):
            return self.mock_tokenizer

        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_engine_client.get_tokenizer = mock_get_tokenizer
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers

        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)

        self.mock_engine_client.generate.side_effect = mock_generate

        self.mock_vllm_utils.merge_async_iterators.side_effect = lambda *args: args[0]

        self.mock_serving_completion._get_model_name.return_value = "test-model"
        self.mock_serving_completion.request_output_to_completion_response.return_value = self.mock_completion_response

        result = asyncio.run(self.patch_serving_completion.create_completion_patch(
            self.mock_serving_completion,
            self.mock_completion_request,
            self.mock_raw_request
        ))

        self.assertEqual(result, self.mock_completion_response)

        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertIn('data_parallel_rank', call_args.kwargs)
        self.assertIsNone(call_args.kwargs['data_parallel_rank'])

    def test_create_completion_patch_with_dp_rank(self):
        """Test create_completion_patch with X-Dp-Rank header"""
        self.mock_raw_request.headers["X-Dp-Rank"] = "2"

        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_get_tokenizer(*args, **kwargs):
            return self.mock_tokenizer

        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_engine_client.get_tokenizer = mock_get_tokenizer
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers

        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)

        self.mock_engine_client.generate.side_effect = mock_generate

        self.mock_vllm_utils.merge_async_iterators.side_effect = lambda *args: args[0]

        self.mock_serving_completion._get_model_name.return_value = "test-model"
        self.mock_serving_completion.request_output_to_completion_response.return_value = self.mock_completion_response

        result = asyncio.run(self.patch_serving_completion.create_completion_patch(
            self.mock_serving_completion,
            self.mock_completion_request,
            self.mock_raw_request
        ))

        self.assertEqual(result, self.mock_completion_response)

        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertEqual(call_args.kwargs['data_parallel_rank'], 2)

    def test_create_chat_completion_patch_without_dp_rank(self):
        """Test create_chat_completion_patch without X-Dp-Rank header"""
        mock_conversation = MagicMock()
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_get_tokenizer(*args, **kwargs):
            return self.mock_tokenizer

        async def mock_preprocess_chat(*args, **kwargs):
            return (mock_conversation, [mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_chat._check_model = mock_check_model
        self.mock_serving_chat._base_request_id.return_value = "12345"
        self.mock_engine_client.get_tokenizer = mock_get_tokenizer
        self.mock_serving_chat._preprocess_chat = mock_preprocess_chat
        self.mock_serving_chat._maybe_get_adapters.return_value = None
        self.mock_serving_chat._log_inputs = MagicMock()
        self.mock_serving_chat._get_trace_headers = mock_get_trace_headers

        async def mock_generate(*args, **kwargs):
            yield self.mock_request_output

        self.mock_engine_client.generate.side_effect = mock_generate

        self.mock_serving_chat._get_model_name.return_value = "test-model"

        async def mock_chat_completion_full_generator(*args, **kwargs):
            return self.mock_chat_response

        self.mock_serving_chat.chat_completion_full_generator = mock_chat_completion_full_generator

        result = asyncio.run(self.patch_serving_completion.create_chat_completion_patch(
            self.mock_serving_chat,
            self.mock_chat_request,
            self.mock_raw_request
        ))

        self.assertEqual(result, self.mock_chat_response)

        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertIn('data_parallel_rank', call_args.kwargs)
        self.assertIsNone(call_args.kwargs['data_parallel_rank'])

    def test_create_chat_completion_patch_with_dp_rank(self):
        """Test create_chat_completion_patch with X-Dp-Rank header"""
        self.mock_raw_request.headers["X-Dp-Rank"] = "1"

        mock_conversation = MagicMock()
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}

        async def mock_check_model(*args, **kwargs):
            return None

        async def mock_get_tokenizer(*args, **kwargs):
            return self.mock_tokenizer

        async def mock_preprocess_chat(*args, **kwargs):
            return (mock_conversation, [mock_request_prompt], [mock_engine_prompt])

        async def mock_get_trace_headers(*args, **kwargs):
            return None

        self.mock_serving_chat._check_model = mock_check_model
        self.mock_serving_chat._base_request_id.return_value = "12345"
        self.mock_engine_client.get_tokenizer = mock_get_tokenizer
        self.mock_serving_chat._preprocess_chat = mock_preprocess_chat
        self.mock_serving_chat._maybe_get_adapters.return_value = None
        self.mock_serving_chat._log_inputs = MagicMock()
        self.mock_serving_chat._get_trace_headers = mock_get_trace_headers

        async def mock_generate(*args, **kwargs):
            yield self.mock_request_output

        self.mock_engine_client.generate.side_effect = mock_generate

        self.mock_serving_chat._get_model_name.return_value = "test-model"

        async def mock_chat_completion_full_generator(*args, **kwargs):
            return self.mock_chat_response

        self.mock_serving_chat.chat_completion_full_generator = mock_chat_completion_full_generator

        result = asyncio.run(self.patch_serving_completion.create_chat_completion_patch(
            self.mock_serving_chat,
            self.mock_chat_request,
            self.mock_raw_request
        ))

        self.assertEqual(result, self.mock_chat_response)

        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertEqual(call_args.kwargs['data_parallel_rank'], 1)


if __name__ == '__main__':
    unittest.main()
