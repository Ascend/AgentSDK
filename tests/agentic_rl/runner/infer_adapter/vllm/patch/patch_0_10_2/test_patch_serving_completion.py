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


class TestPatchServingCompletion(unittest.TestCase):
    """Test patch_serving_completion.py module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock request objects
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
        
        # Mock sampling params methods
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
        
        # Mock sampling params methods for chat request
        self.mock_chat_request.to_sampling_params.return_value = self.mock_sampling_params
        self.mock_chat_request.to_beam_search_params.return_value = self.mock_sampling_params
        
        # Create mock raw request with headers
        self.mock_raw_request = MagicMock()
        self.mock_raw_request.headers = {}
        self.mock_raw_request.state = MagicMock()
        
        # Create mock engine client
        self.mock_engine_client = MagicMock()
        self.mock_engine_client.errored = False
        
        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        
        # Create mock response objects
        self.mock_completion_response = MagicMock()
        self.mock_chat_response = MagicMock()
        
        # Create mock request output
        self.mock_request_output = MagicMock()
        self.mock_request_output.prompt = None
        self.mock_request_output.prompt_token_ids = [1, 2, 3]
        self.mock_request_output.outputs = [MagicMock()]
        
        # Create mock model config
        self.mock_model_config = MagicMock()
        self.mock_model_config.skip_tokenizer_init = False
        self.mock_model_config.logits_processor_pattern = None
        
        # Create mock instances for the serving classes
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
        
        # Create a minimal mock hierarchy for the modules we need to import
        self.mock_modules = {
            'vllm': MagicMock(),
            'vllm.entrypoints': MagicMock(),
            'vllm.entrypoints.openai': MagicMock(),
            'vllm.entrypoints.openai.protocol': MagicMock(),
            'vllm.entrypoints.openai.serving_engine': MagicMock(),
            'vllm.logger': MagicMock(),
            'vllm.entrypoints.utils': MagicMock(),
            'vllm.inputs': MagicMock(),
            'vllm.inputs.data': MagicMock(),
            'vllm.outputs': MagicMock(),
            'vllm.sampling_params': MagicMock(),
            'vllm.utils': MagicMock(),
            'vllm.entrypoints.openai.serving_completion': MagicMock(),
            'vllm.entrypoints.openai.serving_chat': MagicMock(),
            'vllm.transformers_utils': MagicMock(),
            'vllm.transformers_utils.tokenizer': MagicMock(),
            'vllm.transformers_utils.tokenizers': MagicMock(),
            'jinja2': MagicMock(),
            'fastapi': MagicMock(),
            'typing_extensions': MagicMock(),
        }
        
        # Set up specific mocks for the modules
        self.mock_modules['vllm.entrypoints.openai.serving_completion'].OpenAIServingCompletion = MagicMock()
        self.mock_modules['vllm.entrypoints.openai.serving_chat'].OpenAIServingChat = MagicMock()
        self.mock_modules['vllm.logger'].init_logger.return_value = MagicMock()
        self.mock_modules['vllm.utils'].merge_async_iterators = MagicMock()
        self.mock_modules['vllm.inputs.data'].is_embeds_prompt = MagicMock(return_value=False)
        self.mock_modules['vllm.inputs.data'].is_tokens_prompt = MagicMock(return_value=True)
        self.mock_modules['vllm.entrypoints.openai.serving_engine'].is_text_tokens_prompt = MagicMock(return_value=True)
        
        # Set up mock for assert_never
        self.mock_modules['typing_extensions'].assert_never = MagicMock()
        
        # Set up proper type mocks for sampling_params
        self.mock_modules['vllm.sampling_params'].SamplingParams = type('SamplingParams', (), {})
        self.mock_modules['vllm.sampling_params'].BeamSearchParams = type('BeamSearchParams', (), {})
        
        # Mock the get_max_tokens function
        self.mock_modules['vllm.entrypoints.utils'].get_max_tokens = MagicMock(return_value=100)
        
        # Mock jinja2 TemplateError as an exception class
        self.mock_modules['jinja2'].TemplateError = type('TemplateError', (Exception,), {})
        
        # Mock MistralTokenizer for the chat completion test
        self.mock_modules['vllm.transformers_utils.tokenizer'].MistralTokenizer = type('MistralTokenizer', (), {})
        
        # Mock functions from tokenizers module
        self.mock_modules['vllm.transformers_utils.tokenizers'].maybe_serialize_tool_calls = MagicMock()
        self.mock_modules['vllm.transformers_utils.tokenizers'].truncate_tool_call_ids = MagicMock()
        self.mock_modules['vllm.transformers_utils.tokenizers'].validate_request_params = MagicMock()
        
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    def test_create_completion_patch_with_error_response(self):
        """Test create_completion_patch with error response from _check_model"""
        # Set up test scenario where _check_model returns an error
        mock_error_response = MagicMock()
        
        async def mock_check_model(*args, **kwargs):
            return mock_error_response
        
        self.mock_serving_completion._check_model = mock_check_model
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
                )
                
                # Verify the error response was returned
                self.assertEqual(result, mock_error_response)
            finally:
                loop.close()
    
    def test_create_completion_patch_with_suffix(self):
        """Test create_completion_patch with suffix parameter (should return error)"""
        # Set up test scenario with suffix
        self.mock_completion_request.suffix = "test suffix"
        
        async def mock_check_model(*args, **kwargs):
            return None
        
        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion.create_error_response = MagicMock()
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
                )
                
                # Verify create_error_response was called
                self.mock_serving_completion.create_error_response.assert_called_once_with(
                    "suffix is not currently supported"
                )
            finally:
                loop.close()
    
    def test_create_completion_patch_with_echo_and_embeds(self):
        """Test create_completion_patch with echo and prompt_embeds (should return error)"""
        # Set up test scenario with echo and prompt_embeds
        self.mock_completion_request.echo = True
        self.mock_completion_request.prompt_embeds = [1.0, 2.0, 3.0]
        
        async def mock_check_model(*args, **kwargs):
            return None
        
        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion.create_error_response = MagicMock()
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
                )
                
                # Verify create_error_response was called
                self.mock_serving_completion.create_error_response.assert_called_once_with(
                    "Echo is unsupported with prompt embeds."
                )
            finally:
                loop.close()
    
    def test_create_completion_patch_with_engine_error(self):
        """Test create_completion_patch when engine_client is in error state"""
        # Set up test scenario with engine error
        self.mock_engine_client.errored = True
        self.mock_engine_client.dead_error = RuntimeError("Engine died")
        
        async def mock_check_model(*args, **kwargs):
            return None
        
        self.mock_serving_completion._check_model = mock_check_model
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with self.assertRaises(RuntimeError) as context:
                    loop.run_until_complete(
                        create_completion_patch(self.mock_serving_completion, self.mock_completion_request)
                    )
                
                # Verify the exception is the engine's dead_error
                self.assertEqual(context.exception, self.mock_engine_client.dead_error)
            finally:
                loop.close()
    
    def test_create_completion_patch_skip_tokenizer_init(self):
        """Test create_completion_patch with skip_tokenizer_init=True"""
        # Set up test scenario
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Set skip_tokenizer_init to True
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
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            # The code creates a result generator for each prompt
            # Since we have one prompt in our test, we yield one result
            # at index 0
            yield (0, self.mock_request_output)
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Set up a proper mock for merge_async_iterators that returns the same generator
        self.mock_modules['vllm.utils'].merge_async_iterators.side_effect = lambda *generators: generators[0] if generators else MagicMock()
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
                )
                
                # Verify get_tokenizer was not called
                self.mock_engine_client.get_tokenizer.assert_not_called()
            finally:
                loop.close()
    
    def test_create_completion_patch_with_streaming(self):
        """Test create_completion_patch with streaming=True"""
        # Set up test scenario
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Enable streaming
        self.mock_completion_request.stream = True
        self.mock_completion_request.best_of = None
        
        async def mock_check_model(*args, **kwargs):
            return None
        
        async def mock_preprocess_completion(*args, **kwargs):
            return ([mock_request_prompt], [mock_engine_prompt])
        
        async def mock_get_trace_headers(*args, **kwargs):
            return None
        
        # Mock completion_stream_generator to return a mock async generator
        async def mock_completion_stream_generator(*args, **kwargs):
            yield "data: {}\n\n"
        
        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._base_request_id.return_value = "12345"
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion._maybe_get_adapters.return_value = None
        self.mock_serving_completion._log_inputs = MagicMock()
        self.mock_serving_completion._get_trace_headers = mock_get_trace_headers
        self.mock_serving_completion.completion_stream_generator = mock_completion_stream_generator
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
                )
                
                # Verify we got an async generator for streaming
                self.assertTrue(asyncio.iscoroutinefunction(result.__anext__))
            finally:
                loop.close()
    
    def test_create_completion_patch_with_preprocessing_error(self):
        """Test create_completion_patch with ValueError during preprocessing"""
        # Set up test scenario
        async def mock_check_model(*args, **kwargs):
            return None
        
        # Mock preprocess_completion to raise ValueError
        async def mock_preprocess_completion(*args, **kwargs):
            raise ValueError("Invalid prompt")
        
        self.mock_serving_completion._check_model = mock_check_model
        self.mock_serving_completion._preprocess_completion = mock_preprocess_completion
        self.mock_serving_completion.create_error_response = MagicMock(return_value="Error response")
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
                )
                
                # Verify create_error_response was called with the correct message
                self.mock_serving_completion.create_error_response.assert_called_once()
            finally:
                loop.close()
    
    def test_create_completion_patch_with_value_error(self):
        """Test create_completion_patch with ValueError during generator creation"""
        # Set up test scenario
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
        
        # Mock sampling params
        mock_sampling_params = MagicMock()
        self.mock_completion_request.to_sampling_params.return_value = mock_sampling_params
        
        # Mock _log_inputs to raise ValueError
        self.mock_serving_completion._log_inputs.side_effect = ValueError("Test error")
        
        # Import the create_completion_patch function with mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2 import patch_serving_completion
            
            # Mock the directly imported functions
            patch_serving_completion.is_embeds_prompt = lambda prompt: "prompt_embeds" in prompt
            patch_serving_completion.is_tokens_prompt = lambda prompt: "prompt_token_ids" in prompt
            patch_serving_completion.get_max_tokens = lambda **kwargs: 100
            
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Run the test with explicit event loop creation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    create_completion_patch(self.mock_serving_completion, self.mock_completion_request, self.mock_raw_request)
                )
                
                # Verify create_error_response was called
                self.mock_serving_completion.create_error_response.assert_called_once()
            finally:
                loop.close()
    
    def test_create_completion_patch_without_dp_rank(self):
        """Test create_completion_patch without X-Dp-Rank header"""
        # Set up test scenario
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Mock the necessary methods (note: async methods need to return awaitable objects)
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
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Mock merge_async_iterators to return the same generator
        self.mock_modules['vllm.utils'].merge_async_iterators.side_effect = lambda *args: args[0]
        
        # Mock other necessary methods
        self.mock_serving_completion._get_model_name.return_value = "test-model"
        self.mock_serving_completion.request_output_to_completion_response.return_value = self.mock_completion_response
        
        # Import the patch function inside the test with module mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Call the patch function using asyncio.run
            result = asyncio.run(create_completion_patch(
                self.mock_serving_completion,
                self.mock_completion_request,
                self.mock_raw_request
            ))
        
        # Verify the function behavior
        self.assertEqual(result, self.mock_completion_response)
        
        # Verify that generate was called with data_parallel_rank=None (no header)
        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertIn('data_parallel_rank', call_args.kwargs)
        self.assertIsNone(call_args.kwargs['data_parallel_rank'])
        
    def test_create_completion_patch_with_dp_rank(self):
        """Test create_completion_patch with X-Dp-Rank header"""
        # Set up test scenario with X-Dp-Rank header
        self.mock_raw_request.headers["X-Dp-Rank"] = "2"
        
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Mock the necessary methods (note: async methods need to return awaitable objects)
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
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            yield (0, self.mock_request_output)
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Mock merge_async_iterators to return the same generator
        self.mock_modules['vllm.utils'].merge_async_iterators.side_effect = lambda *args: args[0]
        
        # Mock other necessary methods
        self.mock_serving_completion._get_model_name.return_value = "test-model"
        self.mock_serving_completion.request_output_to_completion_response.return_value = self.mock_completion_response
        
        # Import the patch function inside the test with module mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_completion_patch
            
            # Call the patch function using asyncio.run
            result = asyncio.run(create_completion_patch(
                self.mock_serving_completion,
                self.mock_completion_request,
                self.mock_raw_request
            ))
        
        # Verify the function behavior
        self.assertEqual(result, self.mock_completion_response)
        
        # Verify that generate was called with the correct data_parallel_rank
        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertEqual(call_args.kwargs['data_parallel_rank'], 2)
    
    def test_create_chat_completion_patch_without_dp_rank(self):
        """Test create_chat_completion_patch without X-Dp-Rank header"""
        # Set up test scenario
        mock_conversation = MagicMock()
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Mock the necessary methods (note: async methods need to return awaitable objects)
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
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            yield self.mock_request_output
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Mock other necessary methods
        self.mock_serving_chat._get_model_name.return_value = "test-model"
        
        # Mock chat_completion_full_generator as an async method
        async def mock_chat_completion_full_generator(*args, **kwargs):
            return self.mock_chat_response
        
        self.mock_serving_chat.chat_completion_full_generator = mock_chat_completion_full_generator
        
        # Import the patch function inside the test with module mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_chat_completion_patch
            
            # Call the patch function using asyncio.run
            result = asyncio.run(create_chat_completion_patch(
                self.mock_serving_chat,
                self.mock_chat_request,
                self.mock_raw_request
            ))
        
        # Verify the function behavior
        self.assertEqual(result, self.mock_chat_response)
        
        # Verify that generate was called with data_parallel_rank=None (no header)
        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertIn('data_parallel_rank', call_args.kwargs)
        self.assertIsNone(call_args.kwargs['data_parallel_rank'])
    
    def test_create_chat_completion_patch_with_dp_rank(self):
        """Test create_chat_completion_patch with X-Dp-Rank header"""
        # Set up test scenario with X-Dp-Rank header
        self.mock_raw_request.headers["X-Dp-Rank"] = "1"
        
        mock_conversation = MagicMock()
        mock_request_prompt = {"prompt": "Hello, world!"}
        mock_engine_prompt = {"prompt_token_ids": [1, 2, 3]}
        
        # Mock the necessary methods (note: async methods need to return awaitable objects)
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
        
        # Mock the generate method to return a mock async generator
        async def mock_generate(*args, **kwargs):
            yield self.mock_request_output
        
        self.mock_engine_client.generate.side_effect = mock_generate
        
        # Mock other necessary methods
        self.mock_serving_chat._get_model_name.return_value = "test-model"
        
        # Mock chat_completion_full_generator as an async method
        async def mock_chat_completion_full_generator(*args, **kwargs):
            return self.mock_chat_response
        
        self.mock_serving_chat.chat_completion_full_generator = mock_chat_completion_full_generator
        
        # Import the patch function inside the test with module mocking
        with patch.dict('sys.modules', self.mock_modules):
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import create_chat_completion_patch
            
            # Call the patch function using asyncio.run
            result = asyncio.run(create_chat_completion_patch(
                self.mock_serving_chat,
                self.mock_chat_request,
                self.mock_raw_request
            ))
        
        # Verify the function behavior
        self.assertEqual(result, self.mock_chat_response)
        
        # Verify that generate was called with the correct data_parallel_rank
        self.mock_engine_client.generate.assert_called_once()
        call_args = self.mock_engine_client.generate.call_args
        self.assertEqual(call_args.kwargs['data_parallel_rank'], 1)
    
    def test_patch_applied(self):
        """Test that the patches are correctly applied"""
        # First, import the original classes to set up our mocks
        with patch.dict('sys.modules', self.mock_modules):
            from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
            
            # Save the original methods
            original_create_completion = OpenAIServingCompletion.create_completion
            original_create_chat_completion = OpenAIServingChat.create_chat_completion
            
            # Now import our patch module, which should apply the patches
            import agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion
            
            # Import the patch functions to compare
            from agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_serving_completion import (
                create_completion_patch,
                create_chat_completion_patch
            )
            
            # Verify that the patches were applied
            self.assertEqual(OpenAIServingCompletion.create_completion, create_completion_patch)
            self.assertEqual(OpenAIServingChat.create_chat_completion, create_chat_completion_patch)
            
            # Verify they're different from the original methods
            self.assertNotEqual(OpenAIServingCompletion.create_completion, original_create_completion)
            self.assertNotEqual(OpenAIServingChat.create_chat_completion, original_create_chat_completion)


if __name__ == '__main__':
    unittest.main()
    