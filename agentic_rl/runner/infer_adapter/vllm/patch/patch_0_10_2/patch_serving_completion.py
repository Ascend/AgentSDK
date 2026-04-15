#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import AsyncGenerator
from typing import List, Optional, Union, cast

import jinja2
from fastapi import Request
from typing_extensions import assert_never

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    TextTokensPrompt,
                                                    clamp_prompt_logprobs,
                                                    is_text_tokens_prompt)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import (EmbedsPrompt, TokensPrompt, is_embeds_prompt,
                              is_tokens_prompt)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids,
                                                validate_request_params)
from vllm.utils import as_list, merge_async_iterators

logger = init_logger(__name__)


async def create_completion_patch(
    self,
    request: CompletionRequest,
    raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following feature:
        - suffix (the language models we currently support do not support
        suffix)
    """
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if self.engine_client.errored:
        raise self.engine_client.dead_error

    if request.suffix is not None:
        return self.create_error_response("suffix is not currently supported")

    if request.echo and request.prompt_embeds is not None:
        return self.create_error_response("Echo is unsupported with prompt embeds.")

    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    created_time = int(time.time())
    dp_rank = raw_request.headers.get("X-Dp-Rank", None)

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    try:
        lora_request = self._maybe_get_adapters(request)

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

        request_prompts, engine_prompts = await self._preprocess_completion(
            request,
            tokenizer,
            request.prompt,
            add_special_tokens=request.add_special_tokens,
        )
    except ValueError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except TypeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except RuntimeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except jinja2.TemplateError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))

    # Schedule the request and get the result generator.
    generators: List[AsyncGenerator[RequestOutput, None]] = []
    try:
        for i, engine_prompt in enumerate(engine_prompts):
            sampling_params: Union[SamplingParams, BeamSearchParams]
            # Mypy does not infer that engine_prompt will have only one of
            # "prompt_token_ids" or "prompt_embeds" defined, and both of
            # these as Union[object, the expected type], where it infers
            # object if engine_prompt is a subclass of one of the
            # typeddicts that defines both keys. Worse, because of
            # https://github.com/python/mypy/issues/8586, mypy does not
            # infer the type of engine_prompt correctly because of the
            # enumerate. So we need an unnecessary cast here.
            engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt], engine_prompt)
            
            if is_embeds_prompt(engine_prompt):
                input_length = len(engine_prompt["prompt_embeds"])
            elif is_tokens_prompt(engine_prompt):
                input_length = len(engine_prompt["prompt_token_ids"])
            else:
                assert_never(engine_prompt)

            if self.default_sampling_params is None:
                self.default_sampling_params = {}

            max_tokens = get_max_tokens(
                max_model_len=self.max_model_len,
                request=request,
                input_length=input_length,
                default_sampling_params=self.default_sampling_params,
            )

            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params)
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.model_config.logits_processor_pattern,
                    self.default_sampling_params,
                )

            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                request_prompts[i],
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (None if raw_request is None else
                             await self._get_trace_headers(raw_request.headers))

            engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt], engine_prompt)
            if isinstance(sampling_params, BeamSearchParams):
                generator = self.engine_client.beam_search(
                    prompt=engine_prompt,
                    request_id=request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                )
            else:
                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=None if dp_rank is None else int(dp_rank)
                )

            generators.append(generator)
    except ValueError as e:
        return self.create_error_response(str(e))

    result_generator = merge_async_iterators(*generators)

    model_name = self._get_model_name(request.model, lora_request)
    num_prompts = len(engine_prompts)

    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    if stream:
        return self.completion_stream_generator(
            request,
            request_prompts,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata,
            enable_force_include_usage=self.enable_force_include_usage,
        )

    final_res_batch: List[Optional[RequestOutput]] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res

        for i, final_res in enumerate(final_res_batch):
            if final_res is None:
                raise RuntimeError(
                    f"Request output at index {i} is None, which should not happen"
                )

            if final_res.prompt is None:
                request_prompt = request_prompts[i]
                if is_text_tokens_prompt(request_prompt):
                    final_res.prompt = request_prompt["prompt"]
                else:
                    final_res.prompt = None

        final_res_batch_checked = cast(List[RequestOutput], final_res_batch)

        response = self.request_output_to_completion_response(
            final_res_batch_checked,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")
    except ValueError as e:
        return self.create_error_response(str(e))

    if request.stream:
        response_json = response.model_dump_json()

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return fake_stream_generator()

    return response


async def create_chat_completion_patch(
    self,
    request: ChatCompletionRequest,
    raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
    """
    Chat Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI
    Chat Completion API.
    """
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        logger.error("Error with model %s", error_check_ret)
        return error_check_ret

    if self.engine_client.errored:
        raise self.engine_client.dead_error

    try:
        lora_request = self._maybe_get_adapters(
            request, supports_default_mm_loras=True)

        model_name = self._get_model_name(request.model, lora_request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        tool_parser = self.tool_parser

        if isinstance(tokenizer, MistralTokenizer):
            maybe_serialize_tool_calls(request)
            truncate_tool_call_ids(request)
            validate_request_params(request)

        if (request.tool_choice == "auto" and
                not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony):
            return self.create_error_response(
                "\"auto\" tool choice requires "
                "--enable-auto-tool-choice and --tool-call-parser to be set"
            )

        if (request.tools is None
                or (request.tool_choice == "none"
                    and self.exclude_tools_when_tool_choice_none)):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        if not self.use_harmony:
            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                add_special_tokens=request.add_special_tokens,
            )
        else:
            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = self._make_request_with_harmony(request)
    except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(f"{e} {e.__cause__}")

    dp_rank = raw_request.headers.get("X-Dp-Rank", None) if raw_request else None
    request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
    request_metadata = RequestResponseMetadata(request_id=request.request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    generators: List[AsyncGenerator[RequestOutput, None]] = []
    try:
        for i, engine_prompt in enumerate(engine_prompts):
            sampling_params: Union[SamplingParams, BeamSearchParams]

            if self.default_sampling_params is None:
                self.default_sampling_params = {}

            max_tokens = get_max_tokens(
                max_model_len=self.max_model_len,
                request=request,
                input_length=len(engine_prompt["prompt_token_ids"]),
                default_sampling_params=self.default_sampling_params)

            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params)
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.model_config.logits_processor_pattern,
                    self.default_sampling_params)

            self._log_inputs(
                request_id,
                request_prompts[i],
                params=sampling_params,
                lora_request=lora_request)

            trace_headers = (None if raw_request is None else
                             await self._get_trace_headers(raw_request.headers))

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.engine_client.beam_search(
                    prompt=engine_prompt,
                    request_id=request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                )
            else:
                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=None if dp_rank is None else int(dp_rank)
                )

            generators.append(generator)
    except ValueError as e:
        return self.create_error_response(str(e))

    if len(generators) != 1:
        raise RuntimeError(
            f"Expected exactly one generator for chat completion, but got {len(generators)}"
        )
    
    result_generator = generators[0]

    if request.stream:
        return self.chat_completion_stream_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            enable_force_include_usage=self.enable_force_include_usage)

    try:
        return await self.chat_completion_full_generator(
            request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata)
    except ValueError as e:
        return self.create_error_response(str(e))


OpenAIServingCompletion.create_completion = create_completion_patch
OpenAIServingChat.create_chat_completion = create_chat_completion_patch