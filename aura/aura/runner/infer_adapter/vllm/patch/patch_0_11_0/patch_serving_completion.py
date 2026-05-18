#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Optional, Union, cast

import jinja2
from fastapi import Request
from typing_extensions import assert_never

from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.logger import init_logger
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import EmbedsPrompt, TokensPrompt, is_embeds_prompt, is_tokens_prompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import merge_async_iterators
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

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

    # If the engine is dead, raise the engine's DEAD_ERROR.
    # This is required for the streaming case, where we return a
    # success status before we actually start generating text :).
    if self.engine_client.errored:
        raise self.engine_client.dead_error

    # Return error for unsupported features.
    if request.suffix is not None:
        return self.create_error_response("suffix is not currently supported")

    if request.echo and request.prompt_embeds is not None:
        return self.create_error_response("Echo is unsupported with prompt embeds.")

    if request.prompt_logprobs is not None and request.prompt_embeds is not None:
        return self.create_error_response("prompt_logprobs is not compatible with prompt embeds.")

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
            tokenizer = await self.engine_client.get_tokenizer()
        renderer = self._get_renderer(tokenizer)

        engine_prompts = await renderer.render_prompt_and_embeds(
            prompt_or_prompts=request.prompt,
            prompt_embeds=request.prompt_embeds,
            config=self._build_render_config(request),
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
    generators: list[AsyncGenerator[RequestOutput, None]] = []
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
                sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.model_config.logits_processor_pattern,
                    self.default_sampling_params,
                )

            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

            # Mypy inconsistently requires this second cast in different
            # environments. It shouldn't be necessary (redundant from above)
            # but pre-commit in CI fails without it.
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
                    data_parallel_rank=None if dp_rank is None else int(dp_rank),
                )

            generators.append(generator)
    except ValueError as e:
        return self.create_error_response(str(e))

    result_generator = merge_async_iterators(*generators)

    model_name = self.models.model_name(lora_request)
    num_prompts = len(engine_prompts)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. Noting that best_of is only supported in V0. In addition,
    # we do not stream the results when use beam search.
    stream = (
        request.stream and (request.best_of is None or request.n == request.best_of) and not request.use_beam_search
    )

    # Streaming response
    if stream:
        return self.completion_stream_generator(
            request,
            engine_prompts,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata,
            enable_force_include_usage=self.enable_force_include_usage,
        )

    # Non-streaming response
    final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res

        for i, final_res in enumerate(final_res_batch):
            if final_res is None:
                raise RuntimeError(f"Failed to get response for prompt at index {i}")

            # The output should contain the input text
            # We did not pass it into vLLM engine to avoid being redundant
            # with the inputs token IDs
            if final_res.prompt is None:
                engine_prompt = engine_prompts[i]
                final_res.prompt = None if is_embeds_prompt(engine_prompt) else engine_prompt.get("prompt")

        final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

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

    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    if request.stream:
        response_json = response.model_dump_json()

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return fake_stream_generator()

    return response


OpenAIServingCompletion.create_completion = create_completion_patch
