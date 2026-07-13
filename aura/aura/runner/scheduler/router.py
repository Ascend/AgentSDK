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

import os
import asyncio
import time
import uuid

import numpy as np
import torch
from openai.types.chat import ChatCompletionChunk
from openai.types.completion import Completion

from aura.base.log.loggers import Loggers
from aura.base.misc.misc import app_stats
from aura.base.utils.globals import is_pd_separate
from aura.runner.scheduler.req_scheduler import SchedulerFactory

logger = Loggers(__name__).get_logger()

DEFAULT_OPENAI_API_RETRIES = 3
RATE_LIMIT_RETRY_SLEEP_SECONDS = 5


def _repeat_interleave(value: torch.Tensor | np.ndarray, repeats: int) -> torch.Tensor | np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)


async def poll_completions_openai_stream(address: str, stream_queue=None, **completions_request) -> Completion:
    address, local_dp_rank = address.split(
        '-'
    )  # http://xx.xx.xx.xx/v1-0 TODO: Address suffix will be optimized later, not a reasonable design
    messages = completions_request['prompt']
    model = completions_request['model']
    max_tokens = completions_request['max_tokens']
    engine_args = {"base_url": address, "api_key": 'EMPTY'}

    logger.info(f'# engine_args={engine_args}')
    import openai
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(trust_env=False)
    try:
        client = AsyncOpenAI(**engine_args, http_client=http_client)
        # Remove meta_info if present
        if "meta_info" in completions_request:
            completions_request.pop("meta_info")
        # Remove extra_headers from the payload
        if "extra_headers" in completions_request:
            completions_request.pop("extra_headers")

        retries = DEFAULT_OPENAI_API_RETRIES

        while retries > 0:
            try:
                # Use openai.chat's stream=True interface
                response = await client.completions.create(
                    messages=messages,
                    model=model,
                    timeout=3600,
                    stream=True,
                    max_tokens=max_tokens,
                )

                full_response = ""
                async for chunk in response:
                    logger.info(f"chunk={chunk}")
                    chunk: ChatCompletionChunk = chunk
                    delta = chunk.choices[0].delta
                    text = delta.content or ""  # 可能是None
                    full_response += text
                    if stream_queue:
                        stream_queue.put_nowait(
                            {
                                "event": "raw_response_event",
                                "data": {"response": chunk.model_dump_json(), "type": "raw_response_event"},
                            }
                        )
                return full_response

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                logger.info("Sleep for %s seconds for API limit.", RATE_LIMIT_RETRY_SLEEP_SECONDS)
                await asyncio.sleep(RATE_LIMIT_RETRY_SLEEP_SECONDS)
            except Exception as e:
                logger.error("Error: %s", e)
                return f"Error processing content: {e}"
    finally:
        await http_client.aclose()


async def poll_completions_openai(dp_address: str, stream_queue=None, role="h", **completions_request) -> Completion:
    address, local_dp_rank = dp_address.split('-') if '-' in dp_address else (dp_address, 0)
    # Use aiohttp directly instead of AsyncOpenAI to avoid potential blocking
    base_url = f"http://{address}/v1"
    original_request_id = completions_request.pop("request_id")
    original_request_id = original_request_id + f"-{role}"
    # Remove meta_info if present
    if "meta_info" in completions_request:
        completions_request.pop("meta_info")
    # Remove extra_headers from the payload
    if "extra_headers" in completions_request:
        completions_request.pop("extra_headers")
    prompt = completions_request['prompt']
    model = completions_request['model']
    max_tokens = completions_request['max_tokens']

    max_retries = DEFAULT_OPENAI_API_RETRIES
    request_nonce = uuid.uuid4().hex[:8]

    engine_args = {"base_url": base_url, "api_key": 'EMPTY'}
    logger.info(f'# engine_args={engine_args}')
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(trust_env=False)
    try:
        client = AsyncOpenAI(**engine_args, http_client=http_client)

        for retry in range(max_retries):
            request_id = (
                f"{original_request_id}-{request_nonce}-r{retry}"
                if retry > 0
                else f"{original_request_id}-{request_nonce}"
            )
            try:
                headers = {
                    "X-Request-Id": request_id,
                    "X-Dp-Rank": local_dp_rank,
                }
                # Use openai.chat's stream=True interface
                response = await client.completions.create(
                    prompt=prompt,
                    model=model,
                    timeout=3600,
                    stream=False,
                    max_tokens=max_tokens,
                    extra_headers=headers,
                )

                choices = response.choices[0]
                full_response = choices.text
                logprobs = choices.logprobs.token_logprobs
                response_tokens = choices.token_ids
                prompt_tokens = choices.prompt_token_ids
                http_response = {
                    "message": full_response,
                    "logprobs": logprobs,
                    "response_tokens": response_tokens,
                    "prompt_tokens": prompt_tokens,
                }
                # prefill instance need to return response
                if is_pd_separate():
                    return response
                return http_response

            except Exception as e:
                max_retries -= 1
                if max_retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                import traceback

                traceback.print_exc()
                logger.error(f"poll_completions_openai failed try next! Error for {request_id}: {e}")

        # This should never be reached due to the raise in the loop, but mypy requires it
        raise Exception("All retries failed")
    finally:
        await http_client.aclose()


async def poll_chat_completions_openai(
    dp_address: str, stream_queue=None, role="h", **completions_request
) -> Completion:
    address, local_dp_rank = dp_address.split('-') if '-' in dp_address else (dp_address, 0)
    # Use aiohttp directly instead of AsyncOpenAI to avoid potential blocking
    base_url = f"http://{address}/v1"
    original_request_id = completions_request.pop("request_id")
    original_request_id = original_request_id + f"-{role}"
    # Remove meta_info if present
    if "meta_info" in completions_request:
        completions_request.pop("meta_info")
    # Remove extra_headers from the payload
    if "extra_headers" in completions_request:
        completions_request.pop("extra_headers")
    messages = completions_request['prompt']
    model = completions_request['model']
    max_tokens = completions_request['max_tokens']

    max_retries = DEFAULT_OPENAI_API_RETRIES
    request_nonce = uuid.uuid4().hex[:8]

    engine_args = {"base_url": base_url, "api_key": 'EMPTY'}
    logger.info(f'# engine_args={engine_args}')
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(trust_env=False)
    try:
        client = AsyncOpenAI(**engine_args, http_client=http_client)

        for retry in range(max_retries):
            request_id = (
                f"{original_request_id}-{request_nonce}-r{retry}"
                if retry > 0
                else f"{original_request_id}-{request_nonce}"
            )
            try:
                headers = {
                    "X-Request-Id": request_id,
                    "X-Dp-Rank": local_dp_rank,
                }
                # 使用 openai.chat 的 stream=True 接口
                response = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    timeout=3600,
                    stream=False,
                    max_tokens=max_tokens,
                    extra_headers=headers,
                )

                logger.info(f"response={response}")
                full_response = response.choices[0].message.content
                # prefill instance need to return response
                if is_pd_separate():
                    return response
                return full_response

            except Exception as e:
                max_retries -= 1
                if max_retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                import traceback

                traceback.print_exc()
                logger.error(f"poll_completions_openai failed try next! Error for {request_id}: {e}")

        # This should never be reached due to the raise in the loop, but mypy requires it
        raise Exception("All retries failed")
    finally:
        await http_client.aclose()


class Router:
    _router = None
    """
    Router chooses the least-used server address from a static list of
    server addresses across multiple processes using asyncio locks.
    """

    def __init__(
        self, tokenizer_name_or_path, tokenizer, addresses: list[str], token_in_token_out=False, model_name=None
    ):
        # List of "ip:port" strings
        self.addresses = addresses
        # self.tensor_parallel_size = generate_config.infer_tensor_parallel_size  # config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
        self.dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        self.counter = 0

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        # model_path = config.actor_rollout_ref.model.path
        self.model_path = tokenizer_name_or_path
        # self.model_name = "/".join(model_path.split("/")[-2:])
        self.model_name = "/".join(self.model_path.split("/")[-2:]) if model_name is None else model_name
        self.scheduler = SchedulerFactory.get_scheduler(self.addresses, self.dp_size, None)
        self.token_in_token_out = token_in_token_out

    @classmethod
    def create(
        cls, tokenizer_name_or_path, tokenizer, addresses: list[str], token_in_token_out=False, model_name=None
    ) -> "Router":
        if cls._router is None and addresses is not None:
            class_name = RouterPDSep if is_pd_separate() else Router
            cls._router = class_name(
                tokenizer_name_or_path=tokenizer_name_or_path,
                tokenizer=tokenizer,
                addresses=addresses,
                token_in_token_out=token_in_token_out,
                model_name=model_name,
            )
        return cls._router

    @staticmethod
    def cal_request_id(application_id: str, step_idx: int):
        # vllm will concatenate once more, and the final format will be: cmpl-${application_id}--${step_idx}-0
        return application_id + "--" + str(step_idx)

    def update_address(self, addresses):
        if not addresses:
            logger.warning("Router update_address received empty addresses, skip update.")
            return
        if addresses == self.addresses:
            return
        self.addresses = addresses
        if getattr(self, "scheduler", None) is None:
            self.scheduler = SchedulerFactory.get_scheduler(self.addresses, self.dp_size, None)
            return
        dp_size = max(int(self.dp_size or 0), 1)
        self.scheduler.on_ins_address_updated(addresses, dp_size)

    async def chat(self, prompt, application_id, default_simpling, stream_queue=None, **kwargs):
        step_idx = kwargs.pop("step_idx")
        request_id = self.cal_request_id(application_id, step_idx)
        default_kwargs = dict(
            n=1,  # self.config.actor_rollout_ref.rollout.n,
            request_id=request_id,
        )
        logger.info(f"default simpling: {default_simpling}")
        logger.info(f"kwargs: {kwargs}")
        merged_kwargs = {**default_kwargs, **default_simpling, **kwargs}
        # Same keys will be overridden by kwargs
        logger.debug(f"*************prompt******************\n{prompt}")

        address = await self.scheduler.schedule(application_id, request_id)
        if address is None:
            # Terminate inference
            return None
        app_stats.stat_route(application_id, request_id, address, len(prompt))

        logger.info(
            f"trajectory performance status, chat start time:{time.time()}, appID:{application_id}, "
            f"address:{address}, request_id:{request_id}"
        )
        try:
            if stream_queue is None:
                if self.token_in_token_out:
                    response = await poll_completions_openai(address, prompt=prompt, **merged_kwargs)
                else:
                    response = await poll_chat_completions_openai(address, prompt=prompt, **merged_kwargs)
            else:
                response = await poll_completions_openai_stream(
                    address, prompt=prompt, stream_queue=stream_queue, **merged_kwargs
                )
        finally:
            await self.scheduler.release(address, application_id, request_id)
        return response

    async def stop(self):
        await self.scheduler.cancel_requests()

    def reset(self):
        self.scheduler.reset()

    async def cancel_request(self, application_id):
        await self.scheduler.cancel_request(application_id)


class RouterPDSep:
    """
    Router chooses the least-used server address from a static list of
    server addresses across multiple processes using asyncio locks.
    """

    def __init__(
        self, tokenizer_name_or_path, tokenizer, addresses: list[str], token_in_token_out=False, model_name=None
    ):
        # common params
        self.dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        self._lock = asyncio.Lock()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.model_path = tokenizer_name_or_path
        self.model_name = "/".join(self.model_path.split("/")[-2:]) if model_name is None else model_name
        p_addresses, d_addresses = self.get_pd_addresses(addresses)
        self.p_scheduler = SchedulerFactory.get_scheduler(p_addresses, self.dp_size, None, role="prefill")
        self.d_scheduler = SchedulerFactory.get_scheduler(d_addresses, self.dp_size, None, role="decode")

    def get_pd_addresses(self, addresses):
        # ['prefill-172.16.14.168:60383', 'prefill-172.16.13.246:51751', 'decode-172.16.12.95:37101', 'decode-172.16.11.221:34399']
        prefill_addresses = []
        decode_addresses = []
        for addr in addresses:
            if "prefill" in addr:
                prefill_addresses.append(addr.split('-', 2)[1])
            if "decode" in addr:
                decode_addresses.append(addr.split('-', 2)[1])
        return prefill_addresses, decode_addresses

    @staticmethod
    def cal_request_id(application_id: str, step_idx: int):
        # VLLM will be spliced again, and the final format will be cmpl-$ {application_id}-- $ {step_idx}-0
        return application_id + "--" + str(step_idx)

    def update_address(self, addresses):
        pass

    async def chat_with_prefill(self, prompt, application_id, default_sampling, **kwargs):
        step_idx = kwargs.pop("step_idx")
        request_id = self.cal_request_id(application_id, step_idx)
        default_kwargs = dict(
            n=1,  # self.config.actor_rollout_ref.rollout.n,
            request_id=request_id,
        )
        logger.info(f"default sampling: {default_sampling}")
        logger.info(f"kwargs: {kwargs}")
        merged_kwargs = {**default_kwargs, **default_sampling, **kwargs}
        # Same keys will be overridden by kwargs
        logger.debug(f"*************prompt******************\n{prompt}")
        sched_start_time = time.time()
        address = await self.p_scheduler.schedule(application_id, request_id)
        if address is None:
            logger.error("RouterPDSep.chat_with_prefill failed to schedule an infer instance to run!")
            return None
        prefill_start_time = time.time()

        if address is None:  # end of inference
            return None
        # app_stats_prefill.stat_route(application_id, request_id, prefill_address, len(prompt))
        logger.info(
            f"prefill trajectory performance status, chat start time:{time.time()}, appID:{application_id}, "
            f"address:{address}, request_id:{request_id}"
        )

        # prefill instance only support non stream inference
        response = await poll_completions_openai(
            address, model="/".join(self.model_path.split("/")[-1:]), prompt=prompt, role='p', **merged_kwargs
        )
        end_time = time.time()
        logger.debug(f"\nprefill response: \n{response}\n")

        await self.p_scheduler.release(address, application_id, request_id)
        return response, (prefill_start_time - sched_start_time), (end_time - prefill_start_time)

    async def chat(self, prompt, application_id, default_sampling, stream_queue=None, **kwargs):
        # get prefill response first
        prefill_response, sched_time1, prefill_time = await self.chat_with_prefill(
            prompt, application_id, default_sampling, **kwargs
        )
        if prefill_response is None:
            return None
        # kv_transfer_params = prefill_response.get('kv_transfer_params', {})
        kv_transfer_params = prefill_response.kv_transfer_params
        logger.debug(f"prefill kv_transfer_params: {kv_transfer_params}\n\nstream_queue: {stream_queue}")

        step_idx = kwargs.pop("step_idx")
        request_id = self.cal_request_id(application_id, step_idx)
        default_kwargs = dict(
            n=1,  # self.config.actor_rollout_ref.rollout.n,
            request_id=request_id,
        )
        logger.info(f"default sampling: {default_sampling}")
        logger.info(f"kwargs: {kwargs}")
        merged_kwargs = {**default_kwargs, **default_sampling, **kwargs}
        # Same keys will be overridden by kwargs
        logger.debug(f"*************prompt******************\n{prompt}")
        # get decode address
        sched_start_time = time.time()
        address = await self.d_scheduler.schedule(application_id, request_id)
        if address is None:
            logger.error("RouterPDSep.chat failed to schedule an infer instance to run!")
            return None

        logger.info(
            f"decode trajectory performance status, chat start time:{time.time()}, appID:{application_id}, "
            f"address:{address}, request_id:{request_id}"
        )
        decode_start_time = time.time()
        if stream_queue is None:
            full_response = await poll_completions_openai(
                address,
                model="/".join(self.model_path.split("/")[-1:]),
                prompt=prompt,
                role='d',
                kv_transfer_params=kv_transfer_params,
                **merged_kwargs,
            )
            response = full_response.choices[0].message.content
        else:
            response = await poll_completions_openai_stream(
                address,
                model=self.model_name,
                prompt=prompt,
                stream_queue=stream_queue,
                kv_transfer_params=kv_transfer_params,
                **merged_kwargs,
            )
        await self.d_scheduler.release(address, application_id, request_id)
        end_time = time.time()
        logger.info(
            f"trajectory step status, appID:{application_id}, step_idx:{step_idx} sched-prefill:{sched_time1}"
            f" sched-decode:{decode_start_time - sched_start_time} prefill_time:{prefill_time} decode_time:{end_time - decode_start_time}"
        )
        return response

    async def stop(self):
        pass

    def reset(self):
        pass

    async def cancel_request(self, application_id):
        pass
