#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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


# Standard library imports
import os
import json
import asyncio
import cloudpickle
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypeVar

# Third-party library imports
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

# vLLM imports
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse, 
    CompletionRequest, CompletionResponse, ErrorResponse
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_adapter.async_server import AsyncServerBase
from agentic_rl.runner.scheduler.workload import InstanceWorkLoad
from agentic_rl.runner.scheduler.load_stat import WorkloadStatLogger, vllm_log_stats_periodically

_R = TypeVar("_R", default=Any)
logger = Loggers("vllm_server").get_logger()


def safe_ray_init(namespace: str = "default"):
    """Safely initialize Ray connection."""
    try:
        # First try to connect to an existing cluster
        ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster in namespace '{namespace}'.")
    except Exception as e:
        logger.warning(f"Failed to connect to Ray cluster. Trying to start local instance... error: {e}")
        ray.init(namespace=namespace, ignore_reinit_error=True)
        logger.info(f"Started local Ray instance in namespace '{namespace}'.")
        

class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the executor."""
        self.is_first_wake_up = True
        if self.vllm_config.instance_id is None:
            raise ValueError("instance_id must be set for external ray actors.")

        fields = self.vllm_config.instance_id.split(":")
        if len(fields) != 4:
            raise ValueError(
                f"instance_id: {self.vllm_config.instance_id} must be in the format of "
                f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
            )
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])
        # TODO: Ray occasionally fails here
        safe_ray_init(namespace)
        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        vllm_dp_rank = vllm_dp_rank * dp_size + dp_rank

        actors = [v["Name"] for k, v in ray.state.actors().items()]
        actor_names = [
            actor_name
            for actor_name in actors
            if "ActorHybridWorker" in actor_name or "IntegratedWorker" in actor_name
        ]
        actor_names = sorted(actor_names, key=lambda x: int(x.split("_")[1]))
        self.actor_names = actor_names[vllm_dp_rank * vllm_tp_size: (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers = [ray.get_actor(actor_name, namespace=namespace) for actor_name in self.actor_names]
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        logger.info(f"instance_id: {self.vllm_config.instance_id} initializes finished, workers={self.actor_names}.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Execute collective RPC call."""
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to serialization/deserialization.
        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        return outputs

    def check_health(self) -> None:
        """Check executor health status."""
        return

    def wake_up(self, tags: Optional[list[str]] = None):
        """Wake up the executor."""
        if self.is_first_wake_up:
            self.is_sleeping = True
            self.is_first_wake_up = False
        super().wake_up(tags)


class AsyncVLLMServer(AsyncServerBase):
    """
    AsyncVLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e. AsyncActorRolloutRefWorker.

    AsyncVLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig,
                 tokenizer_name_or_path: str,
                 vllm_dp_size: int,
                 vllm_dp_rank: int,
                 wg_prefix: str):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None
        self.ins_workload: Optional[InstanceWorkLoad] = None

    async def init_engine(self):
        """Initialize vLLM AsyncLLM engine."""
        config = self.config
        model_path = self.tokenizer_name_or_path
        model_name = "/".join(model_path.split("/")[-1:])
        local_path = model_path
        trust_remote_code = config.trust_remote_code

        tensor_parallel_size = config.infer_tensor_parallel_size
        pipeline_parallel_size = config.infer_pipeline_parallel_size
        max_model_len = config.max_model_len
        # max_model_len = max(max_model_len, 32768)
        max_num_batched_tokens = config.max_num_batched_tokens

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        sampling_config = config.sampling_config
        # fix: start vllm engine needs `max_new_tokens``
        kwargs = dict(
            n=1,
            logprobs=sampling_config.logprobs,
            max_new_tokens=sampling_config.max_tokens,
            top_p=sampling_config.top_p,
            top_k=sampling_config.top_k,
            min_p=sampling_config.min_p,
            temperature=sampling_config.temperature,
        )
        dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))

        logger.info(f"override_generation_config: {kwargs}")
        logger.info(f"max_num_batched_tokens={max_num_batched_tokens}, [attention] this num > 8k may cause error for 'chunked prefill'!")
        cudagraph_capture_sizes = None
        # Not None and array config not empty
        if config.cudagraph_capture_sizes is not None and config.cudagraph_capture_sizes:
            cudagraph_capture_sizes = [int(size) for size in config.cudagraph_capture_sizes.replace(" ", "").split(',')]
        additional_config = {"ascend_scheduler_config": {"enabled": not config.enforce_eager,
                             "enable_chunked_prefill": config.enable_chunked_prefill, },
                             "enable_weight_nz_layout": True}
        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.enable_sleep_mode,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=dp_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="dummy" if config.load_format == "megatron" else config.load_format,
            # disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_expert_parallel=config.enable_expert_parallel,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            max_num_seqs=config.max_num_seqs,
            hf_overrides={"max_position_embeddings": max_model_len},
            compilation_config=CompilationConfig(cudagraph_capture_sizes=cudagraph_capture_sizes),
            additional_config=additional_config,
            worker_extension_cls="agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions"
        )
        # init async llm engine
        self.ins_workload = InstanceWorkLoad(dp_size=dp_size)
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        vllm_config.workload = self.ins_workload
        self.engine = AsyncLLM.from_vllm_config(vllm_config, disable_log_stats=config.disable_log_stats,
            stat_loggers=[WorkloadStatLogger])

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            # return_tokens_as_token_ids=True,
        )

        self.openai_serving_completion = OpenAIServingCompletion(
            self.engine,
            model_config,
            models,
            request_logger=None,
            return_tokens_as_token_ids=True,
        )
        if not config.disable_log_stats:
            asyncio.create_task(vllm_log_stats_periodically(self))
        logger.info(f"Async vLLM Server running at {await self.get_server_address()}")


    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            if not isinstance(generator, ChatCompletionResponse):
                raise TypeError("Expected ChatCompletionResponse")
            return JSONResponse(content=generator.model_dump())

    async def completions(self, raw_request: Request):
        """OpenAI completions API.

        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        request_json = await raw_request.json()
        request = CompletionRequest(**request_json)
        generator = await self.openai_serving_completion.create_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            if not isinstance(generator, CompletionResponse):
                raise TypeError("Expected CompletionResponse")
            if generator.choices and generator.choices[0].logprobs:
                generator.choices[0].logprobs.token_logprobs = []
                generator.choices[0].logprobs.top_logprobs = []
                generator.choices[0].logprobs.text_offset = []
            return JSONResponse(content=generator.model_dump())

    async def chat_completion_generator(self, request: ChatCompletionRequest) -> AsyncGenerator[Tuple[int, str]]:
        """Direct chat completion without FastAPI.

        Args:
            request: ChatCompletionRequest, request object.

        Returns:
            AsyncGenerator[Tuple[int, str]]: async generator of (status_code, data) pairs.
        """
        generator = await self.openai_serving_chat.create_chat_completion(request)
        if isinstance(generator, ErrorResponse):
            data = generator.model_dump_json(exclude_unset=True)
            yield generator.error.code, f"data: {data}\n\n"

        if request.stream:
            async for chunk in generator:
                yield 200, chunk
        else:
            if not isinstance(generator, ChatCompletionResponse):
                raise TypeError("Expected ChatCompletionResponse")
            data = generator.model_dump_json(exclude_unset=True)
            yield 200, f"data: {data}\n\n"

    async def wake_up(self, tags: Optional[list[str]] = None):
        await self.engine.wake_up(tags)

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

    async def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> list[_R]:
        return await self.engine.collective_rpc(method, timeout, args, kwargs)

    async def get_workload(self, raw_request: Request):
        return JSONResponse(content=json.dumps(self.ins_workload.to_dict()), status_code=200)

    async def cancel_requests(self, raw_request: Request):
        """Cancel requests from AsyncVLLMServer."""
        request_json = await raw_request.json()
        logger.info(f"vllm async server abort requests: {request_json}")
        await self.engine.abort(request_json["requests"])

    async def reset_prefix_cache(self):
        """Reset the prefix cache."""
        await self.engine.reset_prefix_cache()