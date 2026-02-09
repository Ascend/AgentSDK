#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
from fastapi import FastAPI
import ray
from ray.exceptions import RayError
import uvicorn
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params, CompletionRequestChecker
from agentic_rl.base.utils.ray_secure_init import ray_secure_init
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig, SamplingConfig
from agentic_rl.runner.infer_adapter.async_server_base import AsyncServerBase

logger = Loggers(__name__)


@dataclass
class VLLMConfig:
    """Configuration for VLLM server initialization."""
    model_path: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    dtype: str
    enforce_eager: bool
    gpu_memory_utilization: float
    load_format: str
    enable_sleep_mode: bool
    enable_prefix_caching: bool
    trust_remote_code: bool
    max_num_seqs: int
    sampling_config: SamplingConfig


@dataclass
class ServerConfig:
    """Configuration for server instance."""
    vllm_dp_size: int
    vllm_dp_rank: int
    wg_prefix: str
    namespace: str


# Constants
MAX_BATCHED_TOKENS_WARNING_THRESHOLD = 8192


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    @validate_params(
        method=dict(
            validator=lambda x: isinstance(x, str) or isinstance(x, Callable),
            message="method must be a string or Callable",
        ),
        timeout=dict(validator=lambda x: x is None or isinstance(x, float), message="timeout must be a float or None"),
        args=dict(validator=lambda x: isinstance(x, tuple), message="args must be a tuple"),
        kwargs=dict(validator=lambda x: x is None or isinstance(x, dict), message="kwargs must be a dict or None"),
    )
    def collective_rpc(
            self,
            method: Union[str, Callable],
            timeout: Optional[float] = None,
            args: Tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Execute a method on all workers via RPC."""
        if not hasattr(self, 'workers') or not self.workers:
            raise RuntimeError("Workers not initialized. Call _init_executor first.")

        try:
            sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
            del method
        except RuntimeError as e:
            raise RuntimeError(f"Failed to serialize method: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred, failed to serialize method: {e}") from e

        try:
            outputs = ray.get(
                [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers],
                timeout=timeout
            )
            return outputs
        except RayError as e:
            raise RuntimeError(f"RayError during collective_rpc for method '{sent_method}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during collective_rpc for method '{sent_method}': {e}") from e

    def check_health(self):
        return

    def wake_up(self, tags: Optional[list[str]] = None):
        """Wake up the executor."""
        if tags and not isinstance(tags, list):
            raise ValueError("tags must be a list")

        if self.is_fisrt_wake_up:
            self.is_sleeping = True
            self.is_fisrt_wake_up = False
        super().wake_up(tags)

    def _init_executor(self) -> None:
        self.is_fisrt_wake_up = True
        if self.vllm_config.instance_id is None:
            raise RuntimeError("instance_id must be set for external ray actors.")

        try:
            fields = self.vllm_config.instance_id.split(":")
            if len(fields) != 4:
                raise RuntimeError(f"instance_id: {self.vllm_config.instance_id} must be in the format of "
                                   f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>.")
            namespace, vllm_dp_rank = fields[0], int(fields[3])
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to parse instance_id '{self.vllm_config.instance_id}': {e}") from e

        ray_secure_init(address="auto", extra_init_kwargs={
            "namespace": namespace,
            'ignore_reinit_error': True,
            'runtime_env': {"worker_process_setup_hook": 'agentic_rl.base.utils.logger_patch.patch'}
        })

        try:
            vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        except AttributeError as e:
            raise RuntimeError(f"Failed to access tensor_parallel_size from vllm_config: {e}") from e

        try:
            actors = [v["Name"] for _, v in ray.state.actors().items()]
            actor_names = [actor_name
                           for actor_name in actors
                           if "ActorHybridWorker" in actor_name or "IntegratedWorker" in actor_name]
            actor_names = sorted(actor_names, key=lambda x: int(x.split("_")[1]))
            self.actor_names = actor_names[vllm_dp_rank * vllm_tp_size: (vllm_dp_rank + 1) * vllm_tp_size]

            if not self.actor_names:
                raise RuntimeError(f"No Ray actors found for vllm_dp_rank={vllm_dp_rank}, vllm_tp_size={vllm_tp_size}. "
                                   f"Available actors: {actor_names}")

            self.workers = [ray.get_actor(actor_name, namespace=namespace) for actor_name in self.actor_names]
            logger.info(f"Found and retrieved {len(self.workers)} workers: {self.actor_names}")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to discover Ray actors: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Ray actor discovery: {e}") from e
        kwargs = dict(vllm_config=self.vllm_config, local_rank=None, rank=None,
                      distributed_init_method="env://", is_driver_worker=True)

        _ = self.collective_rpc("init_worker", args=([kwargs],))
        _ = self.collective_rpc("init_device")
        _ = self.collective_rpc("load_model")
        logger.info(f"instance_id: {self.vllm_config.instance_id} init finished, workers={self.actor_names}.")


@ray.remote(num_cpus=1)
class AsyncVLLMServer(AsyncServerBase):
    """
    AsyncVLLMServer provides asynchronous VLLM inference capabilities with Ray-based distributed execution.

    This server wraps AsyncLLM and uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers (AsyncActorRolloutRefWorker instances).
    """

    @validate_params(
        agentic_rl_config=dict(
            validator=lambda x: isinstance(x, AgenticRLConfig), message="agentic_rl_config must be a AgenticRLConfig"
        ),
        vllm_dp_size=dict(validator=lambda x: isinstance(x, int), message="vllm_dp_size must be an integer"),
        vllm_dp_rank=dict(validator=lambda x: isinstance(x, int), message="vllm_dp_rank must be an integer"),
        wg_prefix=dict(validator=lambda x: isinstance(x, str), message="wg_prefix must be a string"),
    )
    def __init__(self, config, agentic_rl_config, tokenizer_name_or_path, vllm_dp_size, vllm_dp_rank, wg_prefix):
        super().__init__()
        self.config = config
        self.agentic_rl_config = agentic_rl_config

        namespace = ray.get_runtime_context().namespace
        self.server_config = ServerConfig(
            vllm_dp_size=vllm_dp_size,
            vllm_dp_rank=vllm_dp_rank,
            wg_prefix=wg_prefix,
            namespace=namespace
        )

        self.vllm_config = self._build_vllm_config(config, tokenizer_name_or_path)

        self.engine: Optional[AsyncLLM] = None
        self.openai_serving_chat: Optional[OpenAIServingChat] = None
        self.openai_serving_completion: Optional[OpenAIServingCompletion] = None
        self._server_task = asyncio.create_task(self._start_fastapi_server())

        logger.info(f"AsyncVLLMServer initialized successfully for rank {vllm_dp_rank}/{vllm_dp_size}")

    async def _start_fastapi_server(self) -> None:
        """Start vllm inference server by fastapi and uvicorn"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info("Vllm inference server lifespan startup begin.")
            try:
                self.server_ready.set()
                yield
            finally:
                logger.info("Vllm inference server lifespan end")

        app = FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/completions", self.completions, methods=["POST"])
        app.router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])

        self.port = self._get_free_port()
        config = uvicorn.Config(
            app,
            host=self.address,
            port=self.port,
            log_level="warning",
            lifespan="on"
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except asyncio.CancelledError as e:
            logger.info("Vllm inference server canceled")
            server.should_exit = True
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during start vllm inference: {e}") from e
        finally:
            logger.info("Vllm inference server exited.")

    def _get_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((f"{self.address}", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _build_generation_config(sampling_config: SamplingConfig) -> Dict[str, Any]:
        return {
            "n": 1,
            "logprobs": sampling_config.logprobs,
            "max_new_tokens": sampling_config.max_tokens,
            "top_p": sampling_config.top_p,
            "top_k": sampling_config.top_k,
            "min_p": sampling_config.min_p,
            "temperature": sampling_config.temperature,
        }

    async def get_server_address(self) -> str:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    async def init_engine(self) -> None:
        """Initialize vLLM AsyncLLM engine and OpenAI serving components."""
        logger.info("Initializing VLLM AsyncLLM engine...")

        generation_config = self._build_generation_config(self.vllm_config.sampling_config)
        engine_args = self._build_engine_args(self.vllm_config, generation_config)

        try:
            # Create engine config and set instance ID
            vllm_config = engine_args.create_engine_config()
            vllm_config.instance_id = (
                f"{self.server_config.namespace}:{self.server_config.wg_prefix}:"
                f"{self.server_config.vllm_dp_size}:{self.server_config.vllm_dp_rank}"
            )
            logger.info(f"Engine instance_id: {vllm_config.instance_id}")
        except (ValueError, AssertionError) as e:
            raise RuntimeError(f"Failed to create engine config: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred when creating engine config: {e}") from e

        try:
            self.engine = AsyncLLM.from_vllm_config(vllm_config)
            model_config = self.engine.model_config
            logger.info("AsyncLLM engine created successfully")
        except ValueError as e:
            raise RuntimeError(f"Failed to initialize AsyncLLM engine: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred when initializing AsyncLLM engine: {e}") from e

        model_name = "/".join(self.vllm_config.model_path.split("/")[-1:])
        self._setup_openai_serving(model_config, model_name, self.vllm_config.model_path)
        logger.info("VLLM AsyncLLM engine initialization completed successfully")

    async def chat_completions(self, raw_request: Dict[str, Any]):
        """OpenAI-compatible HTTP endpoint.
        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
        """
        if "model_name" in raw_request:
            raw_request.pop("model_name")
        try:
            CompletionRequestChecker.validate_chat_input(raw_request)
        except ValueError as e:
            raise ValueError(f"Input validation failed: {e}") from e
        if self.openai_serving_chat is None:
            raise RuntimeError("OpenAI serving chat is not initialized. Call init_engine first.")
        try:
            request = ChatCompletionRequest(**raw_request)
        except ValueError as e:
            raise ValueError(f"Failed to parse chat completion request: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error occurred when parsing chat completion: {e}") from e

        generator = await self.openai_serving_chat.create_chat_completion(request)
        return generator.model_dump()

    async def completions(self, raw_request: Dict[str, Any]):
        """OpenAI completions API.

        Args:
            raw_request: Dictionary containing completion request parameters

        Returns:
            Dictionary containing completion response

        Raises:
            ValueError: If input validation fails
            RuntimeError: If engine is not initialized or generation fails
        """
        if "model_name" in raw_request:
            raw_request.pop("model_name")
        try:
            CompletionRequestChecker.validate_input(raw_request)
        except ValueError as e:
            raise ValueError(f"Input validation failed: {e}") from e

        if self.openai_serving_completion is None:
            raise RuntimeError("OpenAI serving completion not initialized. Call init_engine first.")

        try:
            request = CompletionRequest(**raw_request)
        except ValueError as e:
            raise ValueError(f"Failed to parse completion request: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error occurred when parsing completion request: {e}") from e

        generator = await self.openai_serving_completion.create_completion(request)
        return generator.model_dump()

    async def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """Wake up the engine to load model weights and build KV cache.
        """
        if tags is not None and not isinstance(tags, list):
            raise ValueError("tags must be a list")

        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init_engine first.")

        await self.engine.wake_up(tags)
        logger.info(f"Engine woke up successfully with tags: {tags}")

    async def sleep(self) -> None:
        """Sleep the engine to offload model weights and discard KV cache."""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call init_engine first.")

        try:
            await self.engine.sleep()
            logger.info("Engine sleep completed successfully")
        except ValueError as e:
            raise RuntimeError(f"Failed to put engine to sleep: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred when putting engine to sleep: {e}") from e

    def _build_vllm_config(self, config: GenConfig, tokenizer_name_or_path: str) -> VLLMConfig:
        """Build VLLM configuration from raw config."""
        try:
            return VLLMConfig(
                model_path=tokenizer_name_or_path,
                tensor_parallel_size=config.infer_tensor_parallel_size,
                pipeline_parallel_size=config.infer_pipeline_parallel_size,
                max_model_len=config.max_model_len,
                max_num_batched_tokens=config.max_num_batched_tokens,
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                load_format=self.agentic_rl_config.load_format,
                enable_sleep_mode=self.agentic_rl_config.enable_sleep_mode,
                enable_prefix_caching=config.enable_prefix_caching,
                trust_remote_code=config.trust_remote_code,
                max_num_seqs=config.max_num_seqs,
                sampling_config=config.sampling_config
            )
        except AttributeError as e:
            raise AttributeError(f"Missing required config field while building VLLM config: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred when building VLLM configuration: {e}") from e

    def _build_engine_args(self, vllm_config: VLLMConfig, generation_config: Dict[str, Any]) -> AsyncEngineArgs:
        """Build AsyncEngineArgs from configuration."""
        load_format = "dummy" if vllm_config.load_format == "megatron" else vllm_config.load_format

        # Log warning for large batch sizes
        if vllm_config.max_num_batched_tokens > MAX_BATCHED_TOKENS_WARNING_THRESHOLD:
            logger.warning(
                f"max_num_batched_tokens={vllm_config.max_num_batched_tokens} exceeds "
                f"{MAX_BATCHED_TOKENS_WARNING_THRESHOLD}, which may cause errors for 'chunked prefill'!"
            )

        return AsyncEngineArgs(
            model=vllm_config.model_path,
            enable_sleep_mode=vllm_config.enable_sleep_mode,
            override_generation_config=generation_config,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.pipeline_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=vllm_config.dtype,
            enforce_eager=vllm_config.enforce_eager,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=vllm_config.max_model_len,
            load_format=load_format,
            max_num_batched_tokens=vllm_config.max_num_batched_tokens,
            enable_prefix_caching=vllm_config.enable_prefix_caching,
            trust_remote_code=vllm_config.trust_remote_code,
            seed=self.server_config.vllm_dp_rank,
            max_num_seqs=vllm_config.max_num_seqs,
            hf_overrides={"max_position_embeddings": vllm_config.max_model_len},
        )

    def _setup_openai_serving(self, model_config, model_name: str, model_path: str) -> None:
        """Setup OpenAI serving components."""
        try:
            base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]
            models = OpenAIServingModels(self.engine, model_config, base_model_paths)

            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models,
                "assistant",
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
                enable_auto_tools=True,
                tool_parser="pythonic"
            )

            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine,
                model_config,
                models,
                request_logger=None,
                return_tokens_as_token_ids=True,
            )
            logger.info("OpenAI serving components initialized successfully")
        except TypeError as e:
            raise RuntimeError(f"Failed to setup OpenAI serving components: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected Error: failed to setup openai serving: {e}") from e
