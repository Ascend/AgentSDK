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
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypeVar

# Third-party library imports
import cloudpickle
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
from agentic_rl.runner.infer_adapter.vllm.vllm_async_server import AsyncVLLMServer, ExternalRayDistributedExecutor
from agentic_rl.runner.scheduler.load_stat import WorkloadStatLogger, vllm_log_stats_periodically
from agentic_rl.runner.scheduler.workload import InstanceWorkLoad

_R = TypeVar("_R", default=Any)
logger = Loggers("vllm_server").get_logger()


class AsyncVLLMServerPDSep(AsyncVLLMServer):
    """
    AsyncVLLMServerPDSep is a wrapper for AsyncLLM in prefill-decode seperate mode, it uses
    ExternalRayDistributedExecutor to launch engines in hybrid rollout workers, i.e. AsyncActorRolloutRefWorker.

    AsyncVLLMServerPDSep works as follows:
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
                 wg_prefix: str,
                 infer_mode=None):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__(config, tokenizer_name_or_path, vllm_dp_size, vllm_dp_rank, wg_prefix)
        self.update_config(infer_mode)
        # self.infer_mode = infer_mode

    # get node ips from ranktable
    def get_node_ip_from_ranktable(self):
        import json
        file_path = os.getenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", None)
        if file_path is None:
            raise ValueError("Can't find ranktable file, please set DISAGGREGATED_PREFILL_RANK_TABLE_PATH")
        with open(file_path, "r") as f:
            ranktable = json.load(f)

        prefill_set = set()
        decode_set = set()
        for prefill_item in ranktable["prefill_device_list"]:
            prefill_set.add(prefill_item["server_id"])
        for decode_item in ranktable["decode_device_list"]:
            decode_set.add(decode_item["server_id"])

        return {"prefill":prefill_set, "decode":decode_set}

    def update_prefill_params(self):
        if self.infer_mode == "prefill":
            if self.config.prefill_enforce_eager is not None:
                logger.info(f"Change enforce_eager from {self.config.enforce_eager} to {self.config.prefill_enforce_eager} for prefill instances")
                self.config.enforce_eager = self.config.prefill_enforce_eager

            if self.config.prefill_max_num_seqs is not None:
                logger.info(f"Change max_num_seqs from {self.config.max_num_seqs} to {self.config.prefill_max_num_seqs} for prefill instances")
                self.config.max_num_seqs = self.config.prefill_max_num_seqs

            if self.config.prefill_max_num_batched_tokens is not None:
                logger.info(f"Change max_num_batched_tokens from {self.config.max_num_batched_tokens} to {self.config.prefill_max_num_batched_tokens} for prefill instances")
                self.config.max_num_batched_tokens = self.config.prefill_max_num_batched_tokens

            if self.config.prefill_gpu_memory_utilization is not None:
                logger.info(f"Change gpu_memory_utilization from {self.config.gpu_memory_utilization} to {self.config.prefill_gpu_memory_utilization} for prefill instances")
                self.config.gpu_memory_utilization = self.config.prefill_gpu_memory_utilization

            if self.config.prefill_max_model_len is not None:
                logger.info(f"Change max_model_len from {self.config.max_model_len} to {self.config.prefill_max_model_len} for prefill instances")
                self.config.max_model_len = self.config.prefill_max_model_len


    # TODO: update config according to ranktable
    def update_config(self, infer_mode):
        if hasattr(self.config, 'kv_transfer_config') and \
            getattr(self.config, 'kv_transfer_config', None) is not None:
            return
        node_ips = self.get_node_ip_from_ranktable()
        curr_node_ip = ray.util.get_node_ip_address()

        if infer_mode is None:
            if curr_node_ip in node_ips["prefill"]:
                self.infer_mode = "prefill"
            elif curr_node_ip in node_ips["decode"]:
                self.infer_mode = "decode"
            else:
                raise ValueError(f"current node ip {curr_node_ip} is not found in ranktable")
        else:
            self.infer_mode = infer_mode

        logger.info(f"begin to setup {self.infer_mode} instance")
        # TODO: kv_port and engine_id may conflict?
        default_kv_transfer_config = {
            "kv_connector": "LLMDataDistCMgrConnector",
            "kv_buffer_device": "npu",
            "kv_role": "",
            "kv_parallel_size": 1,
            "kv_port": "20001",
            "engine_id": "0",
            "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
        }
        if self.infer_mode == "prefill":
            default_kv_transfer_config["kv_role"] = "kv_producer"
            self.update_prefill_params()
        else:
            default_kv_transfer_config["kv_role"] = "kv_consumer"

        setattr(self.config, 'kv_transfer_config', default_kv_transfer_config)

    async def get_infer_mode(self) -> str:
        await self.server_ready.wait()
        return self.infer_mode


    # Only update config and add kv_transfer_config to engine_args for PD Seperate mode.
    # The main implementation is consistent with the original function.
    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
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
        logger.info(f"max_num_batched_tokens={max_num_batched_tokens}，"
              f"[attention] this num > 8k may cause error for 'chunked prefill' !")
        cudagraph_capture_sizes = []
        if config.cudagraph_capture_sizes:
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
            worker_extension_cls="agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions",
            kv_transfer_config=config.kv_transfer_config # add kv transfer config
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

    async def update_request_for_prefill(self, request_json:dict):
        request_json['kv_transfer_params'] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None
        }
        request_json["stream"] = False
        request_json["max_tokens"] = 1
        if "max_completion_tokens" in request_json:
            request_json["max_completion_tokens"] = 1
        if "stream_options" in request_json:
            del request_json["stream_options"]

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        logger.debug(f"\n\nraw_request: {raw_request}\n\n")
        # update request for PD seperate mode
        if self.infer_mode == "prefill":
            await self.update_request_for_prefill(request_json)

        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        logger.debug(f"chat generator: {generator}")

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            if not isinstance(generator, ChatCompletionResponse):
                raise TypeError("Generator should be a ChatCompletionResponse for non-streaming requests")
            return JSONResponse(content=generator.model_dump())

    async def completions(self, raw_request: Request):
        """OpenAI completions API.

        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        request_json = await raw_request.json()
        # update request for PD seperate mode
        if self.infer_mode == "prefill":
            await self.update_request_for_prefill(request_json)

        request = CompletionRequest(**request_json)
        generator = await self.openai_serving_completion.create_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            if not isinstance(generator, CompletionResponse):
                raise TypeError("Generator should be a CompletionResponse for non-streaming requests")
            if generator.choices and generator.choices[0].logprobs:
                generator.choices[0].logprobs.token_logprobs = []
                generator.choices[0].logprobs.top_logprobs = []
                generator.choices[0].logprobs.text_offset = []
            return JSONResponse(content=generator.model_dump())


    async def get_workload(self, raw_request: Request):
        return JSONResponse(content=json.dumps(self.ins_workload.to_dict()), status_code=200)

    async def cancel_requests(self, raw_request: Request):
        """cancel requests from AsyncVLLMServer"""
        request_json = await raw_request.json()
        await self.engine.abort(request_json["requests"])
