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


# Standard library imports
import os
import random

# Third-party library imports
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Internal imports
from agentic_rl.base.execution.executor import public_api, Executor 
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.infer_executor import InferExecutor

logger = Loggers(__name__).get_logger()

class InferPrefillExecutor(InferExecutor):
    def __init__(self, engine, engine_kwargs, *args, **kwargs):
        enforce_eager = engine_kwargs.get("enforce_eager", True)
        kv_transfer_config = engine_kwargs.get("kv_transfer_config", None)
        if kv_transfer_config is None:
            kv_transfer_config = {"kv_connector": "LLMDataDistCMgrConnector",
                                  "kv_buffer_device": "npu",
                                  "kv_role": "kv_producer",
                                  "kv_parallel_size": 1,
                                  "kv_port": "20001",
                                  "engine_id": "0",
                                  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
                                 }
        engine_kwargs["kv_transfer_config"] = kv_transfer_config
        engine_kwargs["enforce_eager"] = enforce_eager

        super().__init__(engine, engine_kwargs, *args, **kwargs)

    @public_api(name="chat_completions")
    async def chat_completions(self, *args, **kwargs):
        logger.info(f"args: {args}\nkwargs: {kwargs}")
        request_data = kwargs.get("request_data", None)
        if request_data is None:
            raise ValueError("Request data is None")

        req_data = request_data.copy()
        req_data['kv_transfer_params'] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None
        }
        req_data["stream"] = False
        req_data["max_tokens"] = 1
        if "max_completion_tokens" in req_data:
            req_data["max_completion_tokens"] = 1
        if "stream_options" in req_data:
            del req_data["stream_options"]

        new_kwargs = kwargs.copy()
        new_kwargs["request_data"] = req_data
        response = await self.engine.chat_completions(*args, **new_kwargs)

        return response

    @public_api(name="stream_chat_completions", is_stream=True)
    async def stream_chat_completions(self, *args, **kwargs):
        raise NotImplementedError("Prefill instances do not support stream chat.")


class InferDecodeExecutor(InferExecutor):
    def __init__(self, engine, engine_kwargs, *args, **kwargs):
        enforce_eager = engine_kwargs.get("enforce_eager", True)
        kv_transfer_config = engine_kwargs.get("kv_transfer_config", None)
        if kv_transfer_config is None:
            kv_transfer_config = {"kv_connector": "LLMDataDistCMgrConnector",
                                  "kv_buffer_device": "npu",
                                  "kv_role": "kv_consumer",
                                  "kv_parallel_size": 1,
                                  "kv_port": "20001",
                                  "engine_id": "0",
                                  "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
                                  }
        engine_kwargs["kv_transfer_config"] = kv_transfer_config
        engine_kwargs["enforce_eager"] = enforce_eager

        super().__init__(engine, engine_kwargs, *args, **kwargs)

class InferPDSepExecutor(Executor):
    def __init__(self, engine, engine_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"engine_kwargs: {engine_kwargs}\nkwargs: {kwargs}")
        self.executors = {"prefill": [], "decode": []}
        self.engine_name = engine
        self.engine_kwargs = engine_kwargs
        self.args = args
        self.kwargs = kwargs

    async def setup(self):
        selected_nodes = self.alloc_resources_from_ranktable()
        p_num = self.kwargs.get("p_num", 0)
        d_num = self.kwargs.get("d_num", 0)
        for p_node_index in range(p_num):
            node_id = selected_nodes["prefill"][p_node_index]["node_id"]
            p_executor = await self.create_single_infer_executor("prefill", p_node_index, node_id)
            self.executors["prefill"].append(p_executor)
        for d_node_index in range(d_num):
            node_id = selected_nodes["decode"][d_node_index]["node_id"]
            d_executor = await self.create_single_infer_executor("decode", d_node_index, node_id)
            self.executors["decode"].append(d_executor)

        logger.info("Create infer executor success")

    async def create_single_infer_executor(self, pd_role, resource_index, node_id):
        # get current resource set
        curr_resource_set = self.resource_set.model_copy()
        curr_resource_set.info = self.resource_set.info[resource_index: resource_index+1]
        curr_resource_set.bundles = self.resource_set.bundles[resource_index: resource_index+1]
        executor_class = InferPrefillExecutor if pd_role == "prefill" else InferDecodeExecutor

        # create executor
        scheduling_strategy = NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False
        )
        infer_executor_ref = ray.remote(executor_class).options(
            scheduling_strategy=scheduling_strategy,
        ).remote(resource_set=curr_resource_set, engine=self.engine_name, engine_kwargs=self.engine_kwargs)
        await infer_executor_ref.setup.remote()

        return infer_executor_ref
    
    # todo: 先手动生成ranktable，后续考虑集成到代码中，一键启动
    # Get ranktable
    def get_ranktable(self):
        import json
        file_path = os.getenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", None)
        if file_path is None:
            raise ValueError("Can't find ranktable file, please set DISAGGREGATED_PREFILL_RANK_TABLE_PATH")
        with open(file_path, "r") as f:
            ranktable = json.load(f)
        return ranktable
    
    # Get which nodes PD are deployed on from ranktable
    def alloc_resources_from_ranktable(self):
        ranktable = self.get_ranktable()
        prefill_set = set()
        decode_set = set()
        for prefill_item in ranktable["prefill_device_list"]:
            prefill_set.add(prefill_item["server_id"])
        for decode_item in ranktable["decode_device_list"]:
            decode_set.add(decode_item["server_id"])

        selected_nodes = {"prefill": [], "decode": []}
        node_info = self.get_node_info()
        for node in node_info:
            if node["node_ip"] in prefill_set:
                selected_nodes["prefill"].append(node)
                prefill_set.remove(node["node_ip"])
            elif node["node_ip"] in decode_set:
                selected_nodes["decode"].append(node)
                decode_set.remove(node["node_ip"])

        if prefill_set:
            raise ValueError("prefill set is not empty, there are insufficient nodes for deploying the prefill.")
        if decode_set:
            raise ValueError("decode set is not empty, there are insufficient nodes for deploying the decode.")
        return selected_nodes
    
    # Resource allocation, kept for now
    # allocate resources for P/D instances
    def alloc_resources(self):
        from agentic_rl.base.conf.conf import AgenticRLConf
        conf = AgenticRLConf.load_config()
        num_npus_per_node = 8 if os.getenv("ASCEND_PLATFORM", "A2") == "A2" else 16
        node_info = self.get_node_info()
        print(f"node_info: {node_info}")
        selected_nodes = {"prefill": [], "decode": []}
        for instance_conf in self.conf.infer_pd_instances:
            executor_num = instance_conf.executor_num
            total_npus = int(instance_conf.resource_info[0]["NPU"])
            alloc_nodes = int(executor_num * (total_npus / num_npus_per_node))
            if alloc_nodes > len(node_info):
                raise ValueError(f"Resources are insufficient. executor {instance_conf.role} requires {alloc_nodes} nodes, but the total num of nodes is {len(node_info)}")
            selected_nodes[instance_conf.role] = node_info[:alloc_nodes]
            node_info = node_info[alloc_nodes:]

        return selected_nodes
    
    # get node info in ray
    def get_node_info(self):
        node_info_list = [] # {node_ip: resources}
        nodes = ray.nodes()
        for node in nodes:
            node_info_dict = {}
            node_info_dict["node_id"] = node["NodeID"]
            node_info_dict["node_ip"] = node["NodeManagerAddress"]
            node_info_dict["resources"] = node["Resources"]
            node_info_list.append(node_info_dict)
        
        return node_info_list


    @public_api(name="chat_completions")
    async def chat_completions(self, *args, **kwargs):
        prefill_executor = random.choice(self.executors["prefill"])
        decode_executor = random.choice(self.executors["decode"])
        prefill_response = await prefill_executor.chat_completions.remote(*args, **kwargs)
        kv_transfer_params = prefill_response.get('kv_transfer_params', {})
        request_data = kwargs.get("request_data", None)
        if kv_transfer_params:
            request_data["kv_transfer_params"] = kv_transfer_params
        logger.info(f"kv_transfer_params: {kv_transfer_params}")

        return await decode_executor.chat_completions.remote(*args, **kwargs)

    @public_api(name="stream_chat_completions", is_stream=True)
    async def stream_chat_completions(self, *args, **kwargs):
        prefill_executor = random.choice(self.executors["prefill"])
        decode_executor = random.choice(self.executors["decode"])
        prefill_response = await prefill_executor.chat_completions.remote(*args, **kwargs)
        kv_transfer_params = prefill_response.get('kv_transfer_params', {})
        request_data = kwargs.get("request_data", None)

        if kv_transfer_params:
            request_data["kv_transfer_params"] = kv_transfer_params
        logger.info(f"kv_transfer_params: {kv_transfer_params}")

        async for chat_response in decode_executor.stream_chat_completions.remote(*args, **kwargs):
            chat_response = await chat_response
            yield chat_response

    @public_api(name="wake_up")
    async def wake_up(self, *args, **kwargs):
        for pd_role, server_engines in self.executors.items():
            for engine in server_engines:
                await engine.wake_up(*args, **kwargs)

    @public_api(name="sleep")
    async def sleep(self, *args, **kwargs):
        for pd_role, server_engines in self.executors.items():
            for engine in server_engines:
                await engine.sleep(*args, **kwargs)