#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

from concurrent.futures import ThreadPoolExecutor, as_completed

from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool
from smolagents.agents import RunResult

from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper


class SmolAgentWrapper(BaseEngineWrapper):
    def __init__(
            self,
            server_addresses=None,
            tools=None,
            max_model_len=16384,
            tokenizer_name_or_path=None,
            max_steps=20,
            **kwargs
    ):
        super().__init__(
            server_addresses=server_addresses,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_model_len=max_model_len,
            max_steps=max_steps,
            **kwargs
        )

        model_id = f"hosted_vllm/{tokenizer_name_or_path}"
        model = LiteLLMModel(
            model_id=model_id,
            api_base=server_addresses,
            api_key="EMPTY",
            num_ctx=max_model_len,
        )

        self.agent_args = {
            "model": model,
            "tools": tools,
            "return_full_result": True,
            "max_steps": max_steps
        }

    def init_envs_and_agents(self, tasks):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """

        def _create_agent(i):
            return i, self.agent_class(**self.agent_args)

        agents = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(tasks))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent

        self.agents = agents
        self.n_parallel_agents = len(agents)
        return agents

    def generate_agent_trajectories_async(self, tasks, timing_raw=None, meta_info=None, mode="Token"):
        self.init_envs_and_agents(tasks)

        completions = []
        for i, task in enumerate(tasks):
            query = task["problem"]
            agent = self.agents[i]
            res = agent.run(query)
            if isinstance(res, RunResult):
                last_trace = res.messages[-1]
                for chatmessage in last_trace["model_input_messages"]:
                    message = {
                        "role": chatmessage.role,
                        "content": chatmessage.content,
                        "tool_calls": chatmessage.tool_calls,
                        "raw": chatmessage.raw
                    }
                    completions.append(message)

                completions.append(last_trace["model_output_message"])

        return {
            "chat_completions": completions
        }