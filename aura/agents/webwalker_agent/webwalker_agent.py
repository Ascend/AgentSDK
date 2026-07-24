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
import copy
import json
import logging
import re
import uuid
from typing import Any

from rllm.tools.multi_tool import MultiTool

from agents.webwalker_agent.parser import get_tool_parser
from agents.webwalker_agent.prompt.prompts import SYSTEM_EXPLORER as SYSTEM_EXPLORER_TEMPLATE
from agents.webwalker_agent.webwalker_tools import webwalker_tools
from aura.memory.memory_summary import MemorySummary
from aura.runner.agent_engine_wrapper.base.agent.base_agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

SYSTEM_EXPLORER = SYSTEM_EXPLORER_TEMPLATE.format(
    tool_descs=json.dumps(webwalker_tools, ensure_ascii=False),
    tool_names="visit_page",
    # Chain/beam training provides the actual query in the user message.
    query="The actual question will be provided in the user message.",
)


class WebWalkerAgent(BaseAgent):
    def __init__(
        self,
        system_prompt=SYSTEM_EXPLORER,
        parser_name="webwalker",
        tokenizer_path: str = None,
        max_prompt_length: int = 8192,
        chat_model_name: str = "qwen3_4b",
        use_summary: bool = False,
        tools: list[str] | None = None,
        tool_map: dict | None = None,
        **kwargs
    ):
        logger.info("[WebWalkerAgent] Initializing")

        self.system_prompt = system_prompt
        self.tokenizer_path = tokenizer_path
        self.max_prompt_length = max_prompt_length
        self.chat_model_name = chat_model_name
        self.use_summary = use_summary
        self.tools = MultiTool(tools=[])

        parser_class = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        self._trajectory = Trajectory()

        try:
            if tokenizer_path:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            else:
                logger.warning("[WebWalkerAgent] tokenizer_path is empty; prompt length accounting may be disabled")
                tokenizer = None
        except Exception as e:
            logger.warning("[WebWalkerAgent] failed to load tokenizer: %s", e)
            tokenizer = None
        self.tokenizer = tokenizer
        memory_config = {
            "use_summary": use_summary,
            "max_prompt_length": max_prompt_length,
            "chat_model_name": chat_model_name,
            "train_model_tokenizer_path": tokenizer_path
        }

        try:
            self.memory = MemorySummary(config=memory_config, tokenizer=tokenizer)
        except TypeError as e:
            logger.warning("[WebWalkerAgent] MemorySummary rejected tokenizer argument: %s", e)
            self.memory = MemorySummary(config=memory_config)
            self.memory.tokenizer = tokenizer

        self.current_observation = None

        self.reset()
        logger.info("[WebWalkerAgent] Initialized")


    def _get_tokenizer(self):
        tokenizer = getattr(self.memory, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.tokenizer
        return tokenizer

    def _truncate_text_for_prompt(self, text: Any, max_tokens: int, field_name: str) -> Any:
        if not isinstance(text, str) or not text:
            return text

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return text

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text

        marker_ids = tokenizer.encode("\n\n[... content truncated for prompt budget ...]\n\n", add_special_tokens=False)
        marker_len = len(marker_ids)
        if max_tokens <= marker_len + 32:
            truncated_ids = token_ids[:max_tokens]
        else:
            head_budget = int((max_tokens - marker_len) * 0.7)
            tail_budget = max_tokens - marker_len - head_budget
            truncated_ids = token_ids[:head_budget] + marker_ids + token_ids[-tail_budget:]

        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        logger.info(
            "[WebWalkerAgent] truncated %s from %s to %s tokens for prompt budget",
            field_name,
            len(token_ids),
            len(truncated_ids),
        )
        return truncated_text

    def _truncate_observation_payload(self, text: Any, field_name: str) -> Any:
        budget = max(512, int(self.max_prompt_length * 0.35))
        return self._truncate_text_for_prompt(text, budget, field_name)

    @staticmethod
    def _dedupe_keep_order(items: list[Any]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _build_compact_observation(
        self,
        raw_observation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        field_name: str,
    ) -> str:
        observation = str(raw_observation or "").strip()
        metadata = metadata or {}

        buttons_match = re.search(
            r"(clickable button:\s*\n\n.*?Each button is wrapped in a <button> tag)",
            observation,
            flags=re.IGNORECASE | re.DOTALL,
        )
        buttons_block = buttons_match.group(1).strip() if buttons_match else ""

        system_note_match = re.search(r"(System Note:.*)$", observation, flags=re.DOTALL)
        system_note = system_note_match.group(1).strip() if system_note_match else ""

        page_body = observation
        if buttons_match:
            page_body = page_body.replace(buttons_match.group(1), "")
        if system_note_match:
            page_body = page_body.replace(system_note_match.group(1), "")
        page_body = page_body.strip()

        useful_info = self._dedupe_keep_order(metadata.get("critic_useful_information") or [])
        memory_snapshot = self._dedupe_keep_order(metadata.get("webwalker_memory_snapshot") or [])
        if memory_snapshot:
            memory_snapshot = memory_snapshot[-4:]
        sections: list[str] = []
        if useful_info:
            useful_text = "useful extracted information:\n\n" + "\n".join(useful_info)
            sections.append(self._truncate_observation_payload(useful_text, f"{field_name}:useful_info"))
        elif page_body:
            sections.append(self._truncate_observation_payload(page_body, field_name))

        if memory_snapshot:
            memory_text = "accumulated useful information:\n\n" + "\n".join(memory_snapshot)
            sections.append(self._truncate_observation_payload(memory_text, f"{field_name}:memory"))

        if system_note:
            sections.append(self._truncate_observation_payload(system_note, f"{field_name}:system_note"))

        if buttons_block:
            sections.append(self._truncate_observation_payload(buttons_block, f"{field_name}:buttons"))

        compact = "\n\n".join(section for section in sections if section).strip()
        return compact or str(self._truncate_observation_payload(observation, field_name) or "")

    def _format_observation_as_messages(self, obs: Any, info: dict | None = None) -> list[dict]:
        metadata = info.get("metadata", {}) if isinstance(info, dict) else {}

        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                content = f"Question: {obs['question']}"
                if "root_url" in obs:
                    content += f"\nURL: {obs['root_url']}"
                if "initial_observation" in obs:
                    initial_observation = self._build_compact_observation(
                        obs["initial_observation"],
                        metadata,
                        field_name="initial_observation",
                    )
                    content += f"\n\nObservation: {initial_observation}"
                else:
                    content += "\n\nObservation: Please start your exploration based on the URL."

                messages.append({"role": "user", "content": content})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    tool_output_str = self._build_compact_observation(
                        tool_output_str,
                        metadata,
                        field_name=f"tool_output:{tool_call_id}",
                    )
                    messages.append({"role": "tool", "content": tool_output_str, "tool_call_id": tool_call_id})

        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        obs_messages = self._format_observation_as_messages(observation, info=info)
        self.memory.add_message(obs_messages, metadata=[{"reward": reward}])

        self.current_observation = observation


    def update_from_model(self, response: str, **kwargs) -> Action:
        assistant_content = response

        tool_calls = self.tool_parser.parse(response)

        tool_calls_dict = [
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": tool_call if isinstance(tool_call, dict) else tool_call.to_dict(),
            }
            for tool_call in tool_calls
        ]

        if not tool_calls_dict:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "error_tool",
                        "arguments": {"response": "tool parsing failed"},
                    },
                }
            ]
        assistant_message = {"role": "assistant", "content": assistant_content}
        self.memory.add_message(assistant_message)

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=assistant_content,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        action_result = Action(action=tool_calls_dict)
        return action_result

    def reset(self):
        self._trajectory = Trajectory()

        tools_prompt = str(webwalker_tools)
        self.memory.clear_memory("system", self.system_prompt + "\n\nAvailable Tools:\n" + tools_prompt)


    def snapshot(self):
        """Deep-copy agent state for branch expansion."""
        cfg = getattr(self.memory, "config", None)
        t_path = getattr(cfg, "train_model_tokenizer_path", None) if cfg is not None else None
        if t_path is None:
            t_path = self.tokenizer_path

        _mpl = getattr(cfg, "max_prompt_length", 8192) if cfg is not None else 8192
        new_agent = WebWalkerAgent(
            system_prompt=self.system_prompt,
            parser_name="webwalker",
            tokenizer_path=t_path,
            max_prompt_length=_mpl,
            chat_model_name=getattr(cfg, "chat_model_name", self.chat_model_name),
            use_summary=getattr(cfg, "use_summary", self.use_summary)
        )

        new_agent._trajectory = copy.deepcopy(self._trajectory)

        if hasattr(self.memory, "messages"):
            new_agent.memory.messages = copy.deepcopy(self.memory.messages)
        elif hasattr(self.memory, "_messages"):
            new_agent.memory._messages = copy.deepcopy(self.memory._messages)

        new_agent.current_observation = copy.deepcopy(self.current_observation)

        if hasattr(self.memory, "tokenizer") and self.memory.tokenizer is not None:
            new_agent.memory.tokenizer = self.memory.tokenizer
            new_agent.tokenizer = self.memory.tokenizer

        return new_agent

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.memory.get_prompt_messages()

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
