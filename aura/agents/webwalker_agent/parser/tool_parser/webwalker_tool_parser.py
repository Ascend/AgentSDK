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
import json
import logging
import re
from enum import Enum
from typing import Any

from rllm.parser.tool_parser.tool_parser_base import ToolParser

from agents.webwalker_agent.constants import (
    WEBWALKER_ERROR_TOOL_NAME,
    WEBWALKER_PARSE_TOOL_ERROR,
)
from agents.webwalker_agent.webwalker_tools import webwalker_tools_v1

logger = logging.getLogger(__name__)


class WebWalkerKeyword(Enum):
    PARSE_DONE = "done"
    PARSE_TOOL_ERROR = WEBWALKER_PARSE_TOOL_ERROR


VALID_WEBWALKER_TOOLS = frozenset(webwalker_tools_v1.keys())


class WebWalkerToolParser(ToolParser):
    """Parse WebWalker ReAct-format tool calls.

    Expected format:
    Thought: ...
    Action: visit_page
    Action Input: {"button": "..."}
    """

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        # Ignore CoT-style think blocks so mentions like "Action: visit_page ..."
        # inside reasoning are not mistaken for real tool calls.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _clean_action_input(action_input_str: str) -> str:
        if action_input_str.startswith("```json"):
            action_input_str = action_input_str[7:]
        elif action_input_str.startswith("```"):
            action_input_str = action_input_str[3:]
        if action_input_str.endswith("```"):
            action_input_str = action_input_str[:-3]
        return action_input_str.strip()

    def _parse_last_react_call(self, text: str) -> tuple[str | None, str | None]:
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'

        normalized = "\n" + text.strip()
        i = normalized.rfind(special_func_token)
        j = normalized.rfind(special_args_token)
        k = normalized.rfind(special_obs_token)

        if not (0 <= i < j):
            return None, None

        if k < j:
            k = len(normalized)

        tool_name = normalized[i + len(special_func_token):j].strip()
        action_input_str = normalized[j + len(special_args_token):k].strip()
        return tool_name, action_input_str

    def _parse_direct_json_call(self, text: str) -> tuple[str | None, dict[str, Any] | None]:
        stripped = text.strip()
        if not stripped:
            return None, None

        candidate = None
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
        elif stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped

        if not candidate:
            return None, None

        try:
            parsed_data = json.loads(candidate)
        except json.JSONDecodeError:
            return None, None
        if not isinstance(parsed_data, dict):
            return None, None

        tool_name = parsed_data.get("action", parsed_data.get("name", ""))
        if not tool_name:
            return None, None

        if "parameters" in parsed_data:
            arguments = parsed_data["parameters"]
        elif "button" in parsed_data:
            arguments = {"button": parsed_data["button"]}
        elif "arguments" in parsed_data:
            arguments = parsed_data["arguments"]
        else:
            arguments = {}

        return tool_name, arguments

    @staticmethod
    def _build_error_tool_call() -> dict[str, Any]:
        return {"name": WEBWALKER_ERROR_TOOL_NAME, "arguments": {"response": WebWalkerKeyword.PARSE_TOOL_ERROR.value}}

    def parse(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        try:
            sanitized_text = self._strip_think_blocks(text)

            tool_name, arguments = self._parse_direct_json_call(sanitized_text)
            if tool_name:
                if tool_name in VALID_WEBWALKER_TOOLS:
                    tool_calls.append({"name": tool_name, "arguments": arguments or {}})
                    return tool_calls
                tool_calls.append(self._build_error_tool_call())
                return tool_calls

            tool_name, action_input_str = self._parse_last_react_call(sanitized_text)

            if tool_name and action_input_str is not None:
                action_input_str = self._clean_action_input(action_input_str)

                try:
                    query = json.loads(action_input_str)
                except json.JSONDecodeError:
                    tool_calls.append(self._build_error_tool_call())
                    return tool_calls
                if not isinstance(query, dict):
                    tool_calls.append(self._build_error_tool_call())
                    return tool_calls

                if tool_name in VALID_WEBWALKER_TOOLS:
                    tool_calls.append({"name": tool_name, "arguments": query})
                else:
                    tool_calls.append(self._build_error_tool_call())
            else:
                tool_calls.append(self._build_error_tool_call())

            return tool_calls

        except (json.JSONDecodeError, UnicodeError, KeyError) as e:
            logger.warning(f"Failed to parse WebWalker tool call: {e}")
            tool_calls.append(self._build_error_tool_call())
            return tool_calls

    def get_tool_prompt(self, tools_schema: str):
        return f"Available Tools:\n{tools_schema}"
