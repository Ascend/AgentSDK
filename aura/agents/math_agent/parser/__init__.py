#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
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
from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
from rllm.parser.tool_parser.tool_parser_base import ToolParser

PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    if parser_name not in PARSER_REGISTRY:
        raise ValueError(f"Tool parser {parser_name} not found in {PARSER_REGISTRY}")
    return PARSER_REGISTRY[parser_name]


__all__ = [
    "R1ToolParser",
    "QwenToolParser",
    "ToolParser",
    "get_tool_parser",
    "PARSER_REGISTRY",
]
