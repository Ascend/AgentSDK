#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from clawcodex_ext.query.outbox_types import GenericOutboxEvent

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..protocol import ToolResult


def _structured_output_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    context.outbox.append(GenericOutboxEvent.from_dict({"tool": "StructuredOutput", "structured_output": tool_input}))
    return ToolResult(
        name="StructuredOutput",
        output={
            "data": "Structured output provided successfully",
            "structured_output": tool_input,
        },
    )


StructuredOutputTool: Tool = build_tool(
    name="StructuredOutput",
    input_schema={"type": "object", "additionalProperties": True},
    call=_structured_output_call,
    prompt="Return a final response as structured JSON.",
    description="Return a final response as structured JSON.",
    max_result_size_chars=100_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
)
