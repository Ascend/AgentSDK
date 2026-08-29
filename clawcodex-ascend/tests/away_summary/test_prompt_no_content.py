#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

# pylint: disable=no-name-in-module

from src.agent.conversation import Conversation
from src.types.messages import Message

from clawcodex_ext.away_summary.prompt import build_summary_messages
from clawcodex_ext.types.content_blocks import TextBlock, ToolUseBlock


def test_summary_prompt_omits_no_content_placeholder_but_keeps_tool_call() -> None:
    conversation = Conversation()
    conversation.messages = [
        Message(role="user", content="修改文档"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="[No content]"),
                ToolUseBlock(id="edit-1", name="Edit", input={"path": "report.md"}),
            ],
        ),
    ]

    messages = build_summary_messages(conversation, max_input_tokens=4_000)
    prompt = "\n".join(m["content"] for m in messages)

    assert "[No content]" not in prompt
    assert "tool_use Edit" in prompt
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_summary_prompt_keeps_object_tool_result_content() -> None:
    class ToolResult:
        type = "tool_result"
        content = "updated report.md"

    conversation = Conversation()
    conversation.messages = [
        Message(role="user", content="修改文档"),
        Message(role="assistant", content=[ToolResult()]),
    ]

    prompt = "\n".join(
        message["content"]
        for message in build_summary_messages(
            conversation,
            max_input_tokens=4_000,
        )
    )

    assert "[tool_result updated report.md]" in prompt
