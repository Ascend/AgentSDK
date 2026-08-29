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

"""Conversation management for Claw Codex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clawcodex_ext.types.content_blocks import (
    ToolResultBlock,
    normalize_content_blocks,
)
from clawcodex_ext.types.messages import (
    Message,
    MessageContent,
    create_message,
    message_from_dict,
    message_to_dict,
    normalize_messages_for_api,
)


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    # Bumped from 100 to 2000 (critic ch05/F-consolidation S2).
    # The headless+TUI cutover routes tool_use AND tool_result blocks
    # through ``add_message`` separately (multiple messages per agent
    # turn), so a 30-turn run with ~3 tools/turn produces ~120 messages
    # per single user prompt — well past the old 100-cap. The cap
    # silently truncates from the head, which removes the ORIGINAL
    # user prompt; subsequent API calls go out without the task
    # description and the model degrades. 2000 is enough to cover a
    # long agentic session without changing the truncation semantics;
    # if a deployment needs unbounded history it can override at
    # construction. Compaction (``CompressionPipeline`` /
    # ``/compact``) is the right tool for serious context management;
    # this cap is just a memory-safety backstop.
    max_history: int = 2000

    def add_message(
        self,
        role: str,
        content: MessageContent,
        *,
        usage: dict[str, Any] | None = None,
    ):
        if len(self.messages) >= self.max_history:
            self.messages.pop(0)

        normalized_content = _normalize_message_content(content)
        self.messages.append(create_message(role, normalized_content, usage=usage))

    def add_existing_message(self, message: Message) -> None:
        """Append a fully-formed Message, preserving all of its fields.

        Use this when the caller already has a populated Message (e.g. the
        agent loop's on_message callback) — ``add_message(role, content)``
        rebuilds via ``create_message`` and drops AssistantMessage fields
        like ``usage``, ``model``, and ``requestId`` that aren't on the
        base ``Message`` dataclass.
        """
        if len(self.messages) >= self.max_history:
            self.messages.pop(0)
        self.messages.append(message)

    def add_user_message(self, content: MessageContent):
        # ``MessageContent = str | list[ContentBlock]``: ``add_message`` ->
        # ``_normalize_message_content`` already handles both shapes, so
        # callers can pass a multi-block content list (e.g. text + image
        # blocks from @image.png @-mentions) the same way they pass a
        # plain string.
        self.add_message("user", content)

    def add_assistant_message(self, content: MessageContent):
        self.add_message("assistant", content)

    def add_tool_result_message(
        self,
        tool_use_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool = False,
        duration_ms: int | None = None,
    ):
        block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
            duration_ms=duration_ms,
        )
        self.add_message("user", [block])

    def get_messages(self) -> list[dict[str, Any]]:
        return normalize_messages_for_api(self.messages)

    def clear(self):
        self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message_to_dict(message) for message in self.messages],
            "max_history": self.max_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        conv = cls(max_history=data.get("max_history", 100))
        for msg_data in data.get("messages") or []:
            if isinstance(msg_data, dict):
                conv.messages.append(message_from_dict(msg_data))
        return conv


def _normalize_message_content(content: Any) -> MessageContent:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return normalize_content_blocks(content)
    return str(content)
