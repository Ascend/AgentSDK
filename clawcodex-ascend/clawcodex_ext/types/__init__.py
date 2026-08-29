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

"""Typed message/content/stream-event models — full upstream re-exports.

The fork-side facade at :mod:`src.types` adds the upstream archive
placeholder metadata (SNAPSHOT_PATH, ARCHIVE_NAME, MODULE_COUNT,
SAMPLE_FILES, PORTING_NOTE) which depends on the file location in src/.
"""

from __future__ import annotations

from clawcodex_ext.types.content_blocks import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    content_block_from_dict,
    content_block_to_dict,
    normalize_content_blocks,
)
from clawcodex_ext.types.messages import (
    CANCEL_MESSAGE,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    NO_CONTENT_MESSAGE,
    NO_RESPONSE_REQUESTED,
    REJECT_MESSAGE,
    REJECT_MESSAGE_WITH_REASON_PREFIX,
    SYNTHETIC_MODEL,
    SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
    AssistantMessage,
    AttachmentMessage,
    Message,
    MessageContent,
    MessageLike,
    MessageOrigin,
    MessageType,
    ProgressMessage,
    SystemMessage,
    TypedMessage,
    UserMessage,
    create_assistant_api_error_message,
    create_assistant_message,
    create_message,
    create_progress_message,
    create_system_message,
    create_user_message,
    get_last_assistant_message,
    get_tool_use_ids,
    is_tool_use_request_message,
    is_tool_use_result_message,
    message_from_dict,
    message_to_dict,
    normalize_message_for_api,
    normalize_messages_for_api,
)
from clawcodex_ext.types.stream_events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    stream_event_from_dict,
    stream_event_to_dict,
)

__all__ = [
    "CANCEL_MESSAGE",
    "AssistantMessage",
    "AttachmentMessage",
    "ContentBlock",
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "DocumentBlock",
    "INTERRUPT_MESSAGE",
    "INTERRUPT_MESSAGE_FOR_TOOL_USE",
    "ImageBlock",
    "Message",
    "MessageContent",
    "MessageDelta",
    "MessageLike",
    "MessageOrigin",
    "MessageStart",
    "MessageStop",
    "MessageType",
    "NO_CONTENT_MESSAGE",
    "NO_RESPONSE_REQUESTED",
    "ProgressMessage",
    "REJECT_MESSAGE",
    "REJECT_MESSAGE_WITH_REASON_PREFIX",
    "SYNTHETIC_MODEL",
    "SYNTHETIC_TOOL_RESULT_PLACEHOLDER",
    "RedactedThinkingBlock",
    "StreamEvent",
    "SystemMessage",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "TypedMessage",
    "UserMessage",
    "content_block_from_dict",
    "content_block_to_dict",
    "create_assistant_api_error_message",
    "create_assistant_message",
    "create_message",
    "create_progress_message",
    "create_system_message",
    "create_user_message",
    "get_last_assistant_message",
    "get_tool_use_ids",
    "is_tool_use_request_message",
    "is_tool_use_result_message",
    "message_from_dict",
    "message_to_dict",
    "normalize_content_blocks",
    "normalize_message_for_api",
    "normalize_messages_for_api",
    "stream_event_from_dict",
    "stream_event_to_dict",
]
