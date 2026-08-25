#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Unit tests for Remote API SSE encoding and OpenAI-compatible chunks."""

from __future__ import annotations

import json

import pytest

from extensions.remote_api.sse import (
    chat_chunk,
    chat_usage_chunk,
    encode_done,
    encode_sse,
)


def test_sse_encoding_preserves_unicode_and_multiline_payloads() -> None:
    frame = encode_sse({"text": "你好"}, event="message")
    assert frame.startswith("event: message\ndata: ")
    assert json.loads(frame.split("data: ", 1)[1].strip()) == {"text": "你好"}
    assert encode_sse("first\nsecond") == "data: first\ndata: second\n\n"
    assert encode_done() == "data: [DONE]\n\n"


def test_openai_chat_chunks_include_delta_and_optional_usage() -> None:
    chunk = chat_chunk(
        chunk_id="chatcmpl_1",
        created=1,
        model="model",
        delta={"content": "hi"},
        include_usage=True,
    )

    assert chunk == {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hi"},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }


def test_openai_usage_chunk_has_no_choices() -> None:
    usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}

    assert chat_usage_chunk(
        chunk_id="chatcmpl_1",
        created=1,
        model="model",
        usage=usage,
    ) == {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": [],
        "usage": usage,
    }


@pytest.mark.parametrize(
    "event",
    [
        "message\ndata: injected",
        "message\rdata: injected",
        "message\r\ndata: injected",
    ],
)
def test_sse_event_names_reject_line_breaks(event: str) -> None:
    with pytest.raises(ValueError, match="must not contain CR or LF"):
        encode_sse({"ok": True}, event=event)
