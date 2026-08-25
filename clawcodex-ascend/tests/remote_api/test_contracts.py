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

"""Unit tests for Remote API authentication, normalization, and state."""

from __future__ import annotations

import threading

import pytest

from extensions.remote_api.auth import require_bearer_auth, resolve_api_key
from extensions.remote_api.errors import RemoteAPIError
from extensions.remote_api.normalization import (
    merge_instructions,
    normalize_chat_messages,
    normalize_responses_input,
    reject_workspace_override,
)
from extensions.remote_api.state import ResponseStore
from clawcodex_ext.types.content_blocks import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from clawcodex_ext.types.messages import AssistantMessage, UserMessage


def test_remote_api_error_uses_openai_compatible_shape() -> None:
    error = RemoteAPIError(401, "bad token")

    assert error.code == "unauthorized"
    assert error.to_payload() == {
        "detail": "bad token",
        "error": {
            "message": "bad token",
            "type": "invalid_request_error",
            "code": "unauthorized",
        },
    }
    assert RemoteAPIError(404, "missing").code == "not_found"
    assert RemoteAPIError(429, "slow down").code == "rate_limit_exceeded"
    assert RemoteAPIError(504, "late").code == "timeout"
    assert RemoteAPIError(500, "failed").code == "internal_error"


def test_api_key_precedence_and_explicit_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAWCODEX_API_KEY", "primary")
    monkeypatch.setenv("API_SERVER_KEY", "fallback")

    assert resolve_api_key(None) == "primary"
    assert resolve_api_key("configured") == "configured"
    assert resolve_api_key("") is None

    monkeypatch.delenv("CLAWCODEX_API_KEY")
    assert resolve_api_key(None) == "fallback"


def test_bearer_auth_is_optional_and_uses_exact_token() -> None:
    require_bearer_auth(None, None)
    require_bearer_auth("secret", "Bearer secret")

    with pytest.raises(RemoteAPIError, match="missing bearer token") as missing:
        require_bearer_auth("secret", None)
    assert missing.value.status_code == 401

    with pytest.raises(RemoteAPIError, match="invalid bearer token"):
        require_bearer_auth("secret", "Bearer other")


@pytest.mark.parametrize("field", ["cwd", "workspace", "workdir", "working_dir", "root_dir"])
def test_workspace_override_is_rejected(field: str) -> None:
    with pytest.raises(RemoteAPIError, match=field):
        reject_workspace_override({field: "/tmp/other"})


def test_chat_normalization_preserves_history_instructions_and_tool_blocks() -> None:
    normalized = normalize_chat_messages(
        [
            {"role": "system", "content": "system rule"},
            {"role": "developer", "content": "developer rule"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ]
    )

    assert normalized.instructions == "system rule\n\ndeveloper rule"
    assert isinstance(normalized.messages[0], UserMessage)
    assert normalized.messages[0].content == "hello"
    assistant = normalized.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(assistant.content, list)
    assert assistant.content[0] == TextBlock(text="checking")
    assert assistant.content[1] == ToolUseBlock(
        id="call_1",
        name="Read",
        input={"path": "README.md"},
    )
    result = normalized.messages[2]
    assert isinstance(result, UserMessage)
    assert result.origin == "tool_result"
    assert result.content == [ToolResultBlock(tool_use_id="call_1", content="contents")]


def test_chat_normalization_requires_a_user_and_valid_tool_arguments() -> None:
    with pytest.raises(RemoteAPIError, match="non-empty array"):
        normalize_chat_messages([])
    with pytest.raises(RemoteAPIError, match="include a user"):
        normalize_chat_messages([{"role": "system", "content": "only"}])
    with pytest.raises(RemoteAPIError, match="valid JSON"):
        normalize_chat_messages(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "Read", "arguments": "{"},
                        }
                    ],
                },
            ]
        )


def test_data_image_is_converted_and_remote_image_is_rejected() -> None:
    normalized = normalize_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            }
        ]
    )

    blocks = normalized.messages[0].content
    assert isinstance(blocks, list)
    assert blocks[0] == TextBlock(text="inspect")
    assert blocks[1] == ImageBlock(source={"type": "base64", "media_type": "image/png", "data": "aGVsbG8="})

    with pytest.raises(RemoteAPIError, match="remote image URLs"):
        normalize_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example/image.png"},
                        }
                    ],
                }
            ]
        )


def test_responses_input_image_accepts_dict_image_url() -> None:
    normalized = normalize_responses_input(
        [
            {
                "type": "input_image",
                "image_url": {
                    "url": "data:image/png;base64,QUJD",
                    "detail": "auto",
                },
            }
        ]
    )

    blocks = normalized.messages[0].content
    assert blocks == [ImageBlock(source={"type": "base64", "media_type": "image/png", "data": "QUJD"})]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "file", "file_id": "file_1"}],
        [{"type": "input_file", "file_data": "data:text/plain;base64,SGk="}],
        "data:text/plain;base64,SGk=",
    ],
)
def test_uploaded_files_and_bare_data_urls_are_rejected(content: object) -> None:
    with pytest.raises(RemoteAPIError, match="not supported"):
        normalize_chat_messages([{"role": "user", "content": content}])


def test_responses_input_accepts_string_messages_and_bare_parts() -> None:
    direct = normalize_responses_input("hello")
    assert len(direct.messages) == 1
    assert isinstance(direct.messages[0], UserMessage)
    assert direct.messages[0].content == "hello"

    history = normalize_responses_input(
        [
            {"role": "system", "content": "rule"},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "next"},
        ]
    )
    assert history.instructions == "rule"
    assert [message.role for message in history.messages] == ["assistant", "user"]

    parts = normalize_responses_input([{"type": "input_text", "text": "part"}])
    assert len(parts.messages) == 1
    assert isinstance(parts.messages[0], UserMessage)
    assert parts.messages[0].content == "part"
    assert merge_instructions(" first ", None, "second") == "first\n\nsecond"


def test_response_store_is_lru_and_cleans_conversation_aliases() -> None:
    store = ResponseStore(limit=2)
    store.put("r1", {"id": "r1"}, ["one"], [{"id": "i1"}], conversation="alpha")
    store.put("r2", {"id": "r2"}, ["two"], conversation="beta")
    stored = store.get("r1")
    assert stored is not None
    assert stored.input_items == [{"id": "i1"}]

    store.put("r3", {"id": "r3"}, ["three"], conversation="gamma")

    assert store.get("r2") is None
    assert store.latest_for_conversation("beta") is None
    assert store.latest_for_conversation("alpha").response == {"id": "r1"}
    assert store.counts() == {"responses": 2, "conversations": 2, "limit": 2}

    assert store.delete("r1") is True
    assert store.delete("r1") is False
    assert store.latest_for_conversation("alpha") is None


def test_response_store_overwrite_replaces_alias_and_snapshots_response() -> None:
    store = ResponseStore()
    response = {"id": "r1", "output": [{"text": "original"}]}
    store.put("r1", response, ["one"], conversation="alpha")

    response["output"][0]["text"] = "mutated"
    stored = store.get("r1")
    assert stored is not None
    assert stored.response["output"][0]["text"] == "original"

    store.put("r1", {"id": "r1"}, ["two"], conversation="beta")

    assert store.latest_for_conversation("alpha") is None
    assert store.latest_for_conversation("beta").messages == ["two"]


def test_latest_for_conversation_holds_lock_through_item_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResponseStore(limit=2)
    store.put("r1", {"id": "r1"}, ["one"], conversation="alpha")
    original_get = store.get
    competing_lock_results: list[bool] = []

    def observed_get(response_id: str):
        def probe_lock() -> None:
            acquired = store._lock.acquire(blocking=False)  # pylint: disable=protected-access
            competing_lock_results.append(acquired)
            if acquired:
                store._lock.release()  # pylint: disable=protected-access

        worker = threading.Thread(target=probe_lock)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
        return original_get(response_id)

    monkeypatch.setattr(store, "get", observed_get)

    stored = store.latest_for_conversation("alpha")

    assert stored is not None
    assert stored.response == {"id": "r1"}
    assert competing_lock_results == [False]
