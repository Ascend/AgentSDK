# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for rllm/msg_handler module."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rl.runner.agent_engine_wrapper.rllm.msg_handler import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_conversation():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
    ]


@pytest.fixture
def conversation_with_tool():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Search for python"},
        {"role": "assistant", "content": "I will search."},
        {"role": "tool", "content": "Python is a language."},
        {"role": "user", "content": "Thanks"},
    ]


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.assistant_token = "<assistant>"

    def parse_side_effect(msgs, add_generation_prompt=False, is_first_msg=False):
        result = ""
        for m in msgs:
            if m["role"] == "assistant":
                result += f"<assistant>{m['content']}"
            else:
                result += f"<{m['role']}>{m['content']}"
        if add_generation_prompt:
            result += "<gen>"
        return result

    parser.parse = MagicMock(side_effect=parse_side_effect)
    return parser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    def encode_side_effect(text, add_special_tokens=True):
        return list(range(len(text)))

    tokenizer.encode = MagicMock(side_effect=encode_side_effect)
    return tokenizer


# ---------------------------------------------------------------------------
# get_recent_assistant_user_messages tests
# ---------------------------------------------------------------------------

class TestGetRecentAssistantUserMessages:
    def test_simple_conversation(self, simple_conversation):
        assistant_msg, env_msgs = get_recent_assistant_user_messages(simple_conversation)
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Hi"
        assert len(env_msgs) == 1
        assert env_msgs[0]["content"] == "How are you?"

    def test_conversation_with_tool(self, conversation_with_tool):
        assistant_msg, env_msgs = get_recent_assistant_user_messages(conversation_with_tool)
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "I will search."
        assert len(env_msgs) == 2
        assert env_msgs[0]["role"] == "tool"
        assert env_msgs[1]["role"] == "user"

    def test_no_assistant_message(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg is None
        assert len(env_msgs) == 1
        assert env_msgs[0]["content"] == "hello"

    def test_only_assistant_message(self):
        messages = [{"role": "assistant", "content": "response"}]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg["content"] == "response"
        assert env_msgs == []

    def test_empty_messages(self):
        assistant_msg, env_msgs = get_recent_assistant_user_messages([])
        assert assistant_msg is None
        assert env_msgs == []

    def test_multiple_assistant_messages(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg["content"] == "a2"
        assert len(env_msgs) == 1
        assert env_msgs[0]["content"] == "q3"

    def test_assistant_at_end(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg["content"] == "hi"
        assert env_msgs == []

    def test_env_messages_in_chronological_order(self):
        messages = [
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "first"},
            {"role": "tool", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg["content"] == "thinking"
        assert len(env_msgs) == 3
        assert env_msgs[0]["content"] == "first"
        assert env_msgs[1]["content"] == "second"
        assert env_msgs[2]["content"] == "third"

    def test_system_messages_ignored(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "sys2"},
        ]
        assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
        assert assistant_msg["content"] == "hi"
        assert env_msgs == []


# ---------------------------------------------------------------------------
# convert_messages_to_tokens_and_masks tests
# ---------------------------------------------------------------------------

class TestConvertMessagesToTokensAndMasks:
    def test_user_message_mask_is_zero(self, mock_tokenizer, mock_parser):
        messages = [{"role": "user", "content": "hello"}]
        tokens, masks = convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser
        )
        assert all(m == 0 for m in masks)

    def test_assistant_message_mask_is_one(self, mock_tokenizer, mock_parser):
        messages = [{"role": "assistant", "content": "response"}]
        tokens, masks = convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser
        )
        assert all(m == 1 for m in masks)

    def test_tokens_and_masks_same_length(self, mock_tokenizer, mock_parser):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        tokens, masks = convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser
        )
        assert len(tokens) == len(masks)

    def test_empty_messages(self, mock_tokenizer, mock_parser):
        tokens, masks = convert_messages_to_tokens_and_masks(
            [], tokenizer=mock_tokenizer, parser=mock_parser
        )
        assert tokens == []
        assert masks == []

    def test_contains_first_msg_flag(self, mock_tokenizer, mock_parser):
        messages = [{"role": "user", "content": "hi"}]
        convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser,
            contains_first_msg=True
        )
        mock_parser.parse.assert_called_with(
            [messages[0]], add_generation_prompt=False, is_first_msg=True
        )

    def test_contains_generation_msg_flag(self, mock_tokenizer, mock_parser):
        messages = [{"role": "user", "content": "hi"}]
        convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser,
            contains_generation_msg=True
        )
        mock_parser.parse.assert_called_with(
            [messages[0]], add_generation_prompt=True, is_first_msg=False
        )

    def test_assistant_token_stripped(self, mock_tokenizer, mock_parser):
        messages = [{"role": "assistant", "content": "test"}]
        convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser
        )
        call_args = mock_tokenizer.encode.call_args
        encoded_text = call_args[0][0]
        assert not encoded_text.startswith("<assistant>")

    def test_assistant_token_not_found_raises(self, mock_tokenizer):
        parser = MagicMock()
        parser.assistant_token = "<MISSING_TOKEN>"
        parser.parse = MagicMock(return_value="<other>response")

        messages = [{"role": "assistant", "content": "test"}]
        with pytest.raises(Exception, match="Expected assistant token"):
            convert_messages_to_tokens_and_masks(
                messages, tokenizer=mock_tokenizer, parser=parser
            )

    def test_mixed_roles(self, mock_tokenizer, mock_parser):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "result"},
        ]
        tokens, masks = convert_messages_to_tokens_and_masks(
            messages, tokenizer=mock_tokenizer, parser=mock_parser
        )
        assert len(tokens) == len(masks)
        assert len(tokens) > 0
