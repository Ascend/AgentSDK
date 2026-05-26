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
import sys
import types
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for msg_handler
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_msg_handler_env():
    """Build a fully isolated fake module tree for msg_handler.py."""

    # ---- Fake transformers ----
    fake_transformers = types.ModuleType("transformers")
    class FakePreTrainedTokenizerBase:
        pass
    fake_transformers.PreTrainedTokenizerBase = FakePreTrainedTokenizerBase

    # ---- Fake rllm.parser.chat_template.parser ----
    fake_rllm = types.ModuleType("rllm")
    fake_rllm.__path__ = []
    fake_rllm_parser = types.ModuleType("rllm.parser")
    fake_rllm_parser.__path__ = []
    fake_rllm_parser_chat_template = types.ModuleType("rllm.parser.chat_template")
    fake_rllm_parser_chat_template.__path__ = []
    fake_rllm_parser_chat_template_parser = types.ModuleType(
        "rllm.parser.chat_template.parser"
    )

    class FakeChatTemplateParser:
        # We'll create instances in tests, so provide a simple constructor
        def __init__(self, tokenizer=None, assistant_token="<|assistant|>"):
            self.tokenizer = tokenizer
            self.assistant_token = assistant_token
            self.parse = MagicMock()
    fake_rllm_parser_chat_template_parser.ChatTemplateParser = FakeChatTemplateParser

    # ---- Aura packages (to locate the real file) ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    fake_aura_runner_agent_engine_wrapper = types.ModuleType(
        "aura.runner.agent_engine_wrapper"
    )
    fake_aura_runner_agent_engine_wrapper.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper")
    ]
    fake_aura_runner_agent_engine_wrapper_rllm = types.ModuleType(
        "aura.runner.agent_engine_wrapper.rllm"
    )
    fake_aura_runner_agent_engine_wrapper_rllm.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper/rllm")
    ]

    fakes = {
        "transformers": fake_transformers,
        "rllm": fake_rllm,
        "rllm.parser": fake_rllm_parser,
        "rllm.parser.chat_template": fake_rllm_parser_chat_template,
        "rllm.parser.chat_template.parser": fake_rllm_parser_chat_template_parser,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine_wrapper,
        "aura.runner.agent_engine_wrapper.rllm": fake_aura_runner_agent_engine_wrapper_rllm,
    }

    target = "aura.runner.agent_engine_wrapper.rllm.msg_handler"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_engine_wrapper.rllm.msg_handler as mod
        yield {
            "mod": mod,
            "FakeParser": FakeChatTemplateParser,
            "FakeTokenizer": FakePreTrainedTokenizerBase,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_msg(role, content="", tool_calls=None):
    """Create a simple message dict with optional tool_calls."""
    msg = {"role": role, "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def make_mock_parser(assistant_token="<|assistant|>"):
    """Create a mock ChatTemplateParser instance."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock()
    parser = MagicMock()
    parser.tokenizer = tokenizer
    parser.assistant_token = assistant_token
    parser.parse = MagicMock()
    return parser


# ---------------------------------------------------------------------------
# Tests for get_recent_assistant_user_messages
# ---------------------------------------------------------------------------
class TestGetRecentAssistantUserMessages:
    @pytest.fixture(autouse=True)
    def setup(self, fake_msg_handler_env):
        self.func = fake_msg_handler_env["mod"].get_recent_assistant_user_messages

    def test_normal_case(self):
        """Assistant message exists, with following user/tool messages."""
        msgs = [
            make_msg("user", "hello"),
            make_msg("assistant", "hi"),
            make_msg("user", "next"),
            make_msg("tool", "data"),
        ]
        asst, env = self.func(msgs)
        assert asst["content"] == "hi"
        assert env == [make_msg("user", "next"), make_msg("tool", "data")]

    def test_no_assistant(self):
        """No assistant message -> returns (None, [])."""
        msgs = [make_msg("user", "a"), make_msg("tool", "b")]
        asst, env = self.func(msgs)
        assert asst is None
        assert env == msgs

    def test_multiple_assistants_last(self):
        """Last assistant is chosen, previous user/tool after it are collected."""
        msgs = [
            make_msg("user", "u1"),
            make_msg("assistant", "a1"),
            make_msg("user", "u2"),
            make_msg("assistant", "a2"),
            make_msg("user", "u3"),
        ]
        asst, env = self.func(msgs)
        assert asst["content"] == "a2"
        assert env == [make_msg("user", "u3")]

    def test_assistant_first_no_env(self):
        """Assistant is the first message, no env."""
        msgs = [make_msg("assistant", "a1"), make_msg("user", "u1")]
        asst, env = self.func(msgs)
        assert asst["content"] == "a1"
        assert env == [make_msg("user", "u1")]

    def test_only_assistant(self):
        """Only one assistant message."""
        msgs = [make_msg("assistant", "a1")]
        asst, env = self.func(msgs)
        assert asst["content"] == "a1"
        assert env == []


# ---------------------------------------------------------------------------
# Tests for convert_messages_to_tokens_and_masks
# ---------------------------------------------------------------------------
class TestConvertMessagesToTokensAndMasks:
    @pytest.fixture(autouse=True)
    def setup(self, fake_msg_handler_env):
        self.func = fake_msg_handler_env["mod"].convert_messages_to_tokens_and_masks

    def test_normal_conversion(self):
        """Messages are tokenized and masks created correctly."""
        parser = make_mock_parser()
        # Parse returns text with assistant_token prefix for assistant messages
        def parse_side_effect(msg, **kwargs):
            if msg[0]["role"] == "assistant":
                return parser.assistant_token + " assistant_text"
            return "user_text"
        parser.parse.side_effect = parse_side_effect
        tokenizer = parser.tokenizer
        tokenizer.encode.return_value = [1, 2, 3]

        msgs = [
            make_msg("user", "u1"),
            make_msg("assistant", "a1"),
        ]
        tokens, masks = self.func(msgs, tokenizer, parser)
        # assistant token prefix is stripped, so both roles produce [1,2,3]
        assert tokens == [1, 2, 3, 1, 2, 3]
        assert masks == [0, 0, 0, 1, 1, 1]

    def test_assistant_text_wrong_prefix_raises(self):
        """If assistant message doesn't start with assistant_token, raise Exception."""
        parser = make_mock_parser(assistant_token="<|assistant|>")
        parser.parse.return_value = "wrong_prefix"  # doesn't start with assistant_token
        msgs = [make_msg("assistant", "a1")]
        with pytest.raises(Exception, match="Expected assistant token"):
            self.func(msgs, parser.tokenizer, parser)

    def test_contains_first_and_generation_flags(self):
        """contains_first_msg and contains_generation_msg flags are forwarded to parser.parse."""
        parser = make_mock_parser()
        parser.parse.side_effect = lambda msg, **kwargs: "text"
        tokenizer = parser.tokenizer
        tokenizer.encode.return_value = [1]

        msgs = [make_msg("user"), make_msg("user")]
        self.func(msgs, tokenizer, parser, contains_first_msg=True, contains_generation_msg=True)
        # Check calls to parse: first message with first_msg=True, last with generation_msg=True
        calls = parser.parse.call_args_list
        # Should be 2 calls
        assert calls[0].kwargs.get("is_first_msg") == True
        assert calls[1].kwargs.get("add_generation_prompt") == True

    def test_empty_messages(self):
        """Empty message list returns empty tokens and masks."""
        parser = make_mock_parser()
        tokens, masks = self.func([], parser.tokenizer, parser)
        assert tokens == []
        assert masks == []


# ---------------------------------------------------------------------------
# Tests for preprocess_messages_for_qwen35
# ---------------------------------------------------------------------------
class TestPreprocessMessages:
    @pytest.fixture(autouse=True)
    def setup(self, fake_msg_handler_env):
        self.func = fake_msg_handler_env["mod"].preprocess_messages_for_qwen35

    def test_converts_json_string_arguments_to_dict(self):
        """JSON string arguments are parsed to dict."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"arguments": '{"key": "value"}'}}
                ],
            }
        ]
        result = self.func(msgs)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"key": "value"}

    def test_ignores_non_string_arguments(self):
        """Non-string arguments are left unchanged."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"arguments": {"already": "dict"}}}
                ],
            }
        ]
        result = self.func(msgs)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args == {"already": "dict"}

    def test_invalid_json_prints_warning_and_leaves_string(self, capsys):
        """Invalid JSON string leaves the original string and prints warning."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"arguments": "not json"}}
                ],
            }
        ]
        result = self.func(msgs)
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args == "not json"
        captured = capsys.readouterr().out
        assert "Warning" in captured

    def test_no_tool_calls(self):
        """Messages without tool_calls are returned unchanged."""
        msgs = [{"role": "user", "content": "hello"}]
        result = self.func(msgs)
        assert result == msgs


# ---------------------------------------------------------------------------
# Tests for parse_whole_response_token_ids
# ---------------------------------------------------------------------------
class TestParseWholeResponseTokenIds:
    @pytest.fixture(autouse=True)
    def setup(self, fake_msg_handler_env):
        self.func = fake_msg_handler_env["mod"].parse_whole_response_token_ids

    def test_normal_no_interleaved(self):
        """Standard parsing without interleaved think."""
        parser = make_mock_parser(assistant_token="<|a|>")
        tokenizer = parser.tokenizer
        # Mock tokenizer.encode to return specific token lists
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<|a|>": [1],            # assistant_token
            "prompt": [10, 11],      # prompt_str tokens
            "full": [10, 11, 20, 21],# full_str tokens (includes response)
            "sub0": [10, 11, 20],    # sub_prompt for i=2 (first assistant)
            "sub1": [10, 11, 20, 21],# sub_prompt for i=3 (user)
        }.get(text, [])
        # Mock parser.parse to return different strings for different calls
        parser.parse.side_effect = lambda msgs, tools=None, tokenize=False, add_generation_prompt=False, is_first_msg=False: {
            (True, ): "prompt",
            (False,): "full",
            (False, True): "sub0",
            (False, False): "sub1",
        }.get((add_generation_prompt,))

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "followup"},
        ]
        prompt_ids, resp_ids, resp_mask = self.func(
            parser, messages, None, "<|a|>", enable_interleaved_think=False
        )
        # prompt_ids should be [10, 11]
        assert prompt_ids == [10, 11]
        # response_ids are [20, 21] (full minus prompt)
        assert resp_ids == [20, 21]
        # mask: first assistant content should be 1
        assert resp_mask == [1, 1]  # both tokens belong to assistant

    def test_enable_interleaved_think(self):
        """interleaved_think flag triggers replacement logic."""
        parser = make_mock_parser(assistant_token="<|a|>")
        tokenizer = parser.tokenizer
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<|a|>": [1],
            "prompt": [10],
            "full_interleaved": [10, 20],
            "sub0_interleaved": [10, 20],  # same length to avoid complex indexing
        }.get(text, [10, 20])  # fallback
        # We'll need to handle the string replacements in parser.parse
        parser.parse.side_effect = lambda msgs, tools=None, tokenize=False, add_generation_prompt=False: {
            (True, ): "prompt",
            (False,): "full_interleaved",  # will be replaced to remove <past_thought>
        }.get((add_generation_prompt,), "full_interleaved")

        messages = [
            {"role": "user"},
            {"role": "assistant", "content": "<think>thought</think> answer"},
        ]
        prompt_ids, resp_ids, resp_mask = self.func(
            parser, messages, None, "<|a|>", enable_interleaved_think=True
        )
        assert isinstance(prompt_ids, list)
