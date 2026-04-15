# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/parser/chat_template module."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template import (
    ChatTemplateParser,
    DeepseekQwenChatTemplateParser,
    LlamaChatTemplateParser,
    QwenChatTemplateParser,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.name_or_path = "test-model"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    return tokenizer


@pytest.fixture
def base_parser(mock_tokenizer):
    return ChatTemplateParser(mock_tokenizer)


@pytest.fixture
def deepseek_parser(mock_tokenizer):
    return DeepseekQwenChatTemplateParser(mock_tokenizer)


@pytest.fixture
def deepseek_parser_with_thinking(mock_tokenizer):
    return DeepseekQwenChatTemplateParser(mock_tokenizer, no_thinking=False)


@pytest.fixture
def qwen_parser(mock_tokenizer):
    return QwenChatTemplateParser(mock_tokenizer)


@pytest.fixture
def qwen_parser_thinking_enabled(mock_tokenizer):
    return QwenChatTemplateParser(mock_tokenizer, disable_thinking=False)


@pytest.fixture
def llama_parser(mock_tokenizer):
    return LlamaChatTemplateParser(mock_tokenizer)


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool", "content": "tool result"},
    ]


# ---------------------------------------------------------------------------
# ChatTemplateParser tests
# ---------------------------------------------------------------------------

class TestChatTemplateParser:
    def test_init_sets_tokenizer(self, base_parser, mock_tokenizer):
        assert base_parser.tokenizer is mock_tokenizer
        assert base_parser.assistant_token == ""

    def test_parse_delegates_to_tokenizer(self, base_parser, mock_tokenizer):
        messages = [{"role": "user", "content": "hello"}]
        result = base_parser.parse(messages, add_generation_prompt=True)
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert result == "<formatted>"

    def test_parse_default_no_generation_prompt(self, base_parser, mock_tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        base_parser.parse(messages)
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=False
        )

    def test_verify_equivalence_returns_true_when_equivalent(self, mock_tokenizer):
        mock_tokenizer.apply_chat_template = MagicMock(side_effect=lambda msgs, **kw: "".join(
            f"[{m['role']}:{m['content']}]" for m in msgs
        ))
        parser = ChatTemplateParser(mock_tokenizer)
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert parser.verify_equivalence(messages) is True

    def test_verify_equivalence_returns_false_when_not_equivalent(self, mock_tokenizer):
        call_count = [0]

        def side_effect(msgs, **kw):
            call_count[0] += 1
            if len(msgs) > 1:
                return "BATCH_DIFFERENT"
            return f"[{msgs[0]['role']}]"

        mock_tokenizer.apply_chat_template = MagicMock(side_effect=side_effect)
        parser = ChatTemplateParser(mock_tokenizer)
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert parser.verify_equivalence(messages, verbose=False) is False

    def test_verify_equivalence_raises_when_verbose(self, mock_tokenizer):
        def side_effect(msgs, **kw):
            if len(msgs) > 1:
                return "BATCH_DIFFERENT"
            return f"[{msgs[0]['role']}]"

        mock_tokenizer.apply_chat_template = MagicMock(side_effect=side_effect)
        parser = ChatTemplateParser(mock_tokenizer)
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        with pytest.raises(AssertionError, match="Parser failed equivalence check"):
            parser.verify_equivalence(messages, verbose=True)


class TestChatTemplateParserGetParser:
    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_deepseek_with_llama_tokenizer(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "deepseek-model-v2"
        tokenizer.__class__.__name__ = "LlamaTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, DeepseekQwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_qwen_model(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "Qwen-7B-Chat"
        tokenizer.__class__.__name__ = "QwenTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, QwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_r2e_model_returns_qwen(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "r2e-model"
        tokenizer.__class__.__name__ = "SomeTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, QwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_deepswe_model_returns_qwen(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "deepswe-model"
        tokenizer.__class__.__name__ = "SomeTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, QwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_llama_model(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "llama-3-8B"
        tokenizer.__class__.__name__ = "SomeTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, LlamaChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_default_parser_when_equivalence_passes(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "unknown-model"
        tokenizer.__class__.__name__ = "UnknownTokenizer"
        tokenizer.apply_chat_template = MagicMock(side_effect=lambda msgs, **kw: "".join(
            f"[{m['role']}:{m['content']}]" for m in msgs
        ))
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, ChatTemplateParser)
        assert not isinstance(result, (DeepseekQwenChatTemplateParser, QwenChatTemplateParser, LlamaChatTemplateParser))

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_default_parser_raises_when_equivalence_fails(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "unknown-model"
        tokenizer.__class__.__name__ = "UnknownTokenizer"

        def side_effect(msgs, **kw):
            if len(msgs) > 1:
                return "BATCH_RESULT"
            return f"[{msgs[0]['role']}]"

        tokenizer.apply_chat_template = MagicMock(side_effect=side_effect)
        with pytest.raises(Exception, match="Parser failed equivalence check"):
            ChatTemplateParser.get_parser(tokenizer)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_deepscaler_with_llama_tokenizer(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "deepscaler-model"
        tokenizer.__class__.__name__ = "LlamaTokenizerFast"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, DeepseekQwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_deepcoder_with_llama_tokenizer(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "deepcoder-v1"
        tokenizer.__class__.__name__ = "LlamaTokenizer"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, DeepseekQwenChatTemplateParser)

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template.logger")
    def test_qwen_tokenizer_class(self, mock_logger):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "some-custom-model"
        tokenizer.__class__.__name__ = "QwenTokenizerFast"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        result = ChatTemplateParser.get_parser(tokenizer)
        assert isinstance(result, QwenChatTemplateParser)


# ---------------------------------------------------------------------------
# DeepseekQwenChatTemplateParser tests
# ---------------------------------------------------------------------------

class TestDeepseekQwenChatTemplateParser:
    def test_init_tokens(self, deepseek_parser, mock_tokenizer):
        assert deepseek_parser.bos_token == mock_tokenizer.bos_token
        assert deepseek_parser.eos_token == mock_tokenizer.eos_token
        assert deepseek_parser.user_token == "<｜User｜>"
        assert "Assistant" in deepseek_parser.assistant_token
        assert "</think>" in deepseek_parser.assistant_token

    def test_init_with_thinking(self, deepseek_parser_with_thinking):
        assert "</think>" not in deepseek_parser_with_thinking.assistant_token

    def test_parse_system(self, deepseek_parser):
        msg = {"role": "system", "content": "system prompt"}
        result = deepseek_parser.parse_system(msg)
        assert "system prompt" in result

    def test_parse_user(self, deepseek_parser):
        msg = {"role": "user", "content": "hello"}
        result = deepseek_parser.parse_user(msg)
        assert "<｜User｜>" in result
        assert "hello" in result

    def test_parse_assistant(self, deepseek_parser):
        msg = {"role": "assistant", "content": "response"}
        result = deepseek_parser.parse_assistant(msg)
        assert "response" in result
        assert deepseek_parser.eos_token in result

    def test_parse_tool(self, deepseek_parser):
        msg = {"role": "tool", "content": "tool output"}
        result = deepseek_parser.parse_tool(msg)
        assert "tool output" in result
        assert "tool_output_begin" in result
        assert "tool_output_end" in result

    def test_parse_full_conversation(self, deepseek_parser, sample_messages):
        result = deepseek_parser.parse(sample_messages)
        assert "You are helpful." in result
        assert "Hello" in result
        assert "Hi there" in result
        assert "tool result" in result

    def test_parse_with_generation_prompt(self, deepseek_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = deepseek_parser.parse(messages, add_generation_prompt=True)
        assert deepseek_parser.generation_prompt in result

    def test_parse_with_first_msg(self, deepseek_parser, mock_tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        result = deepseek_parser.parse(messages, is_first_msg=True)
        assert result.startswith(mock_tokenizer.bos_token)

    def test_parse_unsupported_role_raises(self, deepseek_parser):
        messages = [{"role": "unknown", "content": "bad"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            deepseek_parser.parse(messages)

    def test_parse_empty_messages(self, deepseek_parser):
        result = deepseek_parser.parse([])
        assert result == ""


# ---------------------------------------------------------------------------
# QwenChatTemplateParser tests
# ---------------------------------------------------------------------------

class TestQwenChatTemplateParser:
    def test_init_tokens(self, qwen_parser):
        assert qwen_parser.system_token == "<|im_start|>system\n"
        assert qwen_parser.user_token == "<|im_start|>user\n"
        assert "<|im_start|>assistant" in qwen_parser.assistant_token
        assert qwen_parser.eot_token == "<|im_end|>\n"

    def test_init_thinking_disabled(self, qwen_parser):
        assert "think>" in qwen_parser.assistant_token

    def test_init_thinking_enabled(self, qwen_parser_thinking_enabled):
        assert "think>" not in qwen_parser_thinking_enabled.assistant_token

    def test_parse_system(self, qwen_parser):
        msg = {"role": "system", "content": "sys msg"}
        result = qwen_parser.parse_system(msg)
        assert "<|im_start|>system\n" in result
        assert "sys msg" in result
        assert "<|im_end|>" in result

    def test_parse_user(self, qwen_parser):
        msg = {"role": "user", "content": "user msg"}
        result = qwen_parser.parse_user(msg)
        assert "<|im_start|>user\n" in result
        assert "user msg" in result

    def test_parse_assistant(self, qwen_parser):
        msg = {"role": "assistant", "content": "assistant msg"}
        result = qwen_parser.parse_assistant(msg)
        assert "assistant msg" in result

    def test_parse_tool(self, qwen_parser):
        msg = {"role": "tool", "content": "tool result"}
        result = qwen_parser.parse_tool(msg)
        assert "<tool_response>" in result
        assert "tool result" in result
        assert "</tool_response>" in result

    def test_parse_adds_default_system_when_first_msg_not_system(self, qwen_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = qwen_parser.parse(messages, is_first_msg=True)
        assert "You are Qwen" in result

    def test_parse_no_default_system_when_system_present(self, qwen_parser):
        messages = [
            {"role": "system", "content": "Custom system."},
            {"role": "user", "content": "hi"},
        ]
        result = qwen_parser.parse(messages, is_first_msg=True)
        assert "You are Qwen" not in result
        assert "Custom system." in result

    def test_parse_with_generation_prompt(self, qwen_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = qwen_parser.parse(messages, add_generation_prompt=True)
        assert qwen_parser.generation_prompt in result

    def test_parse_unsupported_role_raises(self, qwen_parser):
        messages = [{"role": "function", "content": "bad"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            qwen_parser.parse(messages)

    def test_parse_full_conversation(self, qwen_parser, sample_messages):
        result = qwen_parser.parse(sample_messages)
        assert "You are helpful." in result
        assert "Hello" in result
        assert "Hi there" in result
        assert "tool result" in result


# ---------------------------------------------------------------------------
# LlamaChatTemplateParser tests
# ---------------------------------------------------------------------------

class TestLlamaChatTemplateParser:
    def test_init_tokens(self, llama_parser):
        assert llama_parser.bos_token == "<|begin_of_text|>"
        assert "system" in llama_parser.system_token
        assert "user" in llama_parser.user_token
        assert "assistant" in llama_parser.assistant_token
        assert llama_parser.eot_token == "<|eot_id|>"

    def test_parse_system(self, llama_parser):
        msg = {"role": "system", "content": "sys"}
        result = llama_parser.parse_system(msg)
        assert "system" in result
        assert "sys" in result
        assert "<|eot_id|>" in result

    def test_parse_user(self, llama_parser):
        msg = {"role": "user", "content": "question"}
        result = llama_parser.parse_user(msg)
        assert "user" in result
        assert "question" in result

    def test_parse_assistant(self, llama_parser):
        msg = {"role": "assistant", "content": "answer"}
        result = llama_parser.parse_assistant(msg)
        assert "assistant" in result
        assert "answer" in result

    def test_parse_tool(self, llama_parser):
        msg = {"role": "tool", "content": "tool output"}
        result = llama_parser.parse_tool(msg)
        assert "tool_response" in result
        assert "tool output" in result

    def test_parse_with_bos_on_first_msg(self, llama_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = llama_parser.parse(messages, is_first_msg=True)
        assert result.startswith("<|begin_of_text|>")

    def test_parse_without_bos(self, llama_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = llama_parser.parse(messages, is_first_msg=False)
        assert not result.startswith("<|begin_of_text|>")

    def test_parse_with_generation_prompt(self, llama_parser):
        messages = [{"role": "user", "content": "hi"}]
        result = llama_parser.parse(messages, add_generation_prompt=True)
        assert llama_parser.generation_prompt in result

    def test_parse_unsupported_role_raises(self, llama_parser):
        messages = [{"role": "admin", "content": "bad"}]
        with pytest.raises(NotImplementedError, match="Unsupported message role"):
            llama_parser.parse(messages)

    def test_parse_empty_messages(self, llama_parser):
        result = llama_parser.parse([])
        assert result == ""

    def test_parse_full_conversation(self, llama_parser, sample_messages):
        result = llama_parser.parse(sample_messages)
        assert "You are helpful." in result
        assert "Hello" in result
        assert "Hi there" in result
        assert "tool result" in result
