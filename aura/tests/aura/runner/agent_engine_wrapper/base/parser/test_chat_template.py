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
import copy


# ---------------------------------------------------------------------------
# Fixture: fake module tree for chat_template
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_chat_template_env():
    """Create fake dependencies so chat_template can be imported without real ones."""

    # ---- Fake .utils ----
    fake_utils = types.ModuleType("aura.runner.agent_engine_wrapper.base.parser.utils")
    fake_utils.PARSER_TEST_MESSAGES = [{"role": "user", "content": "hello"}]

    # ---- Fake msg_handler ----
    fake_msg_handler = types.ModuleType(
        "aura.runner.agent_engine_wrapper.rllm.msg_handler"
    )
    fake_msg_handler.preprocess_messages_for_qwen35 = MagicMock(side_effect=lambda msgs: msgs)

    # ---- Fake loggers ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger)))

    # ---- Aura packages to locate real file ----
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
    fake_aura_runner_agent_engine_wrapper_base = types.ModuleType(
        "aura.runner.agent_engine_wrapper.base"
    )
    fake_aura_runner_agent_engine_wrapper_base.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper/base")
    ]
    fake_aura_runner_agent_engine_wrapper_base_parser = types.ModuleType(
        "aura.runner.agent_engine_wrapper.base.parser"
    )
    fake_aura_runner_agent_engine_wrapper_base_parser.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper/base/parser")
    ]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []

    fakes = {
        "aura.runner.agent_engine_wrapper.base.parser.utils": fake_utils,
        "aura.runner.agent_engine_wrapper.rllm.msg_handler": fake_msg_handler,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine_wrapper,
        "aura.runner.agent_engine_wrapper.base": fake_aura_runner_agent_engine_wrapper_base,
        "aura.runner.agent_engine_wrapper.base.parser": fake_aura_runner_agent_engine_wrapper_base_parser,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "copy": copy,
    }

    target = "aura.runner.agent_engine_wrapper.base.parser.chat_template"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_engine_wrapper.base.parser.chat_template as mod
        # Inject copy into module's global namespace for _manual_parse
        mod.copy = copy
        yield {
            "mod": mod,
            "ChatTemplateParser": mod.ChatTemplateParser,
            "DeepseekQwenChatTemplateParser": mod.DeepseekQwenChatTemplateParser,
            "QwenChatTemplateParser": mod.QwenChatTemplateParser,
            "LlamaChatTemplateParser": mod.LlamaChatTemplateParser,
            "Qwen3ChatTemplateParser": mod.Qwen3ChatTemplateParser,
            "PanguVocabLlmV4ChatTemplateParser": mod.PanguVocabLlmV4ChatTemplateParser,
            "mock_logger": mock_logger,
            "fake_msg_handler": fake_msg_handler,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_tokenizer_mock(name_or_path="qwen", cls_name="Qwen2Tokenizer", bos="", eos="<|im_end|>"):
    """Create a mock tokenizer with given attributes."""
    tok = MagicMock()
    tok.name_or_path = name_or_path
    tok.__class__.__name__ = cls_name
    tok.bos_token = bos
    tok.eos_token = eos
    tok.apply_chat_template = MagicMock(return_value="parsed_output")
    return tok


# ---------------------------------------------------------------------------
# Tests for base ChatTemplateParser
# ---------------------------------------------------------------------------
class TestChatTemplateParser:
    def test_parse_calls_apply_chat_template(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["ChatTemplateParser"](tokenizer)
        messages = [{"role": "user", "content": "hello"}]
        result = parser.parse(messages)
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=False
        )
        assert result == "parsed_output"

    def test_verify_equivalence_success(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["ChatTemplateParser"](tokenizer)
        tokenizer.apply_chat_template.side_effect = [
            "AB",    # batch
            "A",     # msg1
            "B",     # msg2
        ]
        result = parser.verify_equivalence(
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], verbose=True
        )
        assert result is True

    def test_verify_equivalence_failure_raises(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["ChatTemplateParser"](tokenizer)
        tokenizer.apply_chat_template.side_effect = ["batch", "different"]
        with pytest.raises(AssertionError, match="Parser failed equivalence check"):
            parser.verify_equivalence([{"role": "user", "content": "x"}], verbose=True)

    def test_get_parser_default(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="unknown", cls_name="UnknownTokenizer")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer)
        assert type(parser).__name__ == "ChatTemplateParser"

    def test_get_parser_qwen(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="Qwen2.5-7B", cls_name="Qwen2Tokenizer")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer)
        assert isinstance(parser, fake_chat_template_env["QwenChatTemplateParser"])

    def test_get_parser_deepseek_llama(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="deepseek-llama", cls_name="LlamaTokenizer")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer)
        assert isinstance(parser, fake_chat_template_env["DeepseekQwenChatTemplateParser"])

    def test_get_parser_llama(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="llama-3", cls_name="LlamaTokenizer")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer)
        assert isinstance(parser, fake_chat_template_env["LlamaChatTemplateParser"])

    def test_get_parser_sophon(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="sophon", cls_name="SophonTokenizer")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer)
        assert isinstance(parser, fake_chat_template_env["PanguVocabLlmV4ChatTemplateParser"])

    def test_get_parser_toolcall_qwen3_coder(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(name_or_path="anything", cls_name="anything")
        parser = fake_chat_template_env["ChatTemplateParser"].get_parser(tokenizer, toolcall_parser="qwen3_coder")
        assert isinstance(parser, fake_chat_template_env["Qwen3ChatTemplateParser"])


# ---------------------------------------------------------------------------
# Tests for DeepseekQwenChatTemplateParser
# ---------------------------------------------------------------------------
class TestDeepseekQwenChatTemplateParser:
    def test_parse_full_flow(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(bos="<BOS>", eos="<EOS>")
        parser = fake_chat_template_env["DeepseekQwenChatTemplateParser"](tokenizer)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
            {"role": "tool", "content": "tool"},
        ]
        result = parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        assert "<BOS>" in result
        assert "<EOS>" in result
        assert "<｜User｜>" in result
        assert "<｜Assistant｜>" in result
        assert "</think>" in result
        assert parser.tool_response_start_token in result

    def test_parse_first_msg_adds_bos(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(bos="<BOS>", eos="<EOS>")
        parser = fake_chat_template_env["DeepseekQwenChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "hi"}], is_first_msg=True)
        assert result.startswith("<BOS>")


# ---------------------------------------------------------------------------
# Tests for QwenChatTemplateParser
# ---------------------------------------------------------------------------
class TestQwenChatTemplateParser:
    def test_parse_adds_system_if_needed(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock(bos="", eos="<|im_end|>")
        parser = fake_chat_template_env["QwenChatTemplateParser"](tokenizer, disable_thinking=True)
        result = parser.parse([{"role": "user", "content": "hi"}], is_first_msg=True)
        assert "system" in result

    def test_parse_all_roles(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["QwenChatTemplateParser"](tokenizer, disable_thinking=False)
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "t"},
        ]
        result = parser.parse(messages, add_generation_prompt=True)
        assert "<|im_start|>" in result
        assert "<tool_response>" in result

    def test_disable_thinking_changes_assistant_token(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["QwenChatTemplateParser"](tokenizer, disable_thinking=True)
        assert "<think>" in parser.assistant_token

    def test_parse_tool_response(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["QwenChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "tool", "content": "tool_out"}])
        assert "<tool_response>" in result


# ---------------------------------------------------------------------------
# Tests for LlamaChatTemplateParser
# ---------------------------------------------------------------------------
class TestLlamaChatTemplateParser:
    def test_parse_all_roles(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["LlamaChatTemplateParser"](tokenizer)
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "t"},
        ]
        result = parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        assert "<|begin_of_text|>" in result
        assert "<|eot_id|>" in result

    def test_parse_first_msg_no_bos_if_not_first(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        parser = fake_chat_template_env["LlamaChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "u"}], is_first_msg=False)
        assert "<|begin_of_text|>" not in result


# ---------------------------------------------------------------------------
# Tests for Qwen3ChatTemplateParser
# ---------------------------------------------------------------------------
class TestQwen3ChatTemplateParser:
    def test_apply_chat_template_success(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.return_value = "parsed"
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "hi"}])
        assert result == "parsed"
        fake_chat_template_env["fake_msg_handler"].preprocess_messages_for_qwen35.assert_called_once()

    def test_apply_chat_template_falls_back_to_manual(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.side_effect = Exception("fail")
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)

    def test_manual_parse_with_tools(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.side_effect = Exception("fail")
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        messages = [{"role": "system", "content": ""}]
        tools = [{"type": "function", "function": {"name": "test", "description": "desc"}}]
        result = parser.parse(messages, tools=tools)
        assert "# Tools" in result

    def test_manual_parse_assistant_with_tool_calls(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.side_effect = Exception("fail")
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        messages = [{"role": "assistant", "content": "result",
                     "tool_calls": [{"function": {"name": "f1", "arguments": "{}"}}]}]
        result = parser.parse(messages)
        assert "<tool_call>" in result

    def test_manual_parse_list_content(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.side_effect = Exception("fail")
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        content_list = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        messages = [{"role": "user", "content": content_list}]
        result = parser.parse(messages)
        assert "part1part2" in result

    def test_manual_parse_add_generation_prompt(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.side_effect = Exception("fail")
        parser = fake_chat_template_env["Qwen3ChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": ""}], add_generation_prompt=True)
        assert parser.assistant_token in result


# ---------------------------------------------------------------------------
# Tests for PanguVocabLlmV4ChatTemplateParser
# ---------------------------------------------------------------------------
class TestPanguVocabLlmV4ChatTemplateParser:
    def test_parse_keeps_bos_for_first_msg(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.return_value = "<|pangu_text_start|>parsed"
        parser = fake_chat_template_env["PanguVocabLlmV4ChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "hi"}], is_first_msg=True)
        assert result.startswith("<|pangu_text_start|>")

    def test_parse_removes_bos_for_non_first_msg(self, fake_chat_template_env):
        tokenizer = make_tokenizer_mock()
        tokenizer.apply_chat_template.return_value = "<|pangu_text_start|>parsed"
        parser = fake_chat_template_env["PanguVocabLlmV4ChatTemplateParser"](tokenizer)
        result = parser.parse([{"role": "user", "content": "hi"}], is_first_msg=False)
        assert not result.startswith("<|pangu_text_start|>")
