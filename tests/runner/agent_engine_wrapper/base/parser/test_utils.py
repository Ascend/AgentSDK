# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/parser/utils module: PARSER_TEST_MESSAGES constant."""

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.parser.utils import PARSER_TEST_MESSAGES


class TestParserTestMessages:
    def test_is_list(self):
        assert isinstance(PARSER_TEST_MESSAGES, list)

    def test_not_empty(self):
        assert len(PARSER_TEST_MESSAGES) > 0

    def test_each_element_is_dict(self):
        for msg in PARSER_TEST_MESSAGES:
            assert isinstance(msg, dict)

    def test_each_element_has_role_and_content(self):
        for msg in PARSER_TEST_MESSAGES:
            assert "role" in msg
            assert "content" in msg

    def test_valid_roles(self):
        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in PARSER_TEST_MESSAGES:
            assert msg["role"] in valid_roles

    def test_contains_system_message(self):
        roles = [msg["role"] for msg in PARSER_TEST_MESSAGES]
        assert "system" in roles

    def test_contains_user_message(self):
        roles = [msg["role"] for msg in PARSER_TEST_MESSAGES]
        assert "user" in roles

    def test_contains_assistant_message(self):
        roles = [msg["role"] for msg in PARSER_TEST_MESSAGES]
        assert "assistant" in roles

    def test_assistant_messages_have_tool_calls(self):
        for msg in PARSER_TEST_MESSAGES:
            if msg["role"] == "assistant":
                assert "tool_calls" in msg
                assert isinstance(msg["tool_calls"], list)
