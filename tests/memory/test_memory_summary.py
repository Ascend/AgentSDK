#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from agentic_rl.memory.memory_summary import MemorySummary, SUMMARY
from agentic_rl.memory.token_counter import SimpleTokenCounter


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=OpenAI)


@pytest.fixture
def memory_summary(mock_openai_client):
    config = {
        "max_summary_length": 20,
        "max_prompt_length": 100,
        "use_summary": True,
        "oai_model_name": "test_model",
        "oai_client": mock_openai_client,
    }

    memory = MemorySummary(config=config)
    memory.token_counter = SimpleTokenCounter(chars_per_token=1)
    return memory


class TestMemorySummary:

    def test_sandwich_construction(self, memory_summary):
        memory_summary.update_config({"before_raw_message": 1, "end_raw_message": -1})

        for i in range(5):
            memory_summary.add_message({"role": "user", "content": f"msg_{i}"})

        effective = memory_summary._get_effective_messages()

        assert len(effective) == 5
        assert effective[0]["content"] == "msg_0"
        assert effective[-1]["content"] == "msg_4"
        assert effective[1]["content"] == "msg_1"

    def test_is_overlength(self, memory_summary):
        memory_summary.update_config({"max_summary_length": 5, "max_prompt_length": 10})
        memory_summary.add_message({"role": "user", "content": "12345678901"})  # 11 chars

        assert memory_summary._is_overlength(memory_summary.get_messages())

    def test_find_summarization_start(self, memory_summary):
        memory_summary.add_message({"role": "system", "content": "sys"})
        memory_summary.add_message({"role": "user", "content": "u1"})

        start = memory_summary._find_summarization_start()
        assert start == 1

        memory_summary.add_message({"role": "summary", "content": "sum"}, insert_id=1)
        start = memory_summary._find_summarization_start()
        assert start == 2

    def test_format_summary_message(self, memory_summary):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "summary", "content": "previous_summary"},
            {"role": "user", "content": "u1"},
        ]

        formatted = memory_summary._format_summary_message(messages)

        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert "sys" in formatted[0]["content"]

        assert "previous_summary" in formatted[0]["content"]
        assert formatted[1]["role"] == "user"

    def test_summarization_flow(self, memory_summary):
        mock_client = MagicMock()
        mock_client.generate_chat_completion.return_value = "ShortSummary"
        memory_summary.chat_client = mock_client

        memory_summary.update_config({
            "max_summary_length": 5,
            "max_prompt_length": 20,
            "before_raw_message": 0,
            "end_raw_message": 0,
        })
        memory_summary.add_message({"role": "user", "content": "1234567890"})
        memory_summary.add_message({"role": "user", "content": "1234567890"})
        memory_summary.get_prompt_messages()

        mock_client.generate_chat_completion.assert_called()
        messages = memory_summary.get_messages()

        has_summary = any(m["role"] == SUMMARY for m in messages)
        assert has_summary

    @patch("agentic_rl.memory.memory_summary.SUMMARY_HISTORY_PREFIX", "")
    @patch("agentic_rl.memory.memory_summary.SUMMARY_HISTORY_SUFFIX", "")
    def test_recursive_summarization(self, memory_summary):
        mock_client = MagicMock()
        mock_client.generate_chat_completion.return_value = "Sum"
        memory_summary.chat_client = mock_client

        memory_summary.update_config({"max_summary_length": 5, "max_prompt_length": 20, "summary_system_prompt": "."})

        for _ in range(10):
            memory_summary.add_message({"role": "user", "content": "123456789"})

        memory_summary.get_prompt_messages()

        assert mock_client.generate_chat_completion.call_count > 1
        final_msgs = memory_summary.get_prompt_messages()
        assert memory_summary._get_total_length(final_msgs) <= memory_summary.config.max_prompt_length
