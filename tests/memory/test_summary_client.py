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

import openai
import pytest

from agentic_rl.memory.summary_client import SummaryClient, SummaryGenerationError


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=openai.OpenAI)


@pytest.fixture
def summary_client(mock_openai_client):
    return SummaryClient(client=mock_openai_client, model_name="test-model")


class TestSummaryClient:

    def test_init_validation(self):
        valid_client = MagicMock(spec=openai.OpenAI)

        with pytest.raises(TypeError, match="client must be an instance of OpenAI"):
            SummaryClient(client=None, model_name="test-model")

        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            SummaryClient(client=valid_client, model_name="")

        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            SummaryClient(client=valid_client, model_name=None)

    @pytest.mark.parametrize(
        "invalid_param, value, expected_msg_part",
        [
            ("max_retries", -1, "max_retries must be a positive integer"),
            ("max_retries", 0, "max_retries must be a positive integer"),
            ("retry_delay", 0, "retry_delay must be a positive integer"),
            ("timeout", 0, "timeout must be a positive integer"),
            ("messages", "not a list", "messages must be a list of dictionaries"),
            ("messages", [{"role": "user"}], "messages must be a list of dictionaries"),
        ]
    )
    def test_parameter_validation(self, summary_client, invalid_param, value, expected_msg_part):
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "max_retries": 3, "retry_delay": 5, "timeout": 3600}
        kwargs[invalid_param] = value

        with pytest.raises(ValueError, match=expected_msg_part):
            summary_client.generate_chat_completion(**kwargs)

    @patch("agentic_rl.memory.summary_client.time.sleep")
    def test_retry_mechanism(self, mock_sleep, summary_client):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Success"))]

        summary_client.client.chat.completions.create.side_effect = [
            openai.RateLimitError(message="Rate limit", response=MagicMock(), body=None),
            openai.APITimeoutError(request=MagicMock()),
            mock_completion,
        ]

        response = summary_client.generate_chat_completion([{"role": "user", "content": "hi"}])

        assert response == "Success"
        assert summary_client.client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    def test_api_error_handling(self, summary_client):
        summary_client.client.chat.completions.create.side_effect = Exception("Generic Error")

        with pytest.raises(SummaryGenerationError, match="Unexpected error"):
            summary_client.generate_chat_completion([{"role": "user", "content": "hi"}], max_retries=1)

    def test_update_config(self, summary_client):
        new_client = MagicMock(spec=openai.OpenAI)
        summary_client.update_config(new_client, "new-model")
        assert summary_client.client == new_client
        assert summary_client.model_name == "new-model"

        with pytest.raises(TypeError):
            summary_client.update_config(None, "model")
