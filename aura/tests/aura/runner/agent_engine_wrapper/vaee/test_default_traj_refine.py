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

import os
import sys
from copy import deepcopy
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestDefaultTrajRefine:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_vaee_types = MagicMock()
        mock_vaee_types.Episode = MagicMock()
        mock_vaee_types.Trajectory = MagicMock()
        mock_vaee_types.Step = MagicMock()
        mock_vaee_types.RequestRecord = MagicMock()

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
        }):
            yield

    def test_preprocess_messages_arguments_str2json(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import preprocess_messages_arguments_str2json

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "test", "arguments": '{"key": "value"}'}}
                ],
            }
        ]
        result = preprocess_messages_arguments_str2json(messages)
        assert isinstance(result[0]["tool_calls"][0]["function"]["arguments"], dict)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}

    def test_preprocess_messages_arguments_invalid_json(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import preprocess_messages_arguments_str2json

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "test", "arguments": "not valid json"}}
                ],
            }
        ]
        result = preprocess_messages_arguments_str2json(messages)
        assert isinstance(result[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_preprocess_messages_no_tool_calls(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import preprocess_messages_arguments_str2json

        messages = [{"role": "user", "content": "hello"}]
        result = preprocess_messages_arguments_str2json(messages)
        assert result[0] == {"role": "user", "content": "hello"}

    def test_extract_logprobs_from_content(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_logprobs

        record = MagicMock()
        record.raw_response = {
            "choices": [{"logprobs": {"content": [{"logprob": 0.1}, {"logprob": 0.2}]}}]
        }
        result = extract_logprobs(record)
        assert result == [0.1, 0.2]

    def test_extract_logprobs_from_token_logprobs(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_logprobs

        record = MagicMock()
        record.raw_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.3, 0.4]}}]
        }
        result = extract_logprobs(record)
        assert result == [0.3, 0.4]

    def test_extract_logprobs_empty(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_logprobs

        record = MagicMock()
        record.raw_response = {"choices": [{}]}
        result = extract_logprobs(record)
        assert result == []

    def test_get_action_from_assistant(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_action_from_assistant

        record = MagicMock()
        record.raw_response = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        record.response_text = "hello"

        result = get_action_from_assistant(record)
        assert result == {"role": "assistant", "content": "hello"}

    def test_get_action_from_assistant_none_message(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_action_from_assistant

        record = MagicMock()
        record.raw_response = {"choices": [{"message": None}]}
        record.response_text = "fallback"

        result = get_action_from_assistant(record)
        assert result == "fallback"

    def test_get_tool_outputs(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_tool_outputs

        record = MagicMock()
        next_record = MagicMock()
        next_record.raw_request = {
            "messages": [
                {"role": "assistant", "content": "hello"},
                {"role": "tool", "content": "result1", "tool_call_id": "1"},
                {"role": "tool", "content": "result2", "tool_call_id": "2"},
            ]
        }

        result = get_tool_outputs(record, next_record)
        assert len(result) == 2
        assert result[0]["content"] == "result2"
        assert result[1]["content"] == "result1"

    def test_get_tool_outputs_none_next(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_tool_outputs

        record = MagicMock()
        result = get_tool_outputs(record, None)
        assert result == []

    def test_get_first_message(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_first_message

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        content, msg = get_first_message("user", messages)
        assert content == "hello"
        assert msg["role"] == "user"

    def test_get_first_message_not_found(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_first_message

        messages = [{"role": "system", "content": "sys"}]
        content, msg = get_first_message("user", messages)
        assert content is None
        assert msg is None

    def test_get_first_message_reversed(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_first_message

        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "last"},
        ]
        content, msg = get_first_message("user", messages, is_reversed=True)
        assert content == "last"

    def test_get_episode_summary(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import get_episode_summary

        step1 = MagicMock()
        step1.id = "step_1"
        step2 = MagicMock()
        step2.id = "step_2"

        traj = MagicMock()
        traj.steps = [step1, step2]

        episode = MagicMock()
        episode.id = "ep_1"
        episode.trajectories = [traj]

        result = get_episode_summary(episode)
        assert "ep_1" in result
        assert "step_1" in result
        assert "step_2" in result

    def test_default_traj_filter_func(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import default_traj_filter_func

        task = {"problem": "test problem"}

        record1 = MagicMock()
        record1.response_text = "normal response"
        record1.raw_response = {"choices": [{"message": {"content": "ok"}}]}
        record1.raw_request = {"messages": [{"role": "user", "content": "test problem"}]}
        record1.messages = [{"role": "user", "content": "test problem"}]

        record2 = MagicMock()
        record2.response_text = "\n\nSAFE"
        record2.raw_response = {"choices": [{"message": {"content": "ok"}}]}
        record2.raw_request = {"messages": [{"role": "user", "content": "test problem"}]}
        record2.messages = [{"role": "user", "content": "test problem"}]

        records = [record1, record2]
        result = default_traj_filter_func("task_1", task, records)
        assert len(result) == 1
        assert result[0] == record1

    def test_default_traj_filter_func_subagent(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import default_traj_filter_func

        task = {"problem": "main problem"}

        record = MagicMock()
        record.response_text = "response"
        record.raw_response = {"choices": [{"message": {"content": "ok"}}]}
        record.raw_request = {"messages": [{"role": "user", "content": "sub problem"}]}
        record.messages = [{"role": "user", "content": "sub problem"}]

        records = [record]
        result = default_traj_filter_func("task_1", task, records)
        assert len(result) == 0

    def test_default_traj_filter_func_none_raw_response(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import default_traj_filter_func

        task = {"problem": "test problem"}
        record = MagicMock()
        record.response_text = "response"
        record.raw_response = None

        records = [record]
        result = default_traj_filter_func("task_1", task, records)
        assert len(result) == 0

    def test_concat_history_message(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import concat_history_message

        record1 = MagicMock()
        record1.messages = [{"role": "user", "content": "hello"}]
        record1.raw_response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        record1.request_id = "req_1"

        record2 = MagicMock()
        record2.messages = [
            {"role": "tool", "content": "result", "tool_call_id": "1"},
            {"role": "assistant", "content": "followup"},
        ]
        record2.raw_response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        record2.request_id = "req_2"

        records = [record1, record2]
        result = concat_history_message(records)
        assert len(result) == 2

    def test_concat_history_message_empty(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import concat_history_message

        result = concat_history_message([])
        assert result == []

    def test_extract_token_ids_and_logprobs_tito_mode(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_token_ids_and_logprobs

        record = MagicMock()
        record.token_ids = [1, 2, 3]
        record.response_ids = [4, 5]
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(
            record, tokenizer=MagicMock()
        )
        assert prompt_ids == [1, 2, 3]
        assert response_ids == [4, 5]
        assert logprobs == [0.1, 0.2]

    def test_extract_token_ids_and_logprobs_non_tito_mode(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import extract_token_ids_and_logprobs

        record = MagicMock()
        record.token_ids = None
        record.response_ids = None
        record.token_response = None
        record.messages = [{"role": "user", "content": "hello"}]
        record.raw_request = {"tools": None}
        record.raw_response = {
            "prompt_token_ids": [1, 2, 3],
            "choices": [{
                "token_ids": [4, 5],
                "logprobs": {"content": [{"logprob": 0.1}, {"logprob": 0.2}]},
            }],
        }

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = [
            [1, 2, 3],           # First call: prompt_ids
            [1, 2, 3, 4, 5],     # Second call: full messages
        ]

        logprobs, prompt_ids, response_ids = extract_token_ids_and_logprobs(
            record, tokenizer
        )
        assert prompt_ids == [1, 2, 3]
        assert response_ids == [4, 5]
        assert logprobs == [0.1, 0.2]

    def test_default_step_traj_refine_func(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import default_step_traj_refine_func

        task = {"problem": "test"}
        record = MagicMock()
        record.start_time = datetime.now()
        record.messages = [{"role": "user", "content": "hello"}]
        record.request_id = "req_1"
        record.response_text = "response"
        record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record.token_ids = [1, 2, 3]
        record.response_ids = [4, 5]
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        tokenizer = MagicMock()
        episode = default_step_traj_refine_func("task_1", task, [record], tokenizer)
        assert episode is not None

    def test_default_token_traj_refine_func(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import default_token_traj_refine_func

        task = {"problem": "test"}
        record = MagicMock()
        record.messages = [{"role": "user", "content": "hello"}]
        record.request_id = "req_1"
        record.response_text = "response"
        record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record.token_ids = [1, 2, 3]
        record.response_ids = [4, 5]
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        tokenizer = MagicMock()
        episode = default_token_traj_refine_func("task_1", task, [record], tokenizer)
        assert episode is not None

    def test_check_token_ids_and_messages_disabled(self):
        os.environ["ENABLE_MESSAGE_DIFF_CHECK"] = "false"
        from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import check_token_ids_and_messages

        record = MagicMock()
        tokenizer = MagicMock()
        check_token_ids_and_messages(record, tokenizer)
        # Should return early without error
