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
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestOzyTrajRefineReward:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_vaee_types = MagicMock()
        mock_vaee_types.Episode = MagicMock()
        mock_vaee_types.Trajectory = MagicMock()
        mock_vaee_types.Step = MagicMock()
        mock_vaee_types.RequestRecord = MagicMock()

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        mock_refine = MagicMock()
        mock_refine.extract_token_ids_and_logprobs = MagicMock(
            return_value=([0.1], [1, 2], [3, 4])
        )
        mock_refine.get_episode_summary = MagicMock(return_value="{}")
        mock_refine.default_traj_filter_func = MagicMock(side_effect=lambda tid, task, recs: recs)
        mock_refine.get_tool_outputs = MagicMock(return_value=[])
        mock_refine.get_action_from_assistant = MagicMock(
            return_value={"role": "assistant", "content": "hello"}
        )
        mock_refine.concat_history_message = MagicMock(side_effect=lambda x: x)

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
            "aura.runner.agent_engine_wrapper.vaee.default_traj_refine": mock_refine,
        }):
            yield

    def test_filter_agent_records_matching(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import _filter_agent_records

        task = {"problem": "test problem"}
        record = MagicMock()
        record.messages = [{"role": "user", "content": "test problem"}]

        result = _filter_agent_records("task_1", task, [record])
        assert len(result) == 1
        assert result[0] == record

    def test_filter_agent_records_mismatch(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import _filter_agent_records

        task = {"problem": "test problem"}
        record = MagicMock()
        record.messages = [{"role": "user", "content": "different problem"}]

        result = _filter_agent_records("task_1", task, [record])
        assert len(result) == 0

    def test_filter_agent_records_empty_messages(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import _filter_agent_records

        task = {"problem": "test problem"}
        record = MagicMock()
        record.messages = []

        result = _filter_agent_records("task_1", task, [record])
        assert len(result) == 0

    def test_filter_agent_records_no_user_message(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import _filter_agent_records

        task = {"problem": "test problem"}
        record = MagicMock()
        record.messages = [{"role": "system", "content": "sys"}]

        result = _filter_agent_records("task_1", task, [record])
        assert len(result) == 0

    def test_ozy_token_traj_refine_func(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import ozy_token_traj_refine_func

        task = {"problem": "test problem", "ground_truth": "42"}
        record = MagicMock()
        record.start_time = datetime.now()
        record.messages = [{"role": "user", "content": "test problem"}]
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

        episode = ozy_token_traj_refine_func(
            "task_1", task, [record], tokenizer=MagicMock()
        )
        assert episode is not None
        assert episode.id == "task_1"

    def test_ozy_token_traj_refine_func_dedup_messages(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import ozy_token_traj_refine_func

        task = {"problem": "test problem", "ground_truth": "42"}
        record1 = MagicMock()
        record1.start_time = datetime(2026, 1, 1, 0, 0, 0)
        record1.messages = [{"role": "user", "content": "test problem"}]
        record1.request_id = "req_1"
        record1.response_text = "response1"
        record1.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record1.token_ids = [1, 2, 3]
        record1.response_ids = [4, 5]
        record1.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        record2 = MagicMock()
        record2.start_time = datetime(2026, 1, 1, 0, 0, 1)
        record2.messages = [{"role": "user", "content": "test problem"}]
        record2.request_id = "req_2"
        record2.response_text = "response2"
        record2.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi2"}}],
        }
        record2.token_ids = [1, 2, 3]
        record2.response_ids = [6, 7]
        record2.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.3, 0.4]}}]
        }

        episode = ozy_token_traj_refine_func(
            "task_1", task, [record1, record2], tokenizer=MagicMock()
        )
        assert episode is not None

    def test_ozy_token_traj_refine_func_with_max_model_len(self):
        from agents.proxy_agent.math.ozy_traj_refine_reward import ozy_token_traj_refine_func

        task = {"problem": "test problem", "ground_truth": "42"}
        record = MagicMock()
        record.start_time = datetime.now()
        record.messages = [{"role": "user", "content": "test problem"}]
        record.request_id = "req_1"
        record.response_text = "response"
        record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record.token_ids = list(range(1000))
        record.response_ids = list(range(1000))
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1] * 1000}}]
        }

        episode = ozy_token_traj_refine_func(
            "task_1", task, [record], tokenizer=MagicMock(), max_model_len=500
        )
        assert episode is not None
