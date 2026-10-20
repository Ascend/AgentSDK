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
from unittest.mock import MagicMock, patch

import pytest


class TestFilterTrajectorySteps:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_vaee_types = MagicMock()
        mock_vaee_types.Episode = MagicMock()
        mock_vaee_types.Step = MagicMock()
        mock_vaee_types.RequestRecord = MagicMock()
        # Make Trajectory() return an object with a real steps list
        mock_vaee_types.Trajectory = MagicMock()
        mock_vaee_types.Trajectory.side_effect = lambda *a, **kw: MagicMock(steps=kw.get("steps", []))

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        mock_default_traj_refine = MagicMock()
        mock_default_traj_refine.extract_token_ids_and_logprobs = MagicMock(
            return_value=([0.1], [1, 2], [3, 4])
        )
        mock_default_traj_refine.get_tool_outputs = MagicMock(return_value=[])
        mock_default_traj_refine.get_action_from_assistant = MagicMock(
            return_value={"role": "assistant", "content": "hello"}
        )
        mock_default_traj_refine.concat_history_message = MagicMock(
            side_effect=lambda x: x
        )

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
            "aura.runner.agent_engine_wrapper.vaee.default_traj_refine": mock_default_traj_refine,
        }):
            yield

    def test_get_no_compress_steps_in_step_trajectories_single(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            get_no_compress_steps_in_step_trajectories,
        )

        task = {"problem": "test"}
        record = MagicMock()
        record.request_id = "req_1"
        record.messages = [{"role": "user", "content": "hello"}]
        record.response_text = "response"
        record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record.token_ids = [1, 2, 3]
        record.response_ids = [4, 5]
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        trajectories = get_no_compress_steps_in_step_trajectories(
            [record], task, tokenizer=MagicMock()
        )
        assert len(trajectories) == 1
        assert len(trajectories[0].steps) == 1

    def test_get_no_compress_steps_in_step_trajectories_multi(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            get_no_compress_steps_in_step_trajectories,
        )

        task = {"problem": "test"}
        record1 = MagicMock()
        record1.request_id = "req_1"
        record1.messages = [{"role": "user", "content": "q1"}]
        record1.response_text = "r1"
        record1.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "a1"}}],
        }
        record1.token_ids = [1, 2]
        record1.response_ids = [3, 4]
        record1.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        record2 = MagicMock()
        record2.request_id = "req_2"
        record2.messages = [{"role": "user", "content": "q2"}]
        record2.response_text = "r2"
        record2.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "a2"}}],
        }
        record2.token_ids = [5, 6]
        record2.response_ids = [7, 8]
        record2.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.3, 0.4]}}]
        }

        trajectories = get_no_compress_steps_in_step_trajectories(
            [record1, record2], task, tokenizer=MagicMock()
        )
        assert len(trajectories) >= 1

    def test_is_messages_match_in_true(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            is_messages_match_in,
        )

        last_record = MagicMock()
        last_record.messages = [{"role": "user", "content": "hello"}]
        last_record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }

        record = MagicMock()
        record.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "result"},
        ]

        assert is_messages_match_in(last_record, record) is True

    def test_is_messages_match_in_false(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            is_messages_match_in,
        )

        last_record = MagicMock()
        last_record.messages = [{"role": "user", "content": "hello"}]
        last_record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }

        record = MagicMock()
        record.messages = [{"role": "user", "content": "different"}]

        assert is_messages_match_in(last_record, record) is False

    def test_prefix_message_match_check_exact(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        assert prefix_message_match_check([1, 2, 3], [4, 5], [1, 2, 3, 4, 5, 6]) is True

    def test_prefix_message_match_check_shorter(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        assert prefix_message_match_check([1, 2, 3], [4, 5], [1, 2]) is False

    def test_prefix_message_match_check_mismatch(self):
        os.environ["ENABLE_MESSAGE_DIFF_CHECK"] = "false"
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        assert prefix_message_match_check([1, 2, 3], [4, 5], [1, 2, 3, 9, 9]) is False

    def test_prefix_message_match_check_tolerance_case_a(self):
        os.environ["ENABLE_MESSAGE_DIFF_CHECK"] = "true"
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        # Case A: 1-to-1 replacement (99 -> 100, but next tokens match)
        assert prefix_message_match_check(
            [1, 2, 99, 3], [4, 5], [1, 2, 100, 3, 4, 5]
        ) is True

    def test_prefix_message_match_check_tolerance_case_b(self):
        os.environ["ENABLE_MESSAGE_DIFF_CHECK"] = "true"
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        # Case B: next_prompt shorter than full_prev due to 2-to-1 merge
        # The function rejects shorter next_prompt by design
        assert prefix_message_match_check(
            [1, 2, 99, 100, 3], [4, 5], [1, 2, 999, 3, 4, 5]
        ) is False

    def test_prefix_message_match_check_tolerance_case_c(self):
        os.environ["ENABLE_MESSAGE_DIFF_CHECK"] = "true"
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            prefix_message_match_check,
        )

        # Case C: 1-to-2 split (99 -> 999, 1000, next tokens match)
        assert prefix_message_match_check(
            [1, 2, 99, 3], [4, 5], [1, 2, 999, 1000, 3, 4, 5]
        ) is True

    def test_get_no_compress_steps_in_token_trajectories(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            get_no_compress_steps_in_token_trajectories,
        )

        task = {"problem": "test"}
        record = MagicMock()
        record.request_id = "req_1"
        record.messages = [{"role": "user", "content": "hello"}]
        record.response_text = "response"
        record.raw_response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        record.token_ids = [1, 2, 3]
        record.response_ids = [4, 5]
        record.token_response = {
            "choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]
        }

        trajectories = get_no_compress_steps_in_token_trajectories(
            [record], task, tokenizer=MagicMock()
        )
        assert len(trajectories) == 1

    def test_convert_step_traj_to_token_traj(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            convert_step_traj_to_token_traj,
        )

        step1 = MagicMock()
        step1.prompt_ids = [1, 2]
        step1.response_ids = [3, 4]
        step1.logprobs = [0.1, 0.2]

        step2 = MagicMock()
        step2.prompt_ids = [1, 2, 3, 4, 5, 6]
        step2.response_ids = [7, 8]
        step2.logprobs = [0.3, 0.4]

        result = convert_step_traj_to_token_traj([step1, step2], task_id="task_1")
        assert result is not None

    def test_compress_trajectories_steps(self):
        from aura.runner.agent_engine_wrapper.vaee.filter_trajectory_steps import (
            compress_trajectories_steps,
        )

        step1 = MagicMock()
        step1.prompt_ids = [1, 2]
        step1.response_ids = [3, 4]
        step1.logprobs = [0.1, 0.2]

        traj = MagicMock()
        traj.steps = [step1]

        result = compress_trajectories_steps([traj], task_id="task_1")
        assert result is not None
