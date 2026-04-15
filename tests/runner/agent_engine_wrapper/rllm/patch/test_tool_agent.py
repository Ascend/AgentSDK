# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for rllm/patch/tool_agent module: _format_observation_as_messages_patch."""

import pytest

from agentic_rl.runner.agent_engine_wrapper.rllm.patch.tool_agent import (
    _format_observation_as_messages_patch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_self():
    """Minimal mock for the self parameter (ToolAgent instance)."""
    return object()


# ---------------------------------------------------------------------------
# Dict observation tests
# ---------------------------------------------------------------------------

class TestFormatObservationDictWithProblem:
    def test_problem_key(self, mock_self):
        obs = {"problem": "What is 2+2?"}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is 2+2?"

    def test_problem_empty_string(self, mock_self):
        obs = {"problem": ""}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["content"] == ""


class TestFormatObservationDictWithToolOutputs:
    def test_single_tool_output(self, mock_self):
        obs = {"tool_outputs": {"call_1": "result_1"}}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "result_1"
        assert result[0]["tool_call_id"] == "call_1"

    def test_multiple_tool_outputs(self, mock_self):
        obs = {"tool_outputs": {"call_1": "result_1", "call_2": "result_2"}}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 2
        ids = {msg["tool_call_id"] for msg in result}
        assert ids == {"call_1", "call_2"}

    def test_empty_tool_outputs(self, mock_self):
        obs = {"tool_outputs": {}}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []


class TestFormatObservationDictOther:
    def test_empty_dict(self, mock_self):
        obs = {}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []

    def test_dict_with_unknown_keys(self, mock_self):
        obs = {"unknown_key": "value"}
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []


# ---------------------------------------------------------------------------
# String observation tests
# ---------------------------------------------------------------------------

class TestFormatObservationString:
    def test_string_observation(self, mock_self):
        obs = "Hello, world!"
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_empty_string(self, mock_self):
        """Empty string is truthy for isinstance(str) but passes the elif branch."""
        obs = ""
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""


# ---------------------------------------------------------------------------
# Other type observation tests
# ---------------------------------------------------------------------------

class TestFormatObservationOtherTypes:
    def test_int_observation(self, mock_self):
        obs = 42
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "42"

    def test_list_observation(self, mock_self):
        obs = [1, 2, 3]
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[1, 2, 3]"

    def test_none_observation(self, mock_self):
        obs = None
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []

    def test_false_observation(self, mock_self):
        obs = False
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []

    def test_zero_observation(self, mock_self):
        obs = 0
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert result == []

    def test_true_observation(self, mock_self):
        obs = True
        result = _format_observation_as_messages_patch(mock_self, obs)
        assert len(result) == 1
        assert result[0]["content"] == "True"
