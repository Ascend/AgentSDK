#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestSystemPrompts:
    """Tests for system_prompts module."""

    def test_system_miniwob_prompt_exists(self):
        """Test SYSTEM_MINIWOB_PROMPT exists."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_MINIWOB_PROMPT
        
        assert SYSTEM_MINIWOB_PROMPT is not None
        assert isinstance(SYSTEM_MINIWOB_PROMPT, str)
        assert len(SYSTEM_MINIWOB_PROMPT) > 0

    def test_system_miniwob_prompt_contains_key_elements(self):
        """Test SYSTEM_MINIWOB_PROMPT contains key elements."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_MINIWOB_PROMPT
        
        assert "Action" in SYSTEM_MINIWOB_PROMPT
        assert "Observation" in SYSTEM_MINIWOB_PROMPT
        assert "Thought" in SYSTEM_MINIWOB_PROMPT

    def test_system_miniwob_prompt_without_thought_exists(self):
        """Test SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT exists."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT
        
        assert SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT is not None
        assert isinstance(SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT, str)

    def test_system_miniwob_prompt_without_thought_format(self):
        """Test SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT format."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT
        
        assert "Action:" in SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT
        assert "Thought:" not in SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT

    def test_system_webarena_prompt_exists(self):
        """Test SYSTEM_WEBARENA_PROMPT exists."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_WEBARENA_PROMPT
        
        assert SYSTEM_WEBARENA_PROMPT is not None
        assert isinstance(SYSTEM_WEBARENA_PROMPT, str)
        assert len(SYSTEM_WEBARENA_PROMPT) > 0

    def test_system_webarena_prompt_contains_actions(self):
        """Test SYSTEM_WEBARENA_PROMPT contains action types."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_WEBARENA_PROMPT
        
        assert "Click" in SYSTEM_WEBARENA_PROMPT
        assert "Type" in SYSTEM_WEBARENA_PROMPT
        assert "Scroll" in SYSTEM_WEBARENA_PROMPT
        assert "Wait" in SYSTEM_WEBARENA_PROMPT
        assert "GoBack" in SYSTEM_WEBARENA_PROMPT
        assert "ANSWER" in SYSTEM_WEBARENA_PROMPT

    def test_swe_system_prompt_fn_call_exists(self):
        """Test SWE_SYSTEM_PROMPT_FN_CALL exists."""
        from agents.math_agent.prompt.system_prompts import SWE_SYSTEM_PROMPT_FN_CALL
        
        assert SWE_SYSTEM_PROMPT_FN_CALL is not None
        assert isinstance(SWE_SYSTEM_PROMPT_FN_CALL, str)

    def test_swe_system_prompt_fn_call_content(self):
        """Test SWE_SYSTEM_PROMPT_FN_CALL content."""
        from agents.math_agent.prompt.system_prompts import SWE_SYSTEM_PROMPT_FN_CALL
        
        assert "programming agent" in SWE_SYSTEM_PROMPT_FN_CALL.lower()
        assert "github" in SWE_SYSTEM_PROMPT_FN_CALL.lower()

    def test_swe_system_prompt_exists(self):
        """Test SWE_SYSTEM_PROMPT exists."""
        from agents.math_agent.prompt.system_prompts import SWE_SYSTEM_PROMPT
        
        assert SWE_SYSTEM_PROMPT is not None
        assert isinstance(SWE_SYSTEM_PROMPT, str)

    def test_swe_system_prompt_contains_function_info(self):
        """Test SWE_SYSTEM_PROMPT contains function information."""
        from agents.math_agent.prompt.system_prompts import SWE_SYSTEM_PROMPT
        
        assert "function" in SWE_SYSTEM_PROMPT.lower()
        assert "file_editor" in SWE_SYSTEM_PROMPT.lower()

    def test_prompts_are_not_empty(self):
        """Test all prompts are non-empty strings."""
        from agents.math_agent.prompt.system_prompts import (
            SYSTEM_MINIWOB_PROMPT,
            SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT,
            SYSTEM_WEBARENA_PROMPT,
            SWE_SYSTEM_PROMPT_FN_CALL,
            SWE_SYSTEM_PROMPT,
        )
        
        prompts = [
            SYSTEM_MINIWOB_PROMPT,
            SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT,
            SYSTEM_WEBARENA_PROMPT,
            SWE_SYSTEM_PROMPT_FN_CALL,
            SWE_SYSTEM_PROMPT,
        ]
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_miniwob_prompts_difference(self):
        """Test that MINIWOB prompts with and without thought are different."""
        from agents.math_agent.prompt.system_prompts import (
            SYSTEM_MINIWOB_PROMPT,
            SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT,
        )
        
        assert SYSTEM_MINIWOB_PROMPT != SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT

    def test_webarena_prompt_contains_guidelines(self):
        """Test SYSTEM_WEBARENA_PROMPT contains guidelines."""
        from agents.math_agent.prompt.system_prompts import SYSTEM_WEBARENA_PROMPT
        
        assert "Guidelines" in SYSTEM_WEBARENA_PROMPT or "guidelines" in SYSTEM_WEBARENA_PROMPT.lower()
