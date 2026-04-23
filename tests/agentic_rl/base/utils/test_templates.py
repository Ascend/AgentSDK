#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Co.,Ltd.
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
import pytest
import unittest.mock as mock

from agentic_rl.base.utils.templates import (
    Role,
    infer_max_len,
    Template,
    Llama2Template,
    get_templates,
    get_model_template,
    _register_template,
    register_custom_template,
    _format_custom_template
)
from agentic_rl.base.utils.formatter import (
    StringFormatter,
    ToolFormatter
)


class TestRoleEnum:
    def test_role_values(self):
        """Test that Role enum has correct values."""
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"
        assert Role.FUNCTION.value == "function"
        assert Role.OBSERVATION.value == "observation"


class TestInferMaxLen:
    def test_infer_max_len_basic(self):
        """Test basic functionality of infer_max_len."""
        # Test with equal source and target lengths
        max_source, max_target = infer_max_len(100, 100, 200, 1)
        assert max_source == 100
        assert max_target == 100
        
        # Test with different lengths
        max_source, max_target = infer_max_len(150, 50, 200, 1)
        assert max_source == 150
        assert max_target == 50
        
        # Test with reserved_label_len
        max_source, max_target = infer_max_len(190, 5, 200, 10)
        assert max_source == 195  # 200 - min(10, 5) = 195
        assert max_target == 10  # Should be at least reserved_label_len
        
        # Test with zero lengths
        max_source, max_target = infer_max_len(0, 0, 200, 1)
        assert max_source == 200  # max_len - min(max_target_len, target_len) = 200 - 0 = 200
        assert max_target == 1  # Even with zero lengths, max_target_len is at least reserved_label_len
        
        # Test with large reserved_label_len
        max_source, max_target = infer_max_len(100, 100, 200, 150)
        assert max_source == 100  # 200 - min(150, 100) = 100
        assert max_target == 150  # Should be at least reserved_label_len
        
        # Test with source_len=0 and target_len>0
        max_source, max_target = infer_max_len(0, 50, 200, 10)
        assert max_target == 200  # When source_len=0, target_len/(source_len+target_len)=1, so max_target_len=200
        assert max_source == 150  # 200 - min(200, 50) = 200 - 50 = 150
        
        # Test with target_len=0 and source_len>0
        max_source, max_target = infer_max_len(50, 0, 200, 10)
        assert max_target == 10  # When target_len=0, max_target_len is set to reserved_label_len
        assert max_source == 200  # 200 - min(10, 0) = 200 - 0 = 200


class TestTemplate:
    def setup_method(self):
        """Set up common fixtures for Template tests."""
        self.mock_tokenizer = mock.Mock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.convert_tokens_to_ids.return_value = 4
        self.mock_tokenizer.bos_token_id = 5
        self.mock_tokenizer.eos_token_id = 6
        
        # Create mock formatters
        self.mock_format_user = mock.Mock()
        self.mock_format_user.apply.return_value = ["user content"]
        
        self.mock_format_assistant = mock.Mock()
        self.mock_format_assistant.apply.return_value = ["assistant content"]
        
        self.mock_format_system = mock.Mock()
        self.mock_format_system.apply.return_value = ["system content"]
        
        self.mock_format_function = mock.Mock()
        self.mock_format_function.apply.return_value = ["function content"]
        
        self.mock_format_observation = mock.Mock()
        self.mock_format_observation.apply.return_value = ["observation content"]
        
        self.mock_format_tools = mock.Mock()
        self.mock_format_tools.apply.return_value = ["tools content"]
        
        self.mock_format_separator = mock.Mock()
        self.mock_format_separator.apply.return_value = ["separator"]
        
        self.mock_format_prefix = mock.Mock()
        self.mock_format_prefix.apply.return_value = ["prefix"]
        
        # Create template instance
        self.template = Template(
            format_user=self.mock_format_user,
            format_assistant=self.mock_format_assistant,
            format_system=self.mock_format_system,
            format_function=self.mock_format_function,
            format_observation=self.mock_format_observation,
            format_tools=self.mock_format_tools,
            format_separator=self.mock_format_separator,
            format_prefix=self.mock_format_prefix,
            default_system="default system",
            stop_words=["stop1", "stop2"],
            efficient_eos=False,
            replace_eos=False,
            force_system=False
        )
    
    def test_template_initialization(self):
        """Test Template initialization and properties."""
        # Verify all properties are set correctly
        assert self.template.format_user == self.mock_format_user
        assert self.template.format_assistant == self.mock_format_assistant
        assert self.template.format_system == self.mock_format_system
        assert self.template.format_function == self.mock_format_function
        assert self.template.format_observation == self.mock_format_observation
        assert self.template.format_tools == self.mock_format_tools
        assert self.template.format_separator == self.mock_format_separator
        assert self.template.format_prefix == self.mock_format_prefix
        assert self.template.default_system == "default system"
        assert self.template.stop_words == ["stop1", "stop2"]
        assert self.template.efficient_eos is False
        assert self.template.replace_eos is False
        assert self.template.force_system is False
    
    def test_convert_elements_to_ids(self):
        """Test _convert_elements_to_ids method."""
        # Test with string elements
        elements = ["hello", "world"]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [1, 2, 3, 1, 2, 3]
        assert self.mock_tokenizer.encode.call_count == 2
        
        # Test with dict elements
        elements = [{"token": "test"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [4]
        self.mock_tokenizer.convert_tokens_to_ids.assert_called_with("test")
        
        # Test with set elements (bos_token)
        elements = [{"bos_token"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [5]
        
        # Test with set elements (eos_token)
        elements = [{"eos_token"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [6]
        
        # Test with mixed elements
        elements = ["hello", {"token": "test"}, {"bos_token"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [1, 2, 3, 4, 5]
        
        # Test with empty string
        elements = [""]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == []
        
        # Test with invalid type
        elements = [123]
        with pytest.raises(ValueError):
            self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        
        # Test with None token in dict
        elements = [{"token": None}]
        self.mock_tokenizer.convert_tokens_to_ids.return_value = None
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [None]  # Current code allows None to be added to token_ids list
        
        # Test with empty set
        elements = [set()]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == []
        
        # Test with None bos_token_id
        self.mock_tokenizer.bos_token_id = None
        elements = [{"bos_token"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == []
        
        # Test with None eos_token_id
        self.mock_tokenizer.eos_token_id = None
        elements = [{"eos_token"}]
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == []
        
        # Test with dict without token key
        elements = [{"not_token": "test"}]
        self.mock_tokenizer.convert_tokens_to_ids.return_value = None
        result = self.template._convert_elements_to_ids(self.mock_tokenizer, elements)
        assert result == [None]  # Current code allows None to be added to token_ids list
    
    def test_make_pairs(self):
        """Test _make_pairs method."""
        # Test basic pairing
        encoded_messages = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        result = self.template._make_pairs(encoded_messages, 100, 1)
        assert len(result) == 2
        assert result[0] == ([1, 2, 3], [4, 5, 6])
        assert result[1] == ([7, 8, 9], [10, 11, 12])
        
        # Test with cutoff
        encoded_messages = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        result = self.template._make_pairs(encoded_messages, 10, 1)
        assert len(result) == 1  # Only first pair fits
        
        # Test with exact cutoff
        encoded_messages = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        result = self.template._make_pairs(encoded_messages, 10, 1)
        assert len(result) == 1
        
        # Test with zero cutoff_len
        encoded_messages = [[1, 2, 3], [4, 5, 6]]
        result = self.template._make_pairs(encoded_messages, 0, 1)
        assert len(result) == 0
        
        # Test with odd number of messages
        encoded_messages = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # Current code throws IndexError when processing odd number of messages because it tries to access a non-existent index
        with pytest.raises(IndexError):
            self.template._make_pairs(encoded_messages, 100, 1)
        
        # Test with empty messages
        encoded_messages = [[], []]
        result = self.template._make_pairs(encoded_messages, 10, 1)
        assert len(result) == 1
        assert result[0] == ([], [])
        
        # Test with total_length exceeding cutoff after first pair
        encoded_messages = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
        result = self.template._make_pairs(encoded_messages, 15, 1)
        assert len(result) == 2  # Current code can accommodate two pairs: first pair length 10, second pair truncated to length 5 (3+2), total 15
        assert result[0] == ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])  # First pair is complete
        assert result[1] == ([11, 12, 13], [16, 17])  # Second pair is truncated
        
    def test_encode_single_turn(self):
        """Test encode_oneturn method."""
        # Mock _encode to return predefined pairs
        mock_pairs = [([1, 2], [3, 4]), ([5, 6], [7, 8])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            prompt_ids, answer_ids = self.template.encode_oneturn(
                self.mock_tokenizer,
                [{}, {}]
            )
            
            # Should concatenate all pairs except last answer
            assert prompt_ids == [1, 2, 3, 4, 5, 6]
            assert answer_ids == [7, 8]
        
        # Test with custom parameters
        mock_pairs = [([9, 10], [11, 12])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            prompt_ids, answer_ids = self.template.encode_oneturn(
                self.mock_tokenizer,
                [{}, {}],
                system="Custom system",
                tools="Custom tools",
                cutoff_len=500,
                reserved_label_len=5
            )
            
            assert prompt_ids == [9, 10]
            assert answer_ids == [11, 12]
        
        # Test with single turn
        mock_pairs = [([13, 14], [15, 16])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            prompt_ids, answer_ids = self.template.encode_oneturn(
                self.mock_tokenizer,
                [{}, {}]
            )
            
            assert prompt_ids == [13, 14]
            assert answer_ids == [15, 16]
    
    def test_encode_multi_turn(self):
        """Test encode_multiturn method."""
        # Mock _encode to return predefined pairs
        mock_pairs = [([1, 2], [3, 4]), ([5, 6], [7, 8])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            result = self.template.encode_multiturn(
                self.mock_tokenizer,
                [{}, {}, {}, {}]
            )
            
            # Should return the exact pairs from _encode
            assert result == mock_pairs
        
        # Test with custom parameters
        mock_pairs = [([9, 10], [11, 12]), ([13, 14], [15, 16])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            result = self.template.encode_multiturn(
                self.mock_tokenizer,
                [{}, {}, {}, {}],
                system="Custom system",
                tools="Custom tools",
                cutoff_len=500,
                reserved_label_len=5
            )
            
            assert result == mock_pairs
        
        # Test with single turn
        mock_pairs = [([17, 18], [19, 20])]
        with mock.patch.object(self.template, '_encode', return_value=mock_pairs):
            result = self.template.encode_multiturn(
                self.mock_tokenizer,
                [{}, {}]
            )
            
            assert result == mock_pairs
    
    def test_encode_method(self):
        """Test Template._encode method with various message types and scenarios."""
        # Test with different message types
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thanks!"}
        ]
        
        # Mock _make_pairs to return predefined pairs
        mock_pairs = [([1, 2], [3, 4]), ([5, 6], [7, 8])]
        with mock.patch.object(self.template, '_make_pairs', return_value=mock_pairs):
            result = self.template._encode(
                self.mock_tokenizer,
                messages,
                "Test system",
                None,
                1000,
                1
            )
            
            assert result == mock_pairs
            
            # Verify format_prefix was called for first turn
            self.mock_format_prefix.apply.assert_called_once()
            
            # Verify system was processed
            self.mock_format_system.apply.assert_called_once_with(content="Test system")
            
            # Verify user and assistant formats were applied twice each
            assert self.mock_format_user.apply.call_count == 2
            assert self.mock_format_assistant.apply.call_count == 2
            
            # Verify separator was called once (between turns)
            self.mock_format_separator.apply.assert_called_once()
        
        # Test with observation and function roles
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_user.apply.reset_mock()
        self.mock_format_assistant.apply.reset_mock()
        self.mock_format_separator.apply.reset_mock()
        
        messages = [
            {"role": "user", "content": "Call a function"},
            {"role": "assistant", "content": "Function called"},
            {"role": "function", "content": "Function result"},
            {"role": "observation", "content": "Observation result"}
        ]
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                None,
                None,
                1000,
                1
            )
            
            # Verify function and observation formats were applied
            self.mock_format_function.apply.assert_called_once_with(content="Function result")
            self.mock_format_observation.apply.assert_called_once_with(content="Observation result")
        
        # Test with unknown role (should raise NotImplementedError)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "unknown", "content": "Unknown role"}
        ]
        
        with pytest.raises(NotImplementedError):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                None,
                None,
                1000,
                1
            )
        
        # Test with system and tools both provided
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                "System prompt",
                "Tools information",
                1000,
                1
            )
            
            # Verify system and tools were processed
            self.mock_format_tools.apply.assert_called_with(content="Tools information")
            self.mock_format_system.apply.assert_called_once()
        
        # Test with only tools provided
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                None,
                "Tools only",
                1000,
                1
            )
            
            # Verify tools were processed even without system
            self.mock_format_tools.apply.assert_called_with(content="Tools only")
            self.mock_format_system.apply.assert_called_once()
        
        # Test with no system and no tools
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                None,
                None,
                1000,
                1
            )
            
            # Current code uses self.default_system when system is None, so format_system.apply is called
            self.mock_format_system.apply.assert_called_once_with(content="default system")
            # When tools is None, format_tools.apply is not called
            self.mock_format_tools.apply.assert_not_called()
        
        # Test with even and odd message counts
        # 重置所有相关mock对象的状态
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_user.apply.reset_mock()
        self.mock_format_assistant.apply.reset_mock()
        self.mock_format_function.apply.reset_mock()
        self.mock_format_observation.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        self.mock_format_separator.apply.reset_mock()
        
        messages_odd = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages_odd,
                None,
                None,
                1000,
                1
            )
            
            # Verify formats were applied correctly
            assert self.mock_format_user.apply.call_count == 2
            assert self.mock_format_assistant.apply.call_count == 1


class TestLlama2Template(TestTemplate):
    def setup_method(self):
        """Set up common fixtures for Llama2Template tests."""
        super().setup_method()
        
        # Create Llama2Template instance
        self.template = Llama2Template(
            format_user=self.mock_format_user,
            format_assistant=self.mock_format_assistant,
            format_system=self.mock_format_system,
            format_function=self.mock_format_function,
            format_observation=self.mock_format_observation,
            format_tools=self.mock_format_tools,
            format_separator=self.mock_format_separator,
            format_prefix=self.mock_format_prefix,
            default_system="default system",
            stop_words=["stop1", "stop2"],
            efficient_eos=False,
            replace_eos=False,
            force_system=False
        )
    
    def test_encode_method(self):
        """Test Llama2Template._encode method specifically."""
        # Test with system prompt and tools
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Mock _make_pairs to return predefined pairs
        mock_pairs = [([1, 2], [3, 4])]
        with mock.patch.object(self.template, '_make_pairs', return_value=mock_pairs):
            result = self.template._encode(
                self.mock_tokenizer,
                messages,
                "Test system",
                "Test tools",
                1000,
                1
            )
            
            assert result == mock_pairs
            
            # Verify format_prefix was called for first turn
            self.mock_format_prefix.apply.assert_called_once()
            
            # Verify system and tools were processed
            self.mock_format_tools.apply.assert_called_with(content="Test tools")
            self.mock_format_system.apply.assert_called_once()
            
            # Verify user and assistant formats were applied
            self.mock_format_user.apply.assert_called_once()
            self.mock_format_assistant.apply.assert_called_once()
        
        # Test with multiple turns
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        self.mock_format_user.apply.reset_mock()
        self.mock_format_assistant.apply.reset_mock()
        self.mock_format_separator.apply.reset_mock()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thanks!"}
        ]
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                "System prompt",
                None,
                1000,
                1
            )
            
            # Verify separator was called for second turn
            self.mock_format_separator.apply.assert_called_once()
            
            # Verify formats were applied correctly
            assert self.mock_format_user.apply.call_count == 2
            assert self.mock_format_assistant.apply.call_count == 2
        
        # Test with no system and no tools
        self.mock_format_prefix.apply.reset_mock()
        self.mock_format_system.apply.reset_mock()
        self.mock_format_tools.apply.reset_mock()
        
        with mock.patch.object(self.template, '_make_pairs', return_value=[]):
            self.template._encode(
                self.mock_tokenizer,
                messages,
                None,
                None,
                1000,
                1
            )
            
            # Current code uses self.default_system when system is None, so format_system.apply is called
            self.mock_format_system.apply.assert_called_once_with(content="default system")
            # When tools is None, format_tools.apply is not called
            self.mock_format_tools.apply.assert_not_called()
            
            # But prefix should still be called
            self.mock_format_prefix.apply.assert_called_once()


class TestTemplateRegistry:
    def setup_method(self):
        """Set up by clearing templates registry before each test."""
        from agentic_rl.base.utils.templates import templates
        self.original_templates = templates.copy()
        templates.clear()
    
    def teardown_method(self):
        """Restore original templates after each test."""
        from agentic_rl.base.utils.templates import templates
        templates.clear()
        templates.update(self.original_templates)
    
    def test_register_template(self):
        """Test _register_template function."""
        # Register a simple template
        _register_template(name="test_template")
        
        # Check if template is registered
        templates = get_templates()
        assert "test_template" in templates
        assert isinstance(templates["test_template"], Template)
        
        # Register a llama2 template
        _register_template(name="llama2_test")
        assert "llama2_test" in templates
        assert isinstance(templates["llama2_test"], Llama2Template)
        
        # Test with custom formatters
        custom_user_formatter = StringFormatter(slots=["[USER] {{content}}"])
        custom_assistant_formatter = StringFormatter(slots=["[ASSISTANT] {{content}}"])
        _register_template(
            name="custom_template",
            format_user=custom_user_formatter,
            format_assistant=custom_assistant_formatter,
            default_system="Custom system",
            stop_words=["stop1", "stop2"],
            efficient_eos=True,
            replace_eos=True,
            force_system=True
        )
        
        custom_template = templates["custom_template"]
        assert custom_template.default_system == "Custom system"
        assert custom_template.stop_words == ["stop1", "stop2"]
        assert custom_template.efficient_eos is True
        assert custom_template.replace_eos is True
        assert custom_template.force_system is True
        
        # Test with None formatters (should use defaults)
        _register_template(name="default_template")
        default_template = templates["default_template"]
        assert default_template is not None
        
        # Test with efficient_eos=True (should not add eos_token to assistant formatter)
        _register_template(name="efficient_eos_template", efficient_eos=True)
        efficient_template = templates["efficient_eos_template"]
        assert efficient_template.efficient_eos is True
        
        # Test with replace_eos=True
        _register_template(name="replace_eos_template", replace_eos=True)
        replace_template = templates["replace_eos_template"]
        assert replace_template.replace_eos is True
        
        # Test with force_system=True
        _register_template(name="force_system_template", force_system=True)
        force_template = templates["force_system_template"]
        assert force_template.force_system is True
    
    def test_register_template_eos_slots(self):
        """Test _register_template eos_slots logic."""
        # Test with efficient_eos=False (should add eos_token)
        _register_template(name="eos_template", efficient_eos=False)
        eos_template = get_templates()["eos_template"]
        assert eos_template.efficient_eos is False
        
        # Test with efficient_eos=True (should not add eos_token)
        _register_template(name="no_eos_template", efficient_eos=True)
        no_eos_template = get_templates()["no_eos_template"]
        assert no_eos_template.efficient_eos is True
    
    def test_register_template_formatter_defaults(self):
        """Test _register_template default formatter selection."""
        # Test with format_user provided but format_observation not (should use format_user)
        custom_user_formatter = StringFormatter(slots=["[CUSTOM_USER] {{content}}"])
        _register_template(name="user_as_observation_test", format_user=custom_user_formatter)
        
        template = get_templates()["user_as_observation_test"]
        assert template.format_user == custom_user_formatter
        assert template.format_observation == custom_user_formatter
        
        # Test with both format_user and format_observation provided (should use format_observation)
        custom_observation_formatter = StringFormatter(slots=["[CUSTOM_OBS] {{content}}"])
        _register_template(
            name="custom_observation_test", 
            format_user=custom_user_formatter, 
            format_observation=custom_observation_formatter
        )
        
        template = get_templates()["custom_observation_test"]
        assert template.format_user == custom_user_formatter
        assert template.format_observation == custom_observation_formatter
    
    def test_register_template_defaults(self):
        """Test _register_template with default parameters."""
        _register_template(name="default_test")
        
        templates = get_templates()
        assert "default_test" in templates
        template = templates["default_test"]
        
        # Verify default values
        assert template.default_system == ""
        assert template.stop_words == []
        assert template.efficient_eos is False
        assert template.replace_eos is False
        assert template.force_system is False
    
    def test_register_template_efficient_eos(self):
        """Test _register_template with efficient_eos=True."""
        _register_template(name="efficient_eos_test", efficient_eos=True)
        
        templates = get_templates()
        assert "efficient_eos_test" in templates
        template = templates["efficient_eos_test"]
        
        # Verify efficient_eos is set correctly
        assert template.efficient_eos is True
    
    def test_register_template_with_tools(self):
        """Test _register_template with custom tools formatter."""
        custom_tool_formatter = ToolFormatter(tool_format="custom")
        _register_template(name="tools_test", format_tools=custom_tool_formatter)
        
        templates = get_templates()
        assert "tools_test" in templates
        template = templates["tools_test"]
        
        # Verify tools formatter is set correctly
        assert template.format_tools == custom_tool_formatter
    
    def test_register_template_llama2_variants(self):
        """Test _register_template with different llama2 variants."""
        # Test various llama2 template names
        llama2_names = [
            "llama2",
            "llama2_chat",
            "llama2_7b",
            "llama2_13b",
            "llama2_70b"
        ]
        
        for name in llama2_names:
            _register_template(name=name)
            templates = get_templates()
            assert name in templates
            assert isinstance(templates[name], Llama2Template)
    
    def test_register_template_default_formatters(self):
        """Test _register_template uses correct default formatters."""
        _register_template(name="test_default_formatters")
        
        templates = get_templates()
        template = templates["test_default_formatters"]
        
        # Verify default formatters are used when None is provided
        assert template.format_user is not None
        assert template.format_assistant is not None
        assert template.format_system is not None
        assert template.format_function is not None
        assert template.format_observation is not None
        assert template.format_tools is not None
        assert template.format_separator is not None
        assert template.format_prefix is not None
    
    def test_register_template_with_observation_formatter(self):
        """Test _register_template with custom observation formatter."""
        custom_observation_formatter = StringFormatter(slots=["[OBS] {{content}}"])
        custom_user_formatter = StringFormatter(slots=["[USR] {{content}}"])
        
        # Test with custom observation formatter
        _register_template(
            name="custom_observation_test",
            format_observation=custom_observation_formatter
        )
        
        templates = get_templates()
        template = templates["custom_observation_test"]
        assert template.format_observation == custom_observation_formatter
        
        # Test with custom user formatter but no observation formatter (should use user formatter)
        _register_template(
            name="user_as_observation_test",
            format_user=custom_user_formatter
        )
        
        templates = get_templates()
        template = templates["user_as_observation_test"]
        assert template.format_observation == custom_user_formatter
    
    def test_get_model_template(self):
        """Test get_model_template function."""
        # Register a template
        _register_template(name="test_template")
        
        # Test getting existing template
        template = get_model_template("test_template", None)
        assert template is not None
        
        # Test with None name (should return placeholder)
        with mock.patch("agentic_rl.base.utils.templates.register_custom_template", return_value=None):
            _register_template(name="empty", format_user=StringFormatter(slots=["{{content}}"]))
            template = get_model_template(None, "dummy_path")
            assert template is not None
        
        # Test with non-existing template (should raise error)
        with mock.patch("agentic_rl.base.utils.templates.register_custom_template", side_effect=ValueError("Template non_existing does not exist.")):
            with pytest.raises(ValueError):
                get_model_template("non_existing", "/dummy/path")
        
        # Test with register_custom_template returning a new name
        with mock.patch("agentic_rl.base.utils.templates.register_custom_template", return_value="new_template_name"):
            _register_template(name="new_template_name")
            template = get_model_template("original_name", "/path/to/template.json")
            assert template is not None
        
        # Test with register_custom_template returning existing name
        with mock.patch("agentic_rl.base.utils.templates.register_custom_template", return_value="test_template"):
            template = get_model_template("custom_template", "/path/to/template.json")
            assert template is not None
            assert template == get_templates()["test_template"]
    
    def test_format_custom_template(self):
        """Test _format_custom_template function."""
        # Test with list values
        slots = {
            "key1": ["value1", "value2"],
            "key2": None
        }
        result = _format_custom_template(slots)
        assert result["key1"] == ["value1", "value2"]
        assert result["key2"] is None
        
        # Test with None
        assert _format_custom_template(None) is None
        
        # Test with non-dict
        assert _format_custom_template("not a dict") == "not a dict"
        assert _format_custom_template(123) == 123
        assert _format_custom_template([]) == []
        
        # Test with set values
        slots = {
            "key1": {"eos_token"},
            "key2": [{"bos_token"}, "regular string"]
        }
        result = _format_custom_template(slots)
        assert result["key1"] == ["eos_token"]  # 集合被转换为字符串列表
        assert result["key2"] == [{"bos_token"}, "regular string"]  # 嵌套集合保持不变
        
        # Test with empty slots dict
        empty_slots = {}
        result = _format_custom_template(empty_slots)
        assert result == {}
        
        # Test with mixed types
        mixed_slots = {
            "str_key": "string_value",
            "list_key": ["item1", "item2"],
            "none_key": None,
            "set_key": {"bos_token"}
        }
        result = _format_custom_template(mixed_slots)
        assert result["str_key"] == list("string_value")  # 字符串被转换为字符列表
        assert result["list_key"] == ["item1", "item2"]  # 列表元素如果是字符串则保持不变
        assert result["none_key"] is None
        assert result["set_key"] == ["bos_token"]  # 集合被转换为字符串列表
        
        # Test with nested structures
        nested_slots = {
            "outer_key": {
                "inner_key": ["value1", {"token": "inner_token"}]
            }
        }
        result = _format_custom_template(nested_slots)
        assert isinstance(result["outer_key"], list)
        assert result["outer_key"] == ["inner_key"]
    
    def test_format_custom_template_edge_cases(self):
        """Test _format_custom_template with edge cases."""
        # Test with deeply nested structures
        nested_slots = {
            "level1": {
                "level2": {
                    "level3": ["value1", {"token": "inner_token"}]
                }
            }
        }
        result = _format_custom_template(nested_slots)
        assert isinstance(result["level1"], list)
        assert result["level1"] == ["level2"]
        
        # Test with tuple values
        tuple_slots = {
            "tuple_key": ("value1", "value2")
        }
        result = _format_custom_template(tuple_slots)
        assert result["tuple_key"] == list(("value1", "value2"))  # 元组被转换为列表
        
        # Test with empty list values
        empty_list_slots = {
            "key1": [],
            "key2": {"slots": []}
        }
        result = _format_custom_template(empty_list_slots)
        assert result["key1"] is None  # 空列表被视为falsy值，返回None
        assert isinstance(result["key2"], list)
        assert result["key2"] == ["slots"]

        # Test with None slot value
        slots_none_value = {
            "key1": None,
            "key2": {"slots": None}
        }
        result = _format_custom_template(slots_none_value)
        assert result["key1"] is None
        assert isinstance(result["key2"], list)
        assert result["key2"] == ["slots"]  # 内部字典的键被保留为列表元素
        
        # Test with non-dict input
        non_dict_input = "not a dict"
        result = _format_custom_template(non_dict_input)
        assert result == "not a dict"
        
        # Test with empty dict
        empty_dict = {}
        result = _format_custom_template(empty_dict)
        assert result == {}
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_formatters(self, mock_json_load, mock_open):
        """Test register_custom_template formatter creation."""
        # Mock json file content with all formatter types
        mock_json_load.return_value = [{
            "name": "formatter_test",
            "format_user": {"slots": ["[USER] {{content}}"]},
            "format_assistant": {"slots": ["[ASSISTANT] {{content}}"]},
            "format_system": {"slots": ["[SYSTEM] {{content}}"]},
            "format_function": {"slots": ["[FUNCTION] {{name}} {{arguments}}"]},
            "format_observation": {"slots": ["[OBSERVATION] {{content}}"]},
            "format_tools": {"tool_format": "custom"},
            "format_separator": {"slots": ["\n\n"]},
            "format_prefix": {"slots": ["[PREFIX]"]}
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test registering template with all formatters
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("formatter_test", "/path/to/template.json")
            assert result == "formatter_test"
            mock_register.assert_called_once()
            
            # Verify all formatters were passed to _register_template
            call_args = mock_register.call_args[1]
            assert call_args["format_user"] is not None
            assert call_args["format_assistant"] is not None
            assert call_args["format_system"] is not None
            assert call_args["format_function"] is not None
            assert call_args["format_observation"] is not None
            assert call_args["format_tools"] is not None
            assert call_args["format_separator"] is not None
            assert call_args["format_prefix"] is not None
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_path_validation(self, mock_json_load, mock_open):
        """Test register_custom_template path validation."""
        # Test with invalid paths
        invalid_paths = [
            "",
            "invalid\0path",
        ]

        for path in invalid_paths:
            with pytest.raises(ValueError, match="Invalid Path"):
                register_custom_template("test_template", path)
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_default_system(self, mock_json_load, mock_open):
        """Test register_custom_template default_system handling."""
        # Mock json file content with list default_system
        mock_json_load.return_value = [{
            "name": "list_system_test",
            "default_system": ["System part 1", "System part 2"]
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("list_system_test", "/path/to/template.json")
            assert result == "list_system_test"
            
            # Verify default_system was joined into a string
            call_args = mock_register.call_args[1]
            assert call_args["default_system"] == "System part 1System part 2"
        
        # Test with mixed type default_system (should not be joined)
        mock_json_load.return_value = [{
            "name": "mixed_system_test",
            "default_system": ["System part 1", 123, "System part 2"]
        }]
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("mixed_system_test", "/path/to/template.json")
            assert result == "mixed_system_test"
            
            # Verify default_system was not joined
            call_args = mock_register.call_args[1]
            assert call_args["default_system"] == ["System part 1", 123, "System part 2"]
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_empty_config(self, mock_json_load, mock_open):
        """Test register_custom_template with empty config values."""
        # Mock json file content with empty values
        mock_json_load.return_value = [{
            "name": "empty_config_test",
            "default_system": "",
            "stop_words": [],
            "efficient_eos": False,
            "replace_eos": False,
            "force_system": False
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("empty_config_test", "/path/to/template.json")
            assert result == "empty_config_test"
            mock_register.assert_called_once()
            
            # Verify empty values were passed correctly
            call_args = mock_register.call_args[1]
            assert call_args["default_system"] == ""
            assert call_args["stop_words"] == []
            assert call_args["efficient_eos"] is False
            assert call_args["replace_eos"] is False
            assert call_args["force_system"] is False
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_formatter_none(self, mock_json_load, mock_open):
        """Test register_custom_template with some formatters set to None."""
        # Mock json file content with some formatters as None
        mock_json_load.return_value = [{
            "name": "formatter_none_test",
            "format_user": {"slots": ["[USER] {{content}}"]},
            "format_assistant": None,
            "format_system": None,
            "format_function": None,
            "format_observation": None,
            "format_tools": None,
            "format_separator": None,
            "format_prefix": None
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("formatter_none_test", "/path/to/template.json")
            assert result == "formatter_none_test"
            mock_register.assert_called_once()
            
            # Verify that None formatters were passed correctly
            call_args = mock_register.call_args[1]
            assert call_args["format_user"] is not None
            assert call_args["format_assistant"] is None
            assert call_args["format_system"] is None
            assert call_args["format_function"] is None
            assert call_args["format_observation"] is None
            assert call_args["format_tools"] is None
            assert call_args["format_separator"] is None
            assert call_args["format_prefix"] is None
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    def test_register_custom_template_default_system_list(self, mock_json_load, mock_open):
        """Test register_custom_template with default_system as list."""
        # Mock json file content with list default_system
        mock_json_load.return_value = [{
            "name": "default_system_list_test",
            "default_system": ["Line 1", "Line 2", "Line 3"]
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("default_system_list_test", "/path/to/template.json")
            assert result == "default_system_list_test"
            mock_register.assert_called_once()
            
            # Verify default_system was joined into a string
            call_args = mock_register.call_args[1]
            assert call_args["default_system"] == "Line 1Line 2Line 3"
        
        # Test with mixed type default_system (should not be joined)
        mock_json_load.return_value = [{
            "name": "default_system_mixed_test",
            "default_system": ["Line 1", 123, "Line 3"]
        }]
        
        with mock.patch("agentic_rl.base.utils.templates._register_template") as mock_register:
            result = register_custom_template("default_system_mixed_test", "/path/to/template.json")
            assert result == "default_system_mixed_test"
            mock_register.assert_called_once()
            
            # Verify default_system was not joined
            call_args = mock_register.call_args[1]
            assert call_args["default_system"] == ["Line 1", 123, "Line 3"]
    
    @mock.patch("builtins.open")
    @mock.patch("json.load")
    @mock.patch("agentic_rl.base.utils.templates._register_template")
    def test_register_custom_template(self, mock_register, mock_json_load, mock_open):
        """Test register_custom_template function."""
        # Mock json file content
        mock_json_load.return_value = [{
            "name": "custom_test",
            "format_user": {"slots": ["{{content}}"]},
            "format_assistant": {"slots": ["{{content}}"]}
        }]
        
        # Mock file opening
        mock_file = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test registering custom template
        result = register_custom_template("custom_test", "/path/to/template.json")
        
        assert result == "custom_test"
        mock_open.assert_called_once_with("/path/to/template.json", 'r')
        mock_json_load.assert_called_once()
        mock_register.assert_called_once()
        
        # Test with already registered template
        result = register_custom_template("custom_test", "/path/to/template.json")
        assert result == "custom_test"
        
        # Test with invalid path
        with pytest.raises(ValueError):
            register_custom_template("custom_test", "invalid_path")
        
        # Test with non-existing template in file
        mock_json_load.return_value = [{"name": "other_template"}]
        with pytest.raises(ValueError):
            register_custom_template("custom_test", "/path/to/template.json")
        
        # Test with comprehensive template configuration
        mock_json_load.return_value = [{
            "name": "comprehensive_template",
            "format_user": {"slots": ["[USER] {{content}}"]},
            "format_assistant": {"slots": ["[ASSISTANT] {{content}}"]},
            "format_system": {"slots": ["[SYSTEM] {{content}}"]},
            "format_function": {"slots": ["[FUNCTION] {{name}} {{arguments}}"]},
            "format_observation": {"slots": ["[OBSERVATION] {{content}}"]},
            "format_tools": {"tool_format": "custom"},
            "format_separator": {"slots": ["\n\n"]},
            "format_prefix": {"slots": ["[PREFIX]"]},
            "default_system": "Default system prompt",
            "stop_words": ["stop1", "stop2", "stop3"],
            "efficient_eos": True,
            "replace_eos": True,
            "force_system": True
        }]
        
        mock_register.reset_mock()
        mock_open.reset_mock()
        mock_json_load.reset_mock()
        
        result = register_custom_template("comprehensive_template", "/path/to/template.json")
        assert result == "comprehensive_template"
        assert mock_open.called
        assert mock_json_load.called
        assert mock_register.called
        
        # Test with default_system as list
        mock_json_load.return_value = [{
            "name": "list_system_test",
            "default_system": ["System part 1", "System part 2"]
        }]
        
        mock_register.reset_mock()
        mock_open.reset_mock()
        mock_json_load.reset_mock()
        
        result = register_custom_template("list_system_test", "/path/to/template.json")
        assert result == "list_system_test"
        
        # Test with default_system as non-string list
        mock_json_load.return_value = [{
            "name": "non_string_system_test",
            "default_system": ["System part 1", 123, "System part 2"]
        }]
        
        mock_register.reset_mock()
        mock_open.reset_mock()
        mock_json_load.reset_mock()
        
        result = register_custom_template("non_string_system_test", "/path/to/template.json")
        assert result == "non_string_system_test"
        
        # Test with valid relative paths
        valid_paths = [
            "./relative/path.json",
            "../parent/path.json",
            "relative/path.json",
            "/absolute/path.json"
        ]
        
        for path in valid_paths:
            mock_register.reset_mock()
            mock_open.reset_mock()
            mock_json_load.reset_mock()
            
            # 设置mock返回值，包含"valid_path_test"模板
            mock_json_load.return_value = [{
                "name": "valid_path_test",
                "format_user": {"slots": ["{{content}}"]},
                "format_assistant": {"slots": ["{{content}}"]},
                "format_system": {"slots": ["{{content}}"]},
                "format_function": {"slots": ["Action: {{name}}\nAction Input: {{arguments}}"]},
                "format_observation": {"slots": ["{{content}}"]},
                "format_tools": {"slots": [""], "tool_format": "default"},
                "format_separator": {"slots": [""]},
                "format_prefix": {"slots": [""]},
                "default_system": "",
                "stop_words": [],
                "efficient_eos": False,
                "replace_eos": False,
                "force_system": False
            }]
            
            result = register_custom_template("valid_path_test", path)
            assert result == "valid_path_test"
            mock_open.assert_called_once_with(path, 'r')
        
        # Test with formatter fields as None
        mock_json_load.return_value = [{
            "name": "none_formatters_test",
            "format_user": None,
            "format_assistant": None,
            "format_system": None,
            "format_function": None,
            "format_observation": None,
            "format_tools": None,
            "format_separator": None,
            "format_prefix": None
        }]
        
        mock_register.reset_mock()
        mock_open.reset_mock()
        mock_json_load.reset_mock()
        
        result = register_custom_template("none_formatters_test", "/path/to/template.json")
        assert result == "none_formatters_test"
    
    def test_get_templates(self):
        """Test get_templates function."""
        # Register a template
        _register_template(name="test_template")

        # Get templates
        templates = get_templates()

        # Check if template is in the returned dict
        assert "test_template" in templates
        assert isinstance(templates["test_template"], Template)

        # Verify that modifying the returned dict affects the original (current behavior)
        templates["new_template"] = "test"
        original_templates = get_templates()
        assert "new_template" in original_templates