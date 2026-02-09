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

import unittest

from agentic_rl.base.utils.checker import Checker, CompletionRequestChecker


class TestChecker(unittest.TestCase):
    def test_validate_param_none_value(self):
        self.assertIsNone(Checker.validate_param('test_field', int, None))

    def test_validate_param_incorrect_type(self):
        with self.assertRaises(TypeError):
            Checker.validate_param('test_field', int, 'string')

    def test_validate_param_correct_type(self):
        self.assertIsNone(Checker.validate_param('test_field', int, 10))

    def test_validate_param_value_less_than_min(self):
        with self.assertRaises(ValueError):
            Checker.validate_param('test_field', int, 5, min_val=10)

    def test_validate_param_value_more_than_max(self):
        with self.assertRaises(ValueError):
            Checker.validate_param('test_field', int, 15, max_val=10)

    def test_validate_param_value_in_range(self):
        self.assertIsNone(Checker.validate_param('test_field', int, 10, min_val=5, max_val=15))


class TestCompletionRequestChecker(unittest.TestCase):
    """Test suite for CompletionRequestValidator class."""

    def test_valid_minimal_request(self):
        """Test validation passes with only required fields."""
        request = {
            "prompt": "Hello, world!"
        }
        # Should not raise any exception
        CompletionRequestChecker.validate_input(request)

    def test_valid_minimal_chat_request(self):
        request = {
            "messages": [{"role": "user", "content": "hello"}]
        }
        CompletionRequestChecker.validate_chat_input(request)

    def test_valid_complete_request(self):
        """Test validation passes with all valid fields."""
        request = {
            "prompt": "Test prompt",
            "n": 1,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "min_p": 0.1,
            "max_tokens": 100,
            "min_tokens": 10,
            "logprobs": True,
            "detokenize": False,
            "seed": 12345,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5
        }
        # Should not raise any exception
        CompletionRequestChecker.validate_input(request)

    def test_valid_complete_chat_request(self):
        """Test validation passes with all valid fields."""
        request = {
            "messages": [{"role": "user", "content": "hello"}],
            "n": 1,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "min_p": 0.1,
            "max_tokens": 100,
            "min_tokens": 10,
            "logprobs": True,
            "detokenize": False,
            "seed": 12345,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5
        }
        # Should not raise any exception
        CompletionRequestChecker.validate_chat_input(request)

    def test_invalid_request_type(self):
        """Test validation fails when request is not a dictionary."""
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_input("not a dict")
        self.assertIn("must be a dictionary", str(context.exception))

    def test_invalid_chat_request_type(self):
        """Test validation fails when request is not a dictionary."""
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_chat_input("not a dict")
        self.assertIn("must be a dictionary", str(context.exception))

    def test_unrecognized_field(self):
        """Test validation fails with unrecognized field."""
        request = {
            "prompt": "Test",
            "invalid_field": "value"
        }
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("Unrecognized field(s)", str(context.exception))
        self.assertIn("invalid_field", str(context.exception))

    def test_unrecognized_chat_field(self):
        """Test validation fails with unrecognized field."""
        request = {
            "messages": [{"role": "user", "content": "hello"}],
            "invalid_field": "value"
        }
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("Unrecognized field(s)", str(context.exception))
        self.assertIn("invalid_field", str(context.exception))

    def test_multiple_unrecognized_fields(self):
        """Test validation fails with multiple unrecognized fields."""
        request = {
            "prompt": "Test",
            "bad_field1": "value1",
            "bad_field2": "value2"
        }
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("Unrecognized field(s)", str(context.exception))

    def test_missing_prompt_field(self):
        """Test validation fails when prompt field is missing."""
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input({})
        self.assertIn("Missing required field: 'prompt'", str(context.exception))

    def test_missing_messages_field(self):
        """Test validation fails when messages field is missing."""
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input({})
        self.assertIn("Missing required field: 'messages'", str(context.exception))

    def test_invalid_prompt_type(self):
        """Test validation fails when prompt is not a string."""
        request = {"prompt": 123}
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("'prompt' must be a string", str(context.exception))

    def test_invalid_messages_type(self):
        """Test validation fails when messages is not a string."""
        request = {"messages": {"role": "user", "content": "hello"}}
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("Field 'messages' must be a list", str(context.exception))

    def test_empty_prompt_string(self):
        """Test validation fails when prompt is an empty string."""
        request = {"prompt": ""}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("'prompt' cannot be an empty string", str(context.exception))

    def test_empty_messages_string(self):
        """Test validation fails when messages is an empty string."""
        request = {"messages": []}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("Field 'messages' cannot be an empty list", str(context.exception))

    def test_invalid_messages_non_list_dict(self):
        request = {"messages": [{"role": "user", "content": "hello"}, "not a dict"]}
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 1 of field 'messages' must be a dict", str(context.exception))

    def test_invalid_messages_without_role(self):
        request = {"messages": [{"content": "hello"}]}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 0 of field 'messages' missing required field: 'role'", str(context.exception))

    def test_invalid_messages_without_content(self):
        request = {"messages": [{"role": "user"}]}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 0 of field 'messages' missing required field: 'content'", str(context.exception))

    def test_invalid_messages_with_invalid_role(self):
        request = {"messages": [{"role": "invalid", "content": "hello"}]}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 0 'role' is invalid", str(context.exception))

    def test_invalid_messages_with_invalid_content(self):
        request = {"messages": [{"role": "user", "content": 123}]}
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 0 'content' must be a string", str(context.exception))

    def test_invalid_messages_with_empty_content(self):
        request = {"messages": [{"role": "user", "content": ""}]}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_chat_input(request)
        self.assertIn("The member 0 'content' cannot be an empty string", str(context.exception))

    def test_temperature_valid_range(self):
        """Test temperature validation with valid values."""
        for temp in [0, 0.5, 1.0, 1.5, 2.0]:
            request = {"prompt": "Test", "temperature": temp}
            CompletionRequestChecker.validate_input(request)

    def test_temperature_below_min(self):
        """Test validation fails when temperature is below 0."""
        request = {"prompt": "Test", "temperature": -0.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("temperature", str(context.exception).lower())

    def test_temperature_above_max(self):
        """Test validation fails when temperature is above 2."""
        request = {"prompt": "Test", "temperature": 2.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("temperature", str(context.exception).lower())

    def test_top_k_valid_values(self):
        """Test top_k validation with valid values."""
        for top_k in [0, 1, 50, 100]:
            request = {"prompt": "Test", "top_k": top_k}
            CompletionRequestChecker.validate_input(request)

    def test_top_k_invalid_value(self):
        """Test validation fails when top_k is below -1."""
        request = {"prompt": "Test", "top_k": -2}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("top_k", str(context.exception).lower())

    def test_top_k_invalid_type(self):
        """Test validation fails when top_k is not an integer."""
        request = {"prompt": "Test", "top_k": 1.5}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("top_k", str(context.exception).lower())

    def test_top_p_valid_range(self):
        """Test top_p validation with valid values."""
        for top_p in [0, 0.5, 0.9, 1.0]:
            request = {"prompt": "Test", "top_p": top_p}
            CompletionRequestChecker.validate_input(request)

    def test_top_p_below_min(self):
        """Test validation fails when top_p is below 0."""
        request = {"prompt": "Test", "top_p": -0.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("top_p", str(context.exception).lower())

    def test_top_p_above_max(self):
        """Test validation fails when top_p is above 1."""
        request = {"prompt": "Test", "top_p": 1.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("top_p", str(context.exception).lower())

    def test_min_p_valid_range(self):
        """Test min_p validation with valid values."""
        for min_p in [0, 0.1, 0.5, 1.0]:
            request = {"prompt": "Test", "min_p": min_p}
            CompletionRequestChecker.validate_input(request)

    def test_min_p_out_of_range(self):
        """Test validation fails when min_p is out of range."""
        request = {"prompt": "Test", "min_p": 1.5}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("min_p", str(context.exception).lower())

    def test_n_valid_values(self):
        """Test n validation with valid positive integers."""
        for n in [1, 5, 10]:
            request = {"prompt": "Test", "n": n}
            CompletionRequestChecker.validate_input(request)

    def test_n_zero_or_negative(self):
        """Test validation fails when n is zero or negative."""
        for n in [0, -1]:
            request = {"prompt": "Test", "n": n}
            with self.assertRaises(ValueError) as context:
                CompletionRequestChecker.validate_input(request)
            self.assertIn("'n'", str(context.exception))

    def test_max_tokens_valid_values(self):
        """Test max_tokens validation with valid positive integers."""
        for max_tokens in [1, 100, 1000]:
            request = {"prompt": "Test", "max_tokens": max_tokens}
            CompletionRequestChecker.validate_input(request)

    def test_max_tokens_zero_or_negative(self):
        """Test validation fails when max_tokens is zero or negative."""
        request = {"prompt": "Test", "max_tokens": 0}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("max_tokens", str(context.exception).lower())

    def test_min_tokens_valid_range(self):
        """Test min_tokens validation with valid values."""
        for min_tokens in [0, 100, 64000]:
            request = {"prompt": "Test", "min_tokens": min_tokens}
            CompletionRequestChecker.validate_input(request)

    def test_min_tokens_negative(self):
        """Test validation fails when min_tokens is negative."""
        request = {"prompt": "Test", "min_tokens": -1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("min_tokens", str(context.exception).lower())

    def test_min_tokens_above_max(self):
        """Test validation fails when min_tokens exceeds 64000."""
        request = {"prompt": "Test", "min_tokens": 64001}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("min_tokens", str(context.exception).lower())

    def test_presence_penalty_valid_range(self):
        """Test presence_penalty validation with valid values."""
        for penalty in [-2, -1, 0, 1, 2]:
            request = {"prompt": "Test", "presence_penalty": penalty}
            CompletionRequestChecker.validate_input(request)

    def test_presence_penalty_below_min(self):
        """Test validation fails when presence_penalty is below -2."""
        request = {"prompt": "Test", "presence_penalty": -2.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("presence_penalty", str(context.exception).lower())

    def test_presence_penalty_above_max(self):
        """Test validation fails when presence_penalty is above 2."""
        request = {"prompt": "Test", "presence_penalty": 2.1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("presence_penalty", str(context.exception).lower())

    def test_frequency_penalty_valid_range(self):
        """Test frequency_penalty validation with valid values."""
        for penalty in [-2, 0, 2]:
            request = {"prompt": "Test", "frequency_penalty": penalty}
            CompletionRequestChecker.validate_input(request)

    def test_frequency_penalty_out_of_range(self):
        """Test validation fails when frequency_penalty is out of range."""
        request = {"prompt": "Test", "frequency_penalty": 3}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("frequency_penalty", str(context.exception).lower())

    def test_logprobs_boolean_values(self):
        """Test logprobs validation with boolean values."""
        for value in [True, False]:
            request = {"prompt": "Test", "logprobs": value}
            CompletionRequestChecker.validate_input(request)

    def test_logprobs_valid_integer(self):
        """Test logprobs validation with non-negative integers."""
        for value in [0, 1, 5]:
            request = {"prompt": "Test", "logprobs": value}
            CompletionRequestChecker.validate_input(request)

    def test_logprobs_negative_integer(self):
        """Test validation fails when logprobs is a negative integer."""
        request = {"prompt": "Test", "logprobs": -1}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("logprobs", str(context.exception).lower())

    def test_detokenize_boolean_values(self):
        """Test detokenize validation with boolean values."""
        for value in [True, False]:
            request = {"prompt": "Test", "detokenize": value}
            CompletionRequestChecker.validate_input(request)

    def test_detokenize_invalid_type(self):
        """Test validation fails when detokenize is not a boolean."""
        request = {"prompt": "Test", "detokenize": "true"}
        with self.assertRaises(TypeError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("detokenize", str(context.exception).lower())

    def test_seed_valid_range(self):
        """Test seed validation with valid values."""
        for seed in [-65535, -1000, 0, 1000, 65535]:
            request = {"prompt": "Test", "seed": seed}
            CompletionRequestChecker.validate_input(request)

    def test_seed_below_min(self):
        """Test validation fails when seed is below -65535."""
        request = {"prompt": "Test", "seed": -65536}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("seed", str(context.exception).lower())

    def test_seed_above_max(self):
        """Test validation fails when seed is above 65535."""
        request = {"prompt": "Test", "seed": 65536}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("seed", str(context.exception).lower())

    def test_seed_invalid_type(self):
        """Test validation fails when seed is not an integer."""
        request = {"prompt": "Test", "seed": 123.45}
        with self.assertRaises(ValueError) as context:
            CompletionRequestChecker.validate_input(request)
        self.assertIn("seed", str(context.exception).lower())


if __name__ == '__main__':
    unittest.main()
