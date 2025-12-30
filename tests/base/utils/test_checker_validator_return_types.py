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

import pytest
from agentic_rl.base.utils.checker import validate_params


class TestValidatorReturnTypes:
    """Test cases for validator function return type checking."""

    def test_validator_returns_true(self):
        """Test that validator function returning True works correctly."""

        @validate_params(name=dict(validator=lambda x: isinstance(x, str)))
        def test_func(name):
            return f"Hello, {name}"

        # Should work without raising any exceptions
        result = test_func("Alice")
        assert result == "Hello, Alice"

    def test_validator_returns_false(self):
        """Test that validator function returning False raises ValueError."""

        @validate_params(age=dict(validator=lambda x: x >= 18, message="Age must be at least 18"))
        def test_func(age):
            return f"Age is {age}"

        # Should raise ValueError for validation failure
        with pytest.raises(ValueError, match="The parameter 'age' of function 'test_func' is invalid"):
            _ = test_func(15)

    def test_validator_returns_none_raises_type_error(self):
        """Test that validator function returning None raises TypeError."""

        @validate_params(
            value=dict(validator=lambda x: None)  # Returns None instead of bool
        )
        def test_func(value):
            return value

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError,
            match="Validator function for parameter 'value' .* must return a boolean value, but got NoneType: None",
        ):
            _ = test_func(42)

    def test_validator_returns_zero_raises_type_error(self):
        """Test that validator function returning 0 raises TypeError."""

        @validate_params(
            number=dict(validator=lambda x: 0)  # Returns 0 instead of bool
        )
        def test_func(number):
            return number

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError, match="Validator function for parameter 'number' .* must return a boolean value, but got int: 0"
        ):
            _ = test_func(123)

    def test_validator_returns_nonzero_int_raises_type_error(self):
        """Test that validator function returning non-zero integer raises TypeError."""

        @validate_params(
            item=dict(validator=lambda x: 1)  # Returns 1 instead of bool
        )
        def test_func(item):
            return item

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError, match="Validator function for parameter 'item' .* must return a boolean value, but got int: 1"
        ):
            _ = test_func("test")

    def test_validator_returns_empty_list_raises_type_error(self):
        """Test that validator function returning empty list raises TypeError."""

        @validate_params(
            data=dict(validator=lambda x: [])  # Returns [] instead of bool
        )
        def test_func(data):
            return data

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError,
            match="Validator function for parameter 'data' .* must return a boolean value, but got list: \\[\\]",
        ):
            _ = test_func("some data")

    def test_validator_returns_non_empty_list_raises_type_error(self):
        """Test that validator function returning non-empty list raises TypeError."""

        @validate_params(
            items=dict(validator=lambda x: [1, 2, 3])  # Returns list instead of bool
        )
        def test_func(items):
            return items

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError,
            match=("Validator function for parameter 'items' .* must return a boolean value, "
                  "but got list: \\[1, 2, 3\\]"),
        ):
            _ = test_func("test")

    def test_validator_returns_empty_string_raises_type_error(self):
        """Test that validator function returning empty string raises TypeError."""

        @validate_params(
            text=dict(validator=lambda x: "")  # Returns "" instead of bool
        )
        def test_func(text):
            return text

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError, match="Validator function for parameter 'text' .* must return a boolean value, but got str: ''"
        ):
            _ = test_func("input")

    def test_validator_returns_non_empty_string_raises_type_error(self):
        """Test that validator function returning non-empty string raises TypeError."""

        @validate_params(
            message=dict(validator=lambda x: "valid")  # Returns string instead of bool
        )
        def test_func(message):
            return message

        # Should raise TypeError for non-boolean return value
        with pytest.raises(
            TypeError,
            match="Validator function for parameter 'message' .* must return a boolean value, but got str: 'valid'",
        ):
            _ = test_func("hello")

    def test_multiple_validators_with_mixed_return_types(self):
        """Test multiple validators where some return correct bool and others don't."""

        @validate_params(
            name=dict(validator=lambda x: isinstance(x, str)),  # Returns bool (correct)
            age=dict(validator=lambda x: None),  # Returns None (incorrect)
        )
        def test_func(name, age):
            return f"{name} is {age} years old"

        # Should raise TypeError for the second validator (age) returning None
        with pytest.raises(
            TypeError,
            match="Validator function for parameter 'age' .* must return a boolean value, but got NoneType: None",
        ):
            _ = test_func("Alice", 25)

    def test_validator_exception_handling_with_attribute_error(self):
        """Test that AttributeError from validator is preserved with proper context."""

        @validate_params(
            value=dict(validator=lambda x: x.nonexistent_method(), message="Value must have required method")
        )
        def test_func(value):
            return value

        # Should raise AttributeError (preserved original type), not ValueError
        with pytest.raises(AttributeError) as exc_info:
            _ = test_func("test")

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'value' of function 'test_func'" in error_msg
        assert "validator message: Value must have required method" in error_msg
        assert "Original error:" in error_msg

    def test_complex_validator_returning_bool_correctly(self):
        """Test complex validation logic that correctly returns boolean."""

        def complex_validator(value):
            # Complex logic that should return bool
            if isinstance(value, str):
                return len(value) > 3 and value.isalpha()
            return False

        @validate_params(
            username=dict(validator=complex_validator, message="Username must be alphabetic and > 3 chars")
        )
        def register_user(username):
            return f"User {username} registered"

        # Valid case
        result = register_user("Alice")
        assert result == "User Alice registered"

        # Invalid case - should raise ValueError, not TypeError
        with pytest.raises(ValueError, match="The parameter 'username' of function 'register_user' is invalid"):
            _ = register_user("ab")

    def test_validator_attribute_error_preserves_original_type(self):
        """Test that AttributeError from validator is preserved as AttributeError."""

        @validate_params(data=dict(validator=lambda x: x.nonexistent_method(), message="Data must have method"))
        def process_data(data):
            return data

        # Should raise AttributeError (not ValueError), with added context
        with pytest.raises(AttributeError) as exc_info:
            _ = process_data("test_string")

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'data' of function 'process_data'" in error_msg
        assert "validator message: Data must have method" in error_msg
        assert "Original error:" in error_msg
        assert "'str' object has no attribute 'nonexistent_method'" in error_msg

    def test_validator_key_error_preserves_original_type(self):
        """Test that KeyError from validator is preserved as KeyError."""

        @validate_params(
            config=dict(validator=lambda x: x["required_key"] is not None, message="Config must have required_key")
        )
        def setup_config(config):
            return config

        # Should raise KeyError (not ValueError), with added context
        with pytest.raises(KeyError) as exc_info:
            _ = setup_config({"other_key": "value"})

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'config' of function 'setup_config'" in error_msg
        assert "validator message: Config must have required_key" in error_msg
        assert "Original error:" in error_msg

    def test_validator_index_error_preserves_original_type(self):
        """Test that IndexError from validator is preserved as IndexError."""

        @validate_params(
            items=dict(validator=lambda x: x[10] is not None, message="Items must have at least 11 elements")
        )
        def process_items(items):
            return items

        # Should raise IndexError (not ValueError), with added context
        with pytest.raises(IndexError) as exc_info:
            _ = process_items([1, 2, 3])

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'items' of function 'process_items'" in error_msg
        assert "validator message: Items must have at least 11 elements" in error_msg
        assert "Original error:" in error_msg

    def test_validator_value_error_preserves_original_type(self):
        """Test that ValueError from validator is preserved as ValueError."""

        @validate_params(number=dict(validator=lambda x: int(x) > 0, message="Number must be positive integer"))
        def calculate(number):
            return number * 2

        # Should raise ValueError (preserved from int() call), with added context
        with pytest.raises(ValueError) as exc_info:
            _ = calculate("not_a_number")

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'number' of function 'calculate'" in error_msg
        assert "validator message: Number must be positive integer" in error_msg
        assert "Original error:" in error_msg

    def test_validator_type_error_from_validator_function(self):
        """Test that TypeError from validator function (not return type check) is properly handled."""

        @validate_params(
            value=dict(
                validator=lambda x: len(None) > 0, message="Value must be valid"
            )  # TypeError: object of type 'NoneType' has no len()
        )
        def test_func(value):
            return value

        # Should raise TypeError (preserved from len(None)), with added context
        with pytest.raises(TypeError) as exc_info:
            _ = test_func(42)

        error_msg = str(exc_info.value)
        assert "TypeError in validator for parameter 'value' of function 'test_func'" in error_msg
        assert "validator message: Value must be valid" in error_msg
        assert "Original error:" in error_msg
        assert "object of type 'NoneType' has no len()" in error_msg

    def test_validator_without_message_in_exception_context(self):
        """Test exception handling when validator doesn't have a message."""

        @validate_params(
            value=dict(validator=lambda x: x.missing_attr)  # No message provided
        )
        def test_func(value):
            return value

        with pytest.raises(AttributeError) as exc_info:
            _ = test_func("test")

        error_msg = str(exc_info.value)
        assert "Exception in validator for parameter 'value' of function 'test_func'" in error_msg
        assert "Original error:" in error_msg
