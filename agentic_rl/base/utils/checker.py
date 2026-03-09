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
import inspect
import functools
from typing import Any, Dict
import torch
import numpy as np


class ValidatorReturnTypeError(TypeError):
    """Custom exception raised when a validator function returns a non-boolean value."""
    pass


def _get_value_from_param(arg_name, func, *args, **kwargs):
    sig = inspect.signature(func)
    # Get the value to validate from the passed arguments
    for param_name, param in sig.bind(*args, **kwargs).arguments.items():
        if arg_name == param_name:
            return param
    # If not found in passed arguments, get default value from function definition
    for name, param in sig.parameters.items():
        if arg_name == name:
            return param.default
    # Throw exception if neither found
    raise ValueError(f"Required parameter '{arg_name}' of function {func.__name__} is missing.")


def validate_params(**validators):
    """
    Define a decorator for validating multiple function parameters. Use annotation on methods
    @validate_params(
        name=dict(validator=lambda x: isinstance(x, str)),
        age=dict(validator=lambda x: 10 <= x <= 30)
    )
    :param validators: A dictionary containing validation functions, each for validating a specific parameter.
                      IMPORTANT: Validator functions MUST return a boolean value (True/False).
                      Returning None, 0, empty lists, or other non-boolean values will raise a TypeError.
    :return: Decorator function
    """

    def decorator(func):
        # Validate that all validator parameter names exist in the function signature
        sig = inspect.signature(func)
        func_param_names = set(sig.parameters.keys())
        validator_param_names = set(validators.keys())

        # Check for invalid parameter names
        invalid_params = validator_param_names - func_param_names
        if invalid_params:
            raise ValueError(
                f"Invalid parameter name(s) in validators for function '{func.__name__}': "
                f"{', '.join(sorted(invalid_params))}. "
                f"Valid parameters are: {', '.join(sorted(func_param_names))}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validation function to each parameter
            for arg_name, validator in validators.items():
                # Check if parameter was passed by position or keyword
                value = _get_value_from_param(arg_name, func, *args, **kwargs)
                # Run validation function
                try:
                    result = validator['validator'](value)
                    # Strictly check if the validator function returns a boolean value
                    if not isinstance(result, bool):
                        raise ValidatorReturnTypeError(
                            f"Validator function for parameter '{arg_name}' in function '{func.__name__}' "
                            f"must return a boolean value, but got {type(result).__name__}: {result!r}"
                            f"{validator.get('message', 'Validation failed')}"
                        )
                    if not result:
                        raise ValueError(f"The parameter '{arg_name}' of function '{func.__name__}' "
                                         f"is invalid, message: {validator.get('message', 'Validation failed')}")
                except ValidatorReturnTypeError:
                    # This is our own custom error from return type checking - re-raise as is
                    raise
                except TypeError as e:
                    # This is a TypeError from the validator function itself - preserve the original type
                    context_msg = f"TypeError in validator for parameter '{arg_name}' of function '{func.__name__}'"
                    validator_message = validator.get('message', 'Validation failed')
                    context_msg += f", validator message: {validator_message}"
                    enhanced_msg = f"{context_msg}. Original error: {str(e)}"
                    raise TypeError(enhanced_msg) from e
                except Exception as e:
                    # Preserve the original exception type and add context information
                    exception_type = type(e)
                    original_message = str(e)
                    context_message = f"Exception in validator for parameter '{arg_name}' of function '{func.__name__}'"
                    validator_message = validator.get('message', 'Validation failed')
                    context_message += f", validator message: {validator_message}"
                    enhanced_message = f"{context_message}. Original error: {original_message}"
                    raise exception_type(enhanced_message) from e
            # If all parameters pass validation, call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


class Checker:
    @staticmethod
    def validate_param(field_name, expected_type, value, min_val=None, max_val=None):
        if value is None:
            return
        if not isinstance(value, expected_type):
            raise TypeError(f"{field_name} must be {expected_type.__name__} type, but got: {type(value).__name__}")
        if min_val is not None and value < min_val:
            raise ValueError(f"{field_name} should be more than {min_val}, but got: {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{field_name} should be less than {max_val}, but got: {value}")


class CompletionRequestChecker:
    """Validator for OpenAI completion API requests."""

    # Define allowed fields for completion requests
    ALLOWED_FIELDS = {
        "prompt", "n", "temperature", "top_k", "top_p", "min_p",
        "max_tokens", "min_tokens", "logprobs", "detokenize",
        "seed", "presence_penalty", "frequency_penalty",
    }

    CHAT_ALLOWED_FIELDS = {
        "messages", "n", "temperature", "top_k", "top_p", "min_p",
        "max_tokens", "min_tokens", "logprobs", "detokenize",
        "seed", "presence_penalty", "frequency_penalty", "model", "stream",
    }

    @staticmethod
    def validate_input(raw_request: Dict[str, Any]) -> None:
        """Validate completion request input parameters.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError or TypeError: If any validation fails
        """
        if not isinstance(raw_request, dict):
            raise TypeError("raw_request must be a dictionary")
        if not all(isinstance(key, str) for key in raw_request.keys()):
            raise ValueError("all keys in raw_request must be strings")

        # Validate no unrecognized fields are present
        CompletionRequestChecker._validate_allowed_fields(raw_request)

        # Validate required fields
        CompletionRequestChecker._validate_required_fields(raw_request)
        CompletionRequestChecker._validate_prompt_field(raw_request["prompt"])

        # Validate optional parameters by category
        CompletionRequestChecker._validate_sampling_params(raw_request)
        CompletionRequestChecker._validate_token_params(raw_request)
        CompletionRequestChecker._validate_penalty_params(raw_request)
        CompletionRequestChecker._validate_misc_params(raw_request)

    @staticmethod
    def validate_chat_input(raw_request: Dict[str, Any]) -> None:
        """Validate completion request input parameters.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError or TypeError: If any validation fails
        """
        if not isinstance(raw_request, dict):
            raise TypeError("raw_request must be a dictionary")
        if not all(isinstance(key, str) for key in raw_request.keys()):
            raise ValueError("all keys in raw_request must be strings")

        # Validate no unrecognized fields are present
        CompletionRequestChecker._validate_chat_allowed_fields(raw_request)

        # Validate required fields
        CompletionRequestChecker._validate_chat_required_fields(raw_request)
        CompletionRequestChecker._validate_messages_field(raw_request["messages"])

        # Validate optional parameters by category
        CompletionRequestChecker._validate_sampling_params(raw_request)
        CompletionRequestChecker._validate_token_params(raw_request)
        CompletionRequestChecker._validate_penalty_params(raw_request)
        CompletionRequestChecker._validate_misc_params(raw_request)

    @staticmethod
    def _validate_allowed_fields(raw_request: Dict[str, Any]) -> None:
        """Validate that only allowed fields are present in the request.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If any unrecognized field is present
        """
        request_fields = set(raw_request.keys())
        unknown_fields = request_fields - CompletionRequestChecker.ALLOWED_FIELDS

        if unknown_fields:
            unknown_list = sorted(unknown_fields)
            raise ValueError(
                f"Unrecognized field(s) in request: {', '.join(unknown_list)}. "
                f"Allowed fields are: {', '.join(sorted(CompletionRequestChecker.ALLOWED_FIELDS))}"
            )

    @staticmethod
    def _validate_chat_allowed_fields(raw_request: Dict[str, Any]) -> None:
        """Validate that only allowed fields are present in the request.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If any unrecognized field is present
        """
        request_fields = set(raw_request.keys())
        unknown_fields = request_fields - CompletionRequestChecker.CHAT_ALLOWED_FIELDS

        if unknown_fields:
            unknown_list = sorted(unknown_fields)
            raise ValueError(
                f"Unrecognized field(s) in request: {', '.join(unknown_list)}. "
                f"Allowed fields are: {', '.join(sorted(CompletionRequestChecker.CHAT_ALLOWED_FIELDS))}"
            )

    @staticmethod
    def _validate_numeric_range(field_name: str, value: Any,
                                expected_type: type, min_val: float, max_val: float) -> None:
        """Validate numeric field is within expected range.

        Args:
            field_name: Name of the field being validated
            value: Value to validate
            expected_type: Expected type (int or float)
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Raises:
            ValueError or TypeError: If validation fails
        """
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__ if expected_type in (int, float) else "number"
            raise TypeError(f"Field '{field_name}' must be a {type_name}")
        if value < min_val or value > max_val:
            raise ValueError(f"Field '{field_name}' must be between {min_val} and {max_val}")

    @staticmethod
    def _validate_integer_field(field_name: str, value: Any, min_val: int = 1) -> None:
        """Validate integer field with minimum value.

        Args:
            field_name: Name of the field being validated
            value: Value to validate
            min_val: Minimum allowed value

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, int) or value < min_val:
            raise ValueError(f"Field '{field_name}' must be an integer >= {min_val}")

    @staticmethod
    def _validate_required_fields(raw_request: Dict[str, Any]) -> None:
        """Validate presence of required fields.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If any required field is missing
        """
        required_fields = ["prompt"]
        for field in required_fields:
            if field not in raw_request:
                raise ValueError(f"Missing required field: '{field}'")

    @staticmethod
    def _validate_chat_required_fields(raw_request: Dict[str, Any]) -> None:
        """Validate presence of required fields.

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If any required field is missing
        """
        required_fields = ["messages"]
        for field in required_fields:
            if field not in raw_request:
                raise ValueError(f"Missing required field: '{field}'")

    @staticmethod
    def _validate_prompt_field(prompt: Any) -> None:
        """Validate prompt field format and content.

        Args:
            prompt: Prompt field value

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(prompt, str):
            raise TypeError("Field 'prompt' must be a string")
        if not prompt:
            raise ValueError("Field 'prompt' cannot be an empty string")

    @staticmethod
    def _validate_messages_field(messages: Any) -> None:
        """Validate messages field format and content.

        Args:
            messages: messages field value

        Raises:
            ValueError: If validation fails
        """
        if messages is None or not isinstance(messages, list):
            raise TypeError("Field 'messages' must be a list")
        if len(messages) == 0:
            raise ValueError("Field 'messages' cannot be an empty list")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"The member {i} of field 'messages' must be a dict")
            if "role" not in msg:
                raise ValueError(f"The member {i} of field 'messages' missing required field: 'role'")
            if "content" not in msg:
                raise ValueError(f"The member {i} of field 'messages' missing required field: 'content'")

            role = msg["role"]
            content = msg["content"]
            if role not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"The member {i} 'role' is invalid")
            if not isinstance(content, str):
                raise TypeError(f"The member {i} 'content' must be a string")
            if content.strip() == "":
                raise ValueError(f"The member {i} 'content' cannot be an empty string")

    @staticmethod
    def _validate_sampling_params(raw_request: Dict[str, Any]) -> None:
        """Validate sampling parameters (temperature, top_k, top_p, min_p).

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If validation fails
        """
        if "temperature" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "temperature", raw_request["temperature"], (int, float), 0, 2
            )

        if "top_k" in raw_request:
            CompletionRequestChecker._validate_integer_field("top_k", raw_request["top_k"], 0)

        if "top_p" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "top_p", raw_request["top_p"], (int, float), 0, 1
            )

        if "min_p" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "min_p", raw_request["min_p"], (int, float), 0, 1
            )

    @staticmethod
    def _validate_token_params(raw_request: Dict[str, Any]) -> None:
        """Validate token-related parameters (max_tokens, min_tokens, n).

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If validation fails
        """
        if "n" in raw_request:
            CompletionRequestChecker._validate_integer_field("n", raw_request["n"], 1)

        if "max_tokens" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "max_tokens", raw_request["max_tokens"], (int,), 1, 64000
            )

        if "min_tokens" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "min_tokens", raw_request["min_tokens"], (int,), 0, 64000
            )

    @staticmethod
    def _validate_penalty_params(raw_request: Dict[str, Any]) -> None:
        """Validate penalty parameters (presence_penalty, frequency_penalty).

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If validation fails
        """
        if "presence_penalty" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "presence_penalty", raw_request["presence_penalty"], (int, float), -2, 2
            )

        if "frequency_penalty" in raw_request:
            CompletionRequestChecker._validate_numeric_range(
                "frequency_penalty", raw_request["frequency_penalty"], (int, float), -2, 2
            )

    @staticmethod
    def _validate_misc_params(raw_request: Dict[str, Any]) -> None:
        """Validate miscellaneous parameters (logprobs, detokenize, model_name, seed).

        Args:
            raw_request: The raw completion request dictionary

        Raises:
            ValueError: If validation fails
        """
        if "logprobs" in raw_request:
            logprobs = raw_request["logprobs"]
            is_valid = isinstance(logprobs, bool) or (isinstance(logprobs, int) and logprobs >= 0)
            if not is_valid:
                raise ValueError("Field 'logprobs' must be a boolean or non-negative integer")

        if "detokenize" in raw_request:
            if not isinstance(raw_request["detokenize"], bool):
                raise TypeError("Field 'detokenize' must be a boolean")

        if "seed" in raw_request:
            seed = raw_request["seed"]
            if not isinstance(seed, int) or not (-65535 <= seed <= 65535):
                raise ValueError("Field 'seed' must be an integer between -65535 and 65535")


class TrajectoryChecker:
    @staticmethod
    def validate_param(trajectory: dict):
        prompt_tokens = trajectory.get("prompt_tokens", None)
        response_tokens = trajectory.get("response_tokens", None)
        response_masks = trajectory.get("response_masks", None)
        idx = trajectory.get("idx", None)
        trajectory_reward = trajectory.get("trajectory_reward", None)
        chat_completions = trajectory.get("chat_completions", None)
        metrics = trajectory.get("metrics", None)

        if prompt_tokens is None:
            raise ValueError(f"Trajectory's prompt_tokens is not set")
        TrajectoryChecker._validate_tensors("prompt_tokens", prompt_tokens)

        if response_tokens is None:
            raise ValueError(f"Trajectory's response_tokens is not set")
        TrajectoryChecker._validate_tensors("response_tokens", response_tokens)

        if response_masks is None:
            raise ValueError(f"Trajectory's response_masks is not set")
        TrajectoryChecker._validate_tensors("response_masks", response_masks)

        if metrics is None:
            raise ValueError(f"Trajectory's metrics is not set")
        TrajectoryChecker._validate_metrics(metrics)

        if trajectory_reward is None:
            raise ValueError(f"Trajectory's trajectory_reward is not set")
        if not np.issubdtype(type(trajectory_reward), np.number):
            raise TypeError(f"trajectory_reward must be a number, got {type(trajectory_reward).__name__}")

        if not isinstance(idx, int):
            raise TypeError(f"Trajectory's idx must be an integer, got {type(idx).__name__}")
        if idx < 0:
            raise ValueError("Trajectory's idx must be non-negative")

        if not isinstance(chat_completions, list) or not all(isinstance(item, dict) for item in chat_completions):
            raise TypeError(f"chat_completions must be a list of dict, got {type(chat_completions).__name__}")
        for item in chat_completions:
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in item.items()):
                raise TypeError("all keys and values in chat_completions dicts must be strings")

    @staticmethod
    def _validate_tensors(field_name: str, data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{field_name} must be torch.Tensor, got {type(data).__name__}")

        if torch.isnan(data).any():
            raise ValueError(f"{field_name} contains NaN values")

        if torch.isinf(data).any():
            raise ValueError(f"{field_name} contains Inf values")

    @staticmethod
    def _validate_metrics(metrics):
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")

        if not all(isinstance(key, str) for key in metrics.keys()):
            raise TypeError("all keys in metrics must be strings")

        expected_keys = {'steps', 'reward_time', 'env_time', 'llm_time', 'total_time', 'toolcall_reward', 'res_reward'}
        current_keys = set(metrics.keys())
        if current_keys != expected_keys:
            raise ValueError(f"metrics must contain exactly these keys: {sorted(expected_keys)}")

        if not isinstance(metrics['steps'], int):
            raise TypeError(f"metric steps must be an integer, got {type(metrics['steps']).__name__}")

        if not np.issubdtype(type(metrics['toolcall_reward']), np.number):
            raise TypeError(f"metric toolcall_reward must be a number, got {type(metrics['toolcall_reward']).__name__}")

        if not np.issubdtype(type(metrics['res_reward']), np.number):
            raise TypeError(f"metric res_reward must be a number, got {type(metrics['res_reward']).__name__}")

        if metrics['steps'] < 0:
            raise ValueError("metric steps must be non-negative")

        for key in current_keys - {'steps', 'env_time', 'llm_time', 'toolcall_reward', 'res_reward'}:
            if metrics[key] is not None and not isinstance(metrics[key], (int, float)):
                raise TypeError(f"metric {key} must be a number or None, got {type(metrics[key]).__name__}")

            if metrics[key] is not None and metrics[key] < 0:
                raise ValueError(f"metric {key} must be non-negative if not None")

    @staticmethod
    def _validate_chat_completions(chat_completions: object) -> None:
        """Verify if chat_completions is valid"""
        if not isinstance(chat_completions, list) or not all(isinstance(item, dict) for item in chat_completions):
            raise TypeError(f"chat_completions must be a list of dict, got {type(chat_completions).__name__}")

        for item in chat_completions:
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in item.items()):
                raise TypeError("all keys and values in chat_completions dicts must be strings")

    @staticmethod
    def validate_step(chat_completions, thought, model_response, info, reward, done, mc_return):
        """Verify if Step dataclass is valid"""
        TrajectoryChecker._validate_chat_completions(chat_completions)
        if not isinstance(thought, str):
            raise TypeError(f"thought must be a string, got {type(thought).__name__}")
        if not isinstance(model_response, str):
            raise TypeError(f"model_response must be a string, got {type(model_response).__name__}")
        if not isinstance(info, dict):
            raise TypeError(f"info must be a dict, got {type(info).__name__}")
        if not isinstance(reward, (int, float)):
            raise TypeError(f"reward must be a number, got {type(reward).__name__}")
        if not isinstance(done, bool):
            raise TypeError(f"done must be a boolean, got {type(done).__name__}")
        if not isinstance(mc_return, (int, float)):
            raise TypeError(f"mc_return must be a number, got {type(mc_return).__name__}")
