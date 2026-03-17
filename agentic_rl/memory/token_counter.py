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

from abc import ABC, abstractmethod

from transformers import AutoTokenizer

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.memory.constants import ROLE_TOKEN_OVERHEAD
from agentic_rl.memory.utils import validate_message

logger = Loggers(__name__)


class TokenCounter(ABC):
    """Abstract base class for token counting strategies."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        pass

    @abstractmethod
    def count_message(self, message: dict) -> int:
        """Count tokens in a message dictionary."""
        pass

    @abstractmethod
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        pass

    @abstractmethod
    def split_text(self, text: str, chunk_size_tokens: int) -> list[str]:
        """Split text into chunks of at most chunk_size_tokens."""
        pass


class HuggingFaceTokenCounter(TokenCounter):
    """Token counter using HuggingFace AutoTokenizer."""

    def __init__(self, model_path: str) -> None:
        FileCheck.check_path_is_exist_and_valid(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, weights_only=True)

    @validate_params(text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"))
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @validate_params(message=dict(validator=validate_message, message="message must be a dictionary with string keys"))
    def count_message(self, message: dict) -> int:
        # Simple approximation: content tokens + 4 overhead tokens (role, etc.)

        content = message.get("content", "")
        return self.count_tokens(content) + ROLE_TOKEN_OVERHEAD

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"),
        max_tokens=dict(validator=lambda x: isinstance(x, int) and x > 0,
                        message="max_tokens must be a positive integer")
    )
    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        logger.info(f"Summary length: {len(tokens)} exceeds maximum prompt length, truncated to {max_tokens}.")
        return self.tokenizer.decode(tokens[:max_tokens])

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"),
        chunk_size_tokens=dict(validator=lambda x: isinstance(x, int) and x > 0,
                               message="chunk_size_tokens must be a positive integer")
    )
    def split_text(self, text: str, chunk_size_tokens: int) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), chunk_size_tokens):
            chunk_tokens = tokens[i: i + chunk_size_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks


class SimpleTokenCounter(TokenCounter):
    """Fallback token counter using character count heuristic."""

    @validate_params(chars_per_token=dict(validator=lambda x: isinstance(x, int) and x > 0,
                                           message="chars_per_token must be a positive integer"))
    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token

    @validate_params(text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"))
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    @validate_params(message=dict(validator=validate_message, message="message must be a dictionary with string keys"))
    def count_message(self, message: dict) -> int:
        content = message.get("content", "")
        return self.count_tokens(content) + ROLE_TOKEN_OVERHEAD

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"),
        max_tokens=dict(validator=lambda x: isinstance(x, int) and x > 0,
                        message="max_tokens must be a positive integer")
    )
    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * self.chars_per_token
        if len(text) <= max_chars:
            return text
        logger.info(f"Summary char length: {len(text)} exceeds maximum prompt length, truncated to {max_chars}.")
        return text[:max_chars]

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str), message="text must be a string"),
        chunk_size_tokens=dict(validator=lambda x: isinstance(x, int) and x > 0,
                               message="chunk_size_tokens must be a positive integer")
    )
    def split_text(self, text: str, chunk_size_tokens: int) -> list[str]:
        chunk_size_chars = chunk_size_tokens * self.chars_per_token
        return [text[i: i + chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
