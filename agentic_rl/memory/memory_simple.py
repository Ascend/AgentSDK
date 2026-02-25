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

import copy
import re
from datetime import datetime, timezone
from typing import Any

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.memory.memory_base import MemoryBase
from agentic_rl.memory.utils import validate_message
from agentic_rl.memory.memory_config import MemoryConfig
from agentic_rl.memory.token_counter import (
    HuggingFaceTokenCounter,
    SimpleTokenCounter,
    TokenCounter
)

logger = Loggers(__name__)


class MemorySimple(MemoryBase):
    """Simple in-memory message storage implementation with token caching.

    This class provides basic message storage and retrieval functionality
    for conversational AI systems. Messages are stored in a list with
    automatic timestamping and ID assignment. It also maintains a cache of
    token counts for stored messages.
    """

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None,
            message="config must be a dictionary or None",
        ),
        token_counter=dict(
            validator=lambda x: isinstance(x, TokenCounter) or x is None,
            message="token_counter must be a TokenCounter instance or None",
        )
    )
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the memory with optional configuration.

        Args:
            config: Optional configuration dictionary for memory settings.
            token_counter: Optional TokenCounter instance. If None, will be initialized from config.
        """
        self.messages: list[dict[str, Any]] = []
        self._token_cache: dict[int, int] = {}
        self.config = MemoryConfig()

        self.token_counter = token_counter
        self.update_config(config)

        # Initialize token counter if not provided or if config update didn't set it
        if self.token_counter is None:
            self._initialize_token_counter()

        self._next_id = 0
        super().__init__(config)

    def __len__(self) -> int:
        """Return the number of messages stored in memory."""
        return len(self.messages)

    @staticmethod
    @validate_params(
        messages=dict(
            validator=lambda x: isinstance(x, str) or validate_message(x),
            message="messages must be a list or string",
        ),
        start_id=dict(
            validator=lambda x: isinstance(x, int) and x >= 0,
            message="start_id must be a non-negative integer",
        ),
        end_id=dict(
            validator=lambda x: isinstance(x, int) and x >= -1,
            message="end_id must be an integer and greater than -1",
        )
    )
    def simplify_or_remove_think(
        messages: list[dict[str, Any]] | str,
        start_id: int = 0,
        end_id: int = -1,
    ) -> list[dict[str, Any]] | str:
        """Simplify thinking content from assistant messages.

        Replaces detailed <think>...</think> content with a placeholder it entirely.
        Processes messages within the range [start_id, end_id) when messages is a list.
        When messages is a string, start_id and end_id parameters are ignored.

        Args:
            messages: Either a list of message dicts or a single string to process.
            start_id: Starting index for processing (inclusive). Default 0. Ignored when messages is a string.
            end_id: Ending index for processing (exclusive). Default -1 will be converted to len(messages).
                    Ignored when messages is a string.

        Returns:
            Modified messages with simplified thinking content.

        Raises:
            TypeError: If messages is not a list or string.
        """
        if isinstance(messages, list):
            if end_id == -1:
                end_id = len(messages)
            else:
                # Ensure end_id doesn't exceed list length
                end_id = min(end_id, len(messages))
            
            # Ensure start_id doesn't exceed end_id
            if start_id >= end_id:
                logger.warning(f"Invalid range: start_id ({start_id}) must be less than end_id ({end_id})")
                return messages

            # Find assistant message indices in the specified range
            for i in range(start_id, end_id):
                if messages[i].get("role") == "assistant":
                    content = messages[i].get("content", "")
                    # Replace thinking content with simplified version
                    modified_content = re.sub(
                        r"<think>.*?</think>",
                        "<think>Thinking process omitted.</think>",
                        content,
                        flags=re.DOTALL,
                    )
                    messages[i]["content"] = modified_content
            return messages
        else:
            # For string input, replace thinking content with simplified version
            return re.sub(r"<think>.*?</think>", "<think>Thinking process omitted.</think>", messages, flags=re.DOTALL)

    @staticmethod
    @validate_params(
        messages=dict(
            validator=validate_message, message="messages must be a list of dictionaries with string keys",
        ),
        save_keys=dict(
            validator=lambda x: isinstance(x, list) or x is None, message="save_keys must be a list or None",
        )
    )
    def _remove_message_other_key(
        messages: list[dict[str, Any]], save_keys: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Filter message fields to retain only specified keys.

        Used to align with OpenAI and other API formats that expect only 'role' and 'content' fields.

        Args:
            messages: List of message dictionaries to filter.
            save_keys: List of keys to retain. Defaults to ['role', 'content'].

        Returns:
            List of filtered message dictionaries.
        """
        if save_keys is None:
            save_keys = ["role", "content"]

        return [{key: msg[key] for key in save_keys if key in msg} for msg in messages]

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None",
        ),
    )
    def update_config(self, config: dict[str, Any] | None = None) -> None:
        """Update memory-related configuration parameters.

        Args:
            config: Dictionary containing configuration parameters to update.
        """
        old_path = self.config.train_model_tokenizer_path
        self.config.update(config)

        # Re-initialize tokenizer if path changed
        if self.config.train_model_tokenizer_path != old_path:
            self._initialize_token_counter(force=True)

    @validate_params(
        message=dict(
            validator=validate_message, message="message must be a dictionary or list of dictionaries",
        ),
        user_id=dict(
            validator=lambda x: isinstance(x, str) or x is None, message="user_id must be a string or None",
        ),
        session_id=dict(
            validator=lambda x: isinstance(x, str) or x is None, message="session_id must be a string or None",
        ),
        insert_id=dict(
            validator=lambda x: isinstance(x, int) or x is None, message="insert_id must be an integer or None",
        )
    )
    def add_message(
        self,
        message: dict[str, Any] | list[dict[str, Any]],
        user_id: str | None = None,
        session_id: str | None = None,
        insert_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Add message(s) to memory.

        Messages are automatically enriched with 'time' and 'id' fields if not present.

        Args:
            message: Single message dict or list of message dicts.
                    Each message should contain 'role' and 'content' fields.
            user_id: Optional user identifier for multi-user scenarios.
            session_id: Optional session identifier for organizing conversations.
            insert_id: Position to insert message(s), or None to append to end.
            **kwargs: Additional keyword arguments (reserved for future use).

        Raises:
            ValueError: If message is not a dict[str, Any] or list[dict[str, Any]].
        """
        if isinstance(message, dict):
            self._process_single_message(message, insert_id)
        else:
            current_insert_id = insert_id
            for msg in message:
                self._process_single_message(msg, current_insert_id)
                if current_insert_id is not None:
                    current_insert_id += 1

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages currently stored in memory."""
        return copy.deepcopy(self.messages)

    @validate_params(
        role=dict(
            validator=lambda x: isinstance(x, str) or x is None, message="role must be a string or None",
        ),
        content=dict(
            validator=lambda x: isinstance(x, str) or x is None, message="content must be a string or None",
        )
    )
    def clear_memory(self, role: str | None = None, content: str | None = None) -> None:
        """Clear all messages from memory.

        Optionally add a new initial message after clearing.

        Args:
            role: Optional role for initial message to add after clearing.
            content: Optional content for initial message to add after clearing.
        """
        self.messages = []
        self._token_cache.clear()
        if role is not None:
            self.add_message({"role": role, "content": content})

    @validate_params(
        limit_size=dict(
            validator=lambda x: isinstance(x, int), message="limit_size must be an integer",
        )
    )
    def get_window_messages(self, limit_size: int = -1) -> list[dict[str, Any]]:
        """Get messages within a context window.

        Returns the most recent messages up to the specified limit,
        with internal fields removed for API compatibility.

        Args:
            limit_size: Maximum number of recent messages to return.
                        Negative integer (default -1) returns all messages.

        Returns:
            List of filtered message dictionaries containing only 'role' and 'content'.
        """
        if limit_size >= 0:
            init_id = max(0, len(self.messages) - limit_size)
        else:
            init_id = 0

        messages_slice = self.messages[init_id:]
        return self._remove_message_other_key(copy.deepcopy(messages_slice))

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None",
        )
    )
    def get_prompt_messages(self, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Get messages formatted for LLM prompt construction.

        Applies transformations based on configuration:
        - Optionally simplifies thinking content
        - Filters to only 'role' and 'content' fields for API compatibility

        Args:
            config: Optional configuration to apply before formatting.
                    If provided, updates the memory configuration.

        Returns:
            List of formatted message dictionaries ready for LLM prompts.
        """
        messages = self.get_messages()

        if config is not None:
            self.update_config(config)

        if self.config.simplify_thinking:
            # Type ignore: simplify_or_remove_think returns list[dict] or str. Here messages is list[dict],
            # so it returns list[dict].
            messages = self.simplify_or_remove_think(
                messages,
                self.config.before_raw_message,
                len(messages) + self.config.end_raw_message
            )

        messages = self._remove_message_other_key(messages)
        return messages

    @validate_params(
        message=dict(
            validator=validate_message, message="message must be a dictionary with string keys"
        )
    )
    def get_message_length(self, message: dict[str, Any]) -> int:
        """Get token length of a message, using cache if available.

        Args:
            message: The message dictionary.

        Returns:
            Token count.
        """
        msg_id = message.get("id")
        if msg_id is not None and msg_id in self._token_cache:
            return self._token_cache[msg_id]

        if self.token_counter:
            return self.token_counter.count_message(message)
        return 0

    @validate_params(
        messages=dict(
            validator=lambda x: validate_message(x) or x is None, message="messages must be a list or None"
        )
    )
    def get_total_length(self, messages: list[dict[str, Any]] | None = None) -> int:
        """Calculate total token length of messages.

        Args:
            messages: Optional list of messages to count. If None, counts all stored messages.

        Returns:
            Total token count.
        """
        if messages is None:
            # Count all stored messages (fast sum)
            return sum(self._token_cache.values())

        return sum(self.get_message_length(msg) for msg in messages)

    @validate_params(
        force=dict(
            validator=lambda x: isinstance(x, bool), message="force must be a boolean"
        )
    )
    def _initialize_token_counter(self, force: bool = False) -> None:
        """Initialize the token counter based on configuration."""
        if self.token_counter is not None and not force:
            return

        if self.config.train_model_tokenizer_path:
            try:
                self.token_counter = HuggingFaceTokenCounter(self.config.train_model_tokenizer_path)
            except OSError as e:
                logger.warning(
                    f"Tokenizer file not found at {self.config.train_model_tokenizer_path}: {e}. Using fallback."
                )
                self.token_counter = SimpleTokenCounter()
            except ValueError as e:
                logger.warning(
                    f"Invalid tokenizer configuration at {self.config.train_model_tokenizer_path}: {e}. Using fallback."
                )
                self.token_counter = SimpleTokenCounter()
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer from {self.config.train_model_tokenizer_path}: {e}. Using fallback."
                )
                self.token_counter = SimpleTokenCounter()
        else:
            self.token_counter = SimpleTokenCounter()

    @validate_params(
        message=dict(
            validator=validate_message, message="message must be a dictionary with string keys"
        ),
        insert_id=dict(
            validator=lambda x: isinstance(x, int) or x is None, message="insert_id must be an integer or None"
        )
    )
    def _process_single_message(self, message: dict[str, Any], insert_id: int | None) -> None:
        """Process and add a single message to memory.

        Automatically adds timestamp and ID if not present.
        Calculates and caches token count.

        Args:
            message: Message dictionary to process and add.
            insert_id: Position to insert message, or None to append.
        """
        if "time" not in message:
            message["time"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if "id" not in message:
            message["id"] = self._next_id
            self._next_id += 1

        # Cache token count
        if self.token_counter:
            self._token_cache[message["id"]] = self.token_counter.count_message(message)

        if insert_id is None:
            self.messages.append(message)
        else:
            self.messages.insert(insert_id, message)
