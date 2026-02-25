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


class MemoryBase(ABC):
    """Abstract base class for memory management in agentic systems.

    This class provides an interface for storing, retrieving, and managing
    conversational messages or interaction history. Implementations should
    handle message persistence, context window management, and prompt formatting.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the memory with optional configuration."""
        pass

    @abstractmethod
    def update_config(self, config: dict) -> None:
        """Update memory-related configuration parameters.

        Args:
            config(dict): Dictionary containing configuration parameters to update.
        """
        pass

    @abstractmethod
    def add_message(
            self,
            message: dict | list[dict],
            user_id: str | None = None,
            session_id: str | None = None,
            insert_id: int | None = None,
            **kwargs,
    ) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """Retrieve all messages currently stored in memory."""
        pass

    @abstractmethod
    def clear_memory(self, role: str | None = None, content: str | None = None) -> None:
        """Clear all messages from memory."""
        pass

    @abstractmethod
    def get_window_messages(self, limit_size: int = -1) -> list[dict]:
        """Retrieve messages within a context window."""
        pass

    @abstractmethod
    def get_prompt_messages(self) -> list[dict]:
        """Get messages formatted for prompt construction."""
        pass
