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

from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from agentic_rl.base.utils.checker import validate_params
from agentic_rl.memory.prompts import PROCEDURAL_MEMORY_SYSTEM_PROMPT


class MemoryConfig(BaseModel):
    """Configuration for memory management in agentic systems.

    This class manages configuration for conversation memory, including:
    - Content processing (thinking simplification)
    - Summary generation and context window management
    - Model endpoints for chat and embeddings

    Attributes:
        simplify_thinking: Whether to remove/simplify thinking content from messages.

        Summary Configuration:
            use_summary: Enable automatic summarization of conversation history.
            max_summary_length: Maximum length (in tokens) for generated summaries.
            max_prompt_length: Maximum total length (in tokens) for prompts including context.
            before_raw_message: Number of initial messages to preserve unmodified (>= 0).
                Messages in range [:before_raw_message] won't be affected by summary/thinking truncation.
            end_raw_message: Number of final messages to preserve unmodified (<= 0).
                Messages in range [end_raw_message:] won't be affected by summary/thinking truncation.
            summary_system_prompt: System prompt template used for generating summaries.
            oai_client: OpenAI client for summarization.
            oai_model_name: OpenAI model name for summarization.

        Model Configuration:
            train_model_tokenizer_path: Path to tokenizer for computing context window size.
    """

    simplify_thinking: bool = False

    # Summary Configuration
    use_summary: bool = False
    max_summary_length: int = Field(default=1024, gt=0)
    max_prompt_length: int = Field(default=8192, gt=0)
    before_raw_message: int = Field(default=0, ge=0)
    end_raw_message: int = Field(default=0, le=0)
    summary_system_prompt: str = Field(default=PROCEDURAL_MEMORY_SYSTEM_PROMPT)
    oai_client: Optional[OpenAI] = Field(default=None)
    oai_model_name: str = Field(default="qwen2.5-7b-instruct")

    # Model Configuration
    train_model_tokenizer_path: str = ""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_summary_length(self) -> "MemoryConfig":
        if self.max_summary_length >= self.max_prompt_length:
            raise ValueError(
                f"max_summary_length ({self.max_summary_length}) must be less than "
                f"max_prompt_length ({self.max_prompt_length})"
            )
        return self

    @validate_params(
        config=dict(
            validator=lambda x: x is None or isinstance(x, dict) or hasattr(x, "__dict__") or hasattr(x, "model_dump"),
            message="config must be a dictionary, Pydantic model, or object with attributes or None",
        )
    )
    def update(self, config: dict[str, Any] | object | None = None) -> None:
        """Update configuration parameters from a dictionary or object.

        Args:
            config: Configuration dictionary or object with attributes to update.
                Can be None (no-op), a dict, a Pydantic model or any object with a __dict__ attribute.

        Raises:
            TypeError: If config cannot be converted to a dictionary.
            AttributeError: If attempting to set a field that doesn't exist.
        """
        if config is None:
            return

        if isinstance(config, dict):
            config_dict = config
        elif hasattr(config, "model_dump"):
            # For Pydantic models, use model_dump() method
            config_dict = config.model_dump()
        elif hasattr(config, "__dict__"):
            # For regular objects with __dict__ attribute
            try:
                config_dict = vars(config)
            except TypeError as e:
                raise TypeError(f"Cannot convert config object to dictionary: {e}") from e
        else:
            raise AttributeError(f"Config object must have a __dict__ attribute.")

        # Update fields
        for key, value in config_dict.items():
            # Convert hyphens to underscores for compatibility
            normalized_key = key.replace("-", "_")

            # Only update fields that are explicitly defined in the model
            if normalized_key in self.model_fields:
                setattr(self, normalized_key, value)