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

import time
from typing import Any, Optional

import openai
from openai import OpenAI

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.memory.utils import validate_message

logger = Loggers(__name__)


class SummaryGenerationError(Exception):
    """Raised when summary generation fails."""

    pass


class SummaryClient:
    """Handles API communication for summary generation.

    This class manages OpenAI-compatible API clients for generating summaries,
    with built-in retry logic and error handling.

    Attributes:
        client: OpenAI client instance.
        model_name: Name of the model to use for completion.
    """

    def __init__(self, client: Optional[OpenAI], model_name: str) -> None:
        self.client: OpenAI = None
        self.model_name: str = model_name
        self.update_config(client, model_name)

    @validate_params(
        messages=dict(validator=validate_message, message="messages must be a list of dictionaries"),
        max_retries=dict(
            validator=lambda x: isinstance(x, int) and x > 0, message="max_retries must be a positive integer"
        ),
        retry_delay=dict(
            validator=lambda x: isinstance(x, int) and x > 0, message="retry_delay must be a positive integer"
        ),
        timeout=dict(validator=lambda x: isinstance(x, int) and x > 0, message="timeout must be a positive integer"),
    )
    def generate_chat_completion(
        self, messages: list[dict], max_retries: int = 3, retry_delay: int = 5, timeout: int = 3600, **kwargs: Any
    ) -> str:
        """Generate a completion using the OpenAI Chat API with retry logic.

        Args:
            messages: List of message dictionaries (role/content).
            max_retries: Maximum number of retry attempts on failure (default: 3).
            retry_delay: Delay in seconds between retry attempts (default: 5).
            timeout: Request timeout in seconds (default: 3600).
            **kwargs: Additional arguments to pass to the API (e.g., max_tokens, temperature).

        Returns:
            The generated completion text (content of the first choice).
        """
        if self.client is None:
            raise RuntimeError("Client is not initialized.")

        def get_response(retries: int) -> str:
            while retries > 0:
                try:
                    response = self.client.chat.completions.create(
                        messages=messages, model=self.model_name, timeout=timeout, **kwargs
                    )
                    logger.info("Generate chat completion response for summarization.")
                    return response.choices[0].message.content or ""
                except openai.RateLimitError as e:
                    retries -= 1
                    if retries == 0:
                        logger.error("Rate limit reached and all retry attempts exhausted.")
                        raise SummaryGenerationError("Rate limit reached and all retry attempts exhausted.") from e
                    logger.warning(f"Rate limit hit. Will retry in {retry_delay} seconds... ({retries} retries left)")
                    time.sleep(retry_delay)
                except openai.APITimeoutError as e:
                    retries -= 1
                    if retries == 0:
                        logger.error("API timeout and all retry attempts exhausted.")
                        raise SummaryGenerationError("API timeout and all retry attempts exhausted.") from e
                    logger.warning(f"API timeout. Will retry in {retry_delay} seconds... ({retries} retries left)")
                    time.sleep(retry_delay)
                except Exception as e:
                    logger.error(f"Unexpected error during completion generation: {e}")
                    raise SummaryGenerationError(f"Unexpected error during completion generation: {e}") from e
            raise SummaryGenerationError("All retry attempts exhausted.")

        return get_response(max_retries)

    def update_config(self, client: OpenAI, model_name: str) -> None:
        if not isinstance(client, OpenAI):
            raise TypeError("client must be an instance of OpenAI")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.client = client
        self.model_name = model_name
