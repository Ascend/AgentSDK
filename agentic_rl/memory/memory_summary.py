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
import time
from typing import Any

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.memory.constants import (
    SUMMARY_HISTORY_PREFIX,
    SUMMARY_HISTORY_SUFFIX,
    THINK_TAG_PATTERN,
    ANSWER_TAG_START,
    ANSWER_TAG_END,
    CHUNK_BUFFER_LENGTH,
    SUMMARY,
    SYSTEM, ROLE_TOKEN_OVERHEAD
)
from agentic_rl.memory.memory_simple import MemorySimple
from agentic_rl.memory.summary_client import SummaryClient, SummaryGenerationError

logger = Loggers(__name__)


class MemorySummary(MemorySimple):
    """Memory management with automatic conversation summarization.

    Extends MemorySimple to provide automatic summarization when context exceeds configured limits.
    Uses an LLM to generate concise summaries that preserve key information while reducing token count.

    Key Features:
    - Automatic summarization when messages exceed max_prompt_length
    - Iterative summarization for very long content
    - Smart summary placement in conversation history
    - Integration with OpenAI-compatible APIs
    """

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None"
        )
    )
    def __init__(self, config: dict | None = None) -> None:
        """Initialize memory with summarization capabilities."""
        self.chat_client: SummaryClient | None = None
        super().__init__(config)

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None"
        )
    )
    def update_config(self, config: dict | None = None) -> None:
        """Update configuration and reinitialize clients if needed."""
        old_config = copy.deepcopy(self.config)
        super().update_config(config)

        self._update_chat_client_if_needed(old_config)

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None"
        )
    )
    def get_prompt_messages(self, config: dict | None = None) -> list[dict]:
        """Get messages formatted for prompt generation with automatic summarization.

        Process flow:
        1. Update configuration if provided
        2. Construct active context (Head + Active Body + Tail)
        3. Trigger summarization if needed
        4. Re-construct context with new summary
        5. Apply thinking filters
        """
        if config is not None:
            self.update_config(config)

        messages = self._get_effective_messages()

        if self.config.use_summary and self._is_overlength(messages):
            logger.info("Prompt length exceeds max_prompt_length, triggering summarization.")
            messages = self._handle_overlength()

        final_messages = self._format_summary_message(messages)

        final_messages = self._apply_thinking_filter(final_messages)

        if self._is_overlength(final_messages):
            logger.warning(
                f"PROMPT_TRUNCATION: Prompt length ({self._get_total_length(final_messages)}) "
                f"exceeds max_prompt_length ({self.config.max_prompt_length})"
            )

        return MemorySimple._remove_message_other_key(final_messages)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer content from text, removing think/answer tags"""
        text = THINK_TAG_PATTERN.sub("", text)

        answer_start = text.find(ANSWER_TAG_START)
        if answer_start != -1:
            text = text[answer_start + len(ANSWER_TAG_START):]

        answer_end = text.find(ANSWER_TAG_END)
        if answer_end != -1:
            text = text[:answer_end]

        return text.strip()

    def _get_effective_messages(self) -> list[dict]:
        """Construct the effective message list (Sandwich Strategy).

        Returns:
            List composed of [Head] + [Summary + Body] + [Tail]
        """
        messages = self.get_messages()
        if not messages:
            return []

        before_raw = self.config.before_raw_message
        end_raw = self.config.end_raw_message

        # 1. Identify Head
        head = messages[:before_raw] if before_raw > 0 else []

        # 2. Identify Last Summary
        last_summary_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == SUMMARY:
                last_summary_idx = i
                break

        # 3. Determine Body Start
        # Start after head, or at last summary if it exists and is after head
        start_idx = before_raw
        if last_summary_idx > start_idx:
            start_idx = last_summary_idx

        # 4. Determine Tail
        tail = []
        body_end = len(messages)

        if end_raw < 0:
            # Preserve last N messages
            cutoff = max(0, len(messages) + end_raw)
            if cutoff > start_idx:
                tail = messages[cutoff:]
                body_end = cutoff
            else:
                # Tail overlaps with start, priority to tail
                tail = messages[start_idx:]
                body_end = start_idx

        # 5. Construct Body
        body = messages[start_idx:body_end]

        # 6. Assemble
        effective = list(head)
        effective.extend(body)
        effective.extend(tail)

        return effective

    def _format_summary_message(self, messages: list[dict]) -> list[dict]:
        """Transform summary messages to system messages and place them correctly."""
        if not messages:
            return []

        # Add system message if not present
        result = []
        if messages[0].get("role") != SYSTEM:
            result.append({"role": SYSTEM, "content": ""})

        for msg in messages:
            if msg.get("role") == SUMMARY:
                # Convert to system message with prefix
                content = SUMMARY_HISTORY_PREFIX + msg.get("content", "") + SUMMARY_HISTORY_SUFFIX
                result[0]["content"] += content
            else:
                result.append(msg)
        return result

    def _update_chat_client_if_needed(self, old_config) -> None:
        """Initialize or update chat client when configuration changes."""
        if self.chat_client and self.config.oai_model_name == old_config.oai_model_name:
            return

        try:
            self.chat_client = SummaryClient(client=self.config.oai_client, model_name=self.config.oai_model_name)
            logger.info(f"Initialized chat client: {self.config.oai_model_name}")
        except RuntimeError as e:
            logger.error(f"Failed to initialize chat client: {e}")
            self.chat_client = None

    def _apply_thinking_filter(self, messages: list[dict]) -> list[dict]:
        """Apply thinking content filtering if configured."""
        if not self.config.simplify_thinking:
            return messages

        return MemorySimple.simplify_or_remove_think(
            messages, self.config.before_raw_message, max(0, len(messages) + self.config.end_raw_message)
        )

    def _is_overlength(self, messages: list[dict]) -> bool:
        """Check if messages exceed max prompt length."""
        total_len = self._get_total_length(messages)
        logger.info(f"total_len: {total_len}")
        return total_len > self.config.max_prompt_length

    def _get_total_length(self, messages: list[dict] | str) -> int:
        """Calculate total token length of messages or string."""
        if self.token_counter is None:
            # Should have been initialized by MemorySimple
            raise RuntimeError("Token counter not initialized.")

        if isinstance(messages, str):
            return self.token_counter.count_tokens(messages)

        if isinstance(messages, list):
            return self.get_total_length(messages)  # Uses MemorySimple cache-aware method

        raise TypeError(f"Unsupported type: {type(messages).__name__}. Expected list[dict] or str")

    def _calculate_summary_range(self, start_idx: int, messages: list[dict] | None = None) -> int:
        """Calculate how many messages from start_idx fit within max_prompt_length."""
        system_prompt = self.config.summary_system_prompt
        # Count system prompt tokens
        accumulated_len = self.token_counter.count_tokens(system_prompt) + ROLE_TOKEN_OVERHEAD

        if messages is None:
            messages = self.get_messages()
        end_idx = start_idx

        # Iterate using cached counts
        for idx in range(start_idx, max(0, len(messages) + self.config.end_raw_message)):
            # Using get_message_length is 0(1) due to cache
            msg_len = self.get_message_length(messages[idx])

            if accumulated_len + msg_len <= self.config.max_prompt_length:
                accumulated_len += msg_len
                end_idx = idx + 1
            else:
                break

        return end_idx

    def _find_summarization_start(self) -> int:
        """Find where to start summarization."""
        summary_indices = [i for i, msg in enumerate(self.get_messages()) if msg.get("role") == SUMMARY]

        start_idx = 0
        if summary_indices:
            start_idx = max(summary_indices) + 1
        else:
            # Skip initial system message if present
            messages = self.get_messages()
            if messages and messages[0].get("role") == SYSTEM:
                start_idx = 1

        # Ensure we don't start before the preserved raw messages
        return max(start_idx, self.config.before_raw_message)

    def _get_next_summary_end_idx(self, start_idx: int) -> int | None:
        """Calculate next summary end index or return None if summarization should stop."""
        messages = self.get_messages()
        end_boundary = max(0, len(messages) + self.config.end_raw_message)

        if start_idx >= end_boundary:
            return None

        end_idx = self._calculate_summary_range(start_idx, messages=messages)
        logger.info(f"start_idx: {start_idx}, end_idx: {end_idx}, len(messages): {len(messages)}")
        if end_idx < end_boundary:
            return end_idx

        # Stop if we've processed all messages
        if start_idx + 1 == end_idx and messages[start_idx].get("role") == SUMMARY and end_idx == end_boundary:
            return None

        return end_idx

    def _truncate_to_max_length(self, text: str | None) -> str:
        """Truncate text to max_summary_length tokens."""
        if self.token_counter is None:
            return text
        return self.token_counter.truncate(text, self.config.max_summary_length)

    def _log_summary_stats(self, original_messages: list[dict[str, Any]], summary: str, elapsed_time: float) -> None:
        """Log summary generation statistics."""
        original_len = self.get_total_length(original_messages)
        summary_len = self.token_counter.count_tokens(summary) + ROLE_TOKEN_OVERHEAD

        logger.info(
            f"Summary: {elapsed_time:.1f}s | "
            f"Original: {original_len} tokens | "
            f"Summary: {summary_len} tokens | "
            f"Saved: {original_len - summary_len} tokens"
        )

    def _handle_overlength(self) -> list[dict[str, Any]]:
        """Handle overlength by triggering summarization and re-extracting messages."""
        self._summarize_conversation()
        return self._get_effective_messages()

    def _summarize_conversation(self) -> None:
        """Summarize conversation when it exceeds max_prompt_length."""
        logger.info("Starting conversation summarization")

        start_idx = self._find_summarization_start()

        while True:
            end_idx = self._get_next_summary_end_idx(start_idx)
            if end_idx is None:
                break

            start_idx = self._create_summary(start_idx, end_idx)
            logger.info(f"Progress: start={start_idx}, end={end_idx}, total={len(self.get_messages())}")

        logger.info("Summarization completed")

    def _create_summary(self, start_idx: int, end_idx: int) -> int:
        """Create summary for message range and insert it."""
        messages = self.get_messages()

        # Single message too long
        if end_idx == start_idx:
            return self._summarize_single_long_message(start_idx)

        # Summary followed by single long message
        if start_idx + 1 == end_idx and messages[start_idx].get("role") == SUMMARY:
            return self._merge_summary_with_next(start_idx)

        # Normal range summarization
        return self._summarize_message_range(start_idx, end_idx)

    def _summarize_single_long_message(self, idx: int) -> int:
        """Summarize a single message that's too long."""
        if self.get_messages()[idx].get("role") == SUMMARY:
            return idx + 1

        try:
            summary_content = self._chunk_and_summarize(idx, previous_summary=None)
            self.add_message({"role": SUMMARY, "content": summary_content}, insert_id=idx + 1)
            return idx + 1
        except SummaryGenerationError as e:
            logger.warning(f"Summary generation failed for single message: {e}")
            return idx + 1

    def _merge_summary_with_next(self, summary_idx: int) -> int:
        """Merge existing summary with following long message."""
        previous_summary = self.get_messages()[summary_idx]
        try:
            summary_content = self._chunk_and_summarize(summary_idx + 1, previous_summary=previous_summary)
            self.add_message({"role": SUMMARY, "content": summary_content}, insert_id=summary_idx + 2)
            return summary_idx + 2
        except SummaryGenerationError as e:
            logger.warning(f"Summary generation failed for merge: {e}")
            return summary_idx + 2

    def _summarize_message_range(self, start_idx: int, end_idx: int) -> int:
        """Summarize a range of messages."""
        messages = self.get_messages()[start_idx:end_idx]
        try:
            summary_content = self._generate_summary(messages)
            self.add_message({"role": SUMMARY, "content": summary_content}, insert_id=end_idx)
            return end_idx
        except SummaryGenerationError as e:
            logger.warning(f"Summary generation failed for range: {e}")
            return end_idx

    def _chunk_and_summarize(self, message_idx: int, previous_summary: dict[str, Any] | None = None) -> str:
        """Chunk and iteratively summarize a single long message."""
        summary_len = 0
        if previous_summary is not None:
            summary_len = self.token_counter.count_tokens(previous_summary["content"]) + ROLE_TOKEN_OVERHEAD

        message = self.get_messages()[message_idx]
        chunk_size = self.config.max_prompt_length - self.config.max_summary_length - summary_len - CHUNK_BUFFER_LENGTH
        if chunk_size <= 0:
            # no need to summarize
            chunks = [message.get("content", "")]
        else:
            # Process content in chunks using TokenCounter splitting
            chunks = self.token_counter.split_text(message.get("content", ""), chunk_size)

        summary_content = "" if previous_summary is None else previous_summary["content"]
        for chunk_text in chunks:
            chunk_messages = [
                {"role": SYSTEM, "content": summary_content},
                {"role": message["role"], "content": chunk_text}
            ]
            summary_content = self._generate_summary(chunk_messages)

        return summary_content if len(summary_content) < len(message["content"]) else message["content"]

    def _generate_summary(self, messages: list[dict[str, Any]], system_prompt: str | None = None) -> str:
        """Generate summary using LLM."""
        if self.chat_client is None:
            raise RuntimeError("Chat client not initialized.")

        start_time = time.time()
        logger.info(f"Generating summary for {len(messages)} messages.")

        # Prepare prompt
        system_prompt = system_prompt or self.config.summary_system_prompt
        prompt_content = system_prompt.replace("{max_summary_length}", str(self.config.max_summary_length))

        raw_messages = [{"role": SYSTEM, "content": prompt_content}, *messages]
        prompt_messages = self._format_summary_message(raw_messages)
        prompt_messages = MemorySimple._remove_message_other_key(prompt_messages)

        response = self.chat_client.generate_chat_completion(
            prompt_messages, max_tokens=self.config.max_summary_length
        )

        # Extract and truncate
        summary = self._extract_answer(response)
        summary = self._truncate_to_max_length(summary)
        self._log_summary_stats(messages, summary, time.time() - start_time)
        return summary
