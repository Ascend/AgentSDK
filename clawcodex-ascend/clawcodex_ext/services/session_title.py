#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# -------------------------------------------------------------------------
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""Session title generation matching TypeScript session/title.ts."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_TITLE_LENGTH = 80


def auto_title_from_message(text: str) -> str:
    """Extract a title from the first user message.

    Truncates to MAX_TITLE_LENGTH chars, removes newlines,
    strips leading/trailing whitespace and punctuation.
    """
    if not text:
        return "Untitled session"

    # Take first line or first sentence
    lines = text.strip().split("\n")
    first_line = lines[0].strip()

    # Remove common prefixes
    for prefix in ("please ", "can you ", "help me ", "i need ", "i want "):
        if first_line.lower().startswith(prefix):
            first_line = first_line[len(prefix) :]
            break

    # Capitalize first letter
    if first_line and first_line[0].islower():
        first_line = first_line[0].upper() + first_line[1:]

    # Truncate
    if len(first_line) > MAX_TITLE_LENGTH:
        first_line = first_line[: MAX_TITLE_LENGTH - 3].rstrip() + "..."

    return first_line or "Untitled session"


async def generate_llm_title(
    messages: list[dict[str, Any]],
    *,
    model: str = "claude-3-5-haiku-20241022",
) -> str | None:
    """Generate a 5-10 word title using a side LLM query.

    Returns None if the LLM call fails.
    """
    try:
        import anthropic

        # Build a concise prompt
        msg_summaries = []
        for msg in messages[:5]:  # First 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(texts)
            if isinstance(content, str):
                content = content[:200]
            msg_summaries.append(f"{role}: {content}")

        summary = "\n".join(msg_summaries)

        from src.services.api.custom_headers import get_anthropic_custom_headers

        client = anthropic.AsyncAnthropic(default_headers=get_anthropic_custom_headers() or None)
        response = await client.messages.create(
            model=model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a concise 5-10 word title for this conversation. "
                        "Return ONLY the title, no quotes or punctuation.\n\n"
                        f"{summary}"
                    ),
                }
            ],
        )

        if response.content and len(response.content) > 0:
            title = response.content[0].text.strip()
            # Clean up
            title = title.strip("\"'")
            if len(title) > MAX_TITLE_LENGTH:
                title = title[: MAX_TITLE_LENGTH - 3] + "..."
            return title

    except Exception as e:
        logger.debug("LLM title generation failed: %s", e)

    return None
