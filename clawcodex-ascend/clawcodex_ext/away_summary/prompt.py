#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Prompt construction for Away Summary.

Two distinct instruction templates are maintained, mirroring the
canonical Claude Code split between the *automatic* "while you were away"
card and the *manual* ``/recap`` command. The automatic variant is
slightly more relaxed (1-3 sentences, narrative-style) so the prompt-card
reads naturally when shown alongside the transcript; the manual variant
is strict (1-2 sentences, three-part structure) so a user who types
``/recap`` gets a tight recap they can scan in one breath.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

# Stable sentinel from ``clawcodex_ext.types.messages``. Keep this Part
# importable until the shared types migration lands.
_NO_CONTENT_MESSAGE = "[No content]"
_CJK_MIN_CHARS = 3
_CJK_TO_LATIN_RATIO = 0.25
# Keep the newest context dominant because it normally contains the current
# task state; retain a smaller leading slice so the original goal is not lost.
_TRUNCATION_HEAD_FRACTION = 0.2


def is_away_summary_message(message: Any) -> bool:
    """Load the same-Part fingerprint helper without package marker overlap."""

    module_name = "clawcodex_ext.away_summary.fingerprint"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return False
        raise
    predicate = getattr(module, "is_away_summary_message", None)
    return bool(predicate(message)) if callable(predicate) else False


# ---------------------------------------------------------------------------
# Manual trigger: /recap command
# ---------------------------------------------------------------------------

_COMMON_FORMAT = """- Then add a short bullet list (1-3 items) with whatever context matters most: the most useful next action, files touched, tools used, or anything else the user needs to pick up quickly.
- On the line immediately before the bullet list, add ONE short phrase that explains the intent of the bullets. Use a plain phrase, not a label ending in a colon. Examples: `后续计划` / `后续步骤` / `Next steps` / `What next` / `To wrap up`. Do NOT use a colon.
- Use `-` as the bullet marker for both English and Chinese.
- Use plain text only. Do NOT use fixed section labels (e.g. labels ending in a colon that name a category). Let the bullets speak for themselves.
- No headings, no bold, no markdown beyond the bullet markers.

Content guidance (consider these dimensions, but do not label them):
- High-level goal: what the user is building or debugging, NOT implementation details.
- Where they left off: the last meaningful state.
- The single most useful next action.
- Files touched or tools used by the assistant, if any."""

_COMMON_EXAMPLE = """Example:
We're debugging why away-summary messages now render in white with less content compared to the earlier light-gray, fuller recaps. The assistant inspected tui/screens/repl.py and confirmed the muted style is still applied.
后续计划
- tui/screens/repl.py
- Read(tui/screens/repl.py), Grep
- Compare the fallback formatter against the earlier structured-label version."""

_COMMON_RULES = """- The recap MUST be written in the language specified above. Do not switch languages mid-recap.
- The first sentence should sound like a handoff, not a status report.
- Do NOT start the recap with a preamble such as "你刚回来，这是之前的会话摘要：", "Here's a summary:", or any meta-introduction about what the text is.
- Use `-` and ONLY `-` as the bullet marker. Do NOT use `•`, `*`, `·`, or numbered bullets.
- Do NOT add a heading or label before the bullet list; only the single intent phrase is allowed, and it must NOT end with a colon.
- Skip root-cause narrative, internal fix details, secondary to-dos, and em-dash tangents.
- Do not mention that you are an AI.
- Do NOT output any internal chain-of-thought, planning notes, or self-checks
  — even if your prompt normally does so. In particular:
    * Never start your reply with "Here's a thinking process", "思考过程:",
      "Thinking process:", "Let me think", or any similar preamble.
    * Never enclose the recap in <think>…</think>, <thinking>…</thinking>, or
      any other reasoning tags.
    * Never include numbered "step 1: analyze, step 2: identify, step 3:
      draft, step 4: check constraints" self-audit scaffolding.
- Return only the recap."""

AWAY_SUMMARY_INSTRUCTIONS = f"""The user stepped away from an interactive coding session and is coming back. Write a brief, natural session recap.

Format:
- 1-2 short, flowing sentences that name the high-level goal and where things stand.
{_COMMON_FORMAT}
- Under 60 words (English) / 90 Chinese characters (中文).

{_COMMON_EXAMPLE}

{{language_instruction}}

Rules:
{_COMMON_RULES}"""


# ---------------------------------------------------------------------------
# Auto trigger: idle 5-min "while you were away" card
# ---------------------------------------------------------------------------

AWAY_SUMMARY_INSTRUCTIONS_AUTO = f"""The user stepped away from an interactive coding session and is coming back. Write a brief, natural session recap for the "while you were away" card.

Format:
- 1-3 short, flowing sentences that name the high-level goal and where things stand.
{_COMMON_FORMAT}
- Under 80 words (English) / 120 Chinese characters (中文).

Additional content guidance:
- If a broader session memory block is provided below, you may weave in the long-running project context so the recap reads like a project handoff rather than a status report.

{_COMMON_EXAMPLE}

{{language_instruction}}

Rules:
- Skip status reports and commit recaps; the user does not want a re-narration of every step that already happened.
{_COMMON_RULES}"""


_LANGUAGE_INSTRUCTION_MAP: dict[str, str] = {
    "Chinese": "MUST write the recap in natural Simplified Chinese.",
    "English": "MUST write the recap in natural English.",
}


def build_summary_messages(
    conversation: Any,
    *,
    max_input_tokens: int,
    response_language: str | None = None,
    trigger: str = "manual",
    memory: str | None = None,
) -> list[dict[str, str]]:
    """Build the (system, user) message pair for the recap request.

    Args:
        conversation: The live conversation whose transcript is summarised.
        max_input_tokens: Token budget for the truncated transcript.
        response_language: Optional explicit language override
            (``"Chinese"`` / ``"English"``). Falls back to
            :func:`infer_response_language`.
        trigger: ``"auto"`` for the idle card (1-3 sentences, narrative),
            ``"manual"`` for ``/recap`` (1-2 sentences, three-part). Any
            other value is treated as ``"manual"`` to preserve backward
            compatibility with callers that pre-date this parameter.
        memory: Optional broader session-memory block. When provided and
            ``trigger == "auto"`` it is prepended to the user message so
            the model can weave long-running project context into the
            recap. ``/recap`` ignores it (manual recaps already see the
            full transcript).
    """
    lang = infer_response_language(conversation, response_language)
    lang_instruction = _LANGUAGE_INSTRUCTION_MAP.get(lang, "Write the recap in natural English.")
    is_auto = trigger == "auto"

    template = AWAY_SUMMARY_INSTRUCTIONS_AUTO if is_auto else AWAY_SUMMARY_INSTRUCTIONS
    system_content = template.format(language_instruction=lang_instruction)

    transcript = _serialize_transcript(conversation)
    transcript = _truncate_transcript(transcript, max_input_tokens=max_input_tokens)

    if is_auto and memory:
        user_content = (
            "Write the recap based on the transcript and broader session "
            "memory below.\n\n"
            f"Session memory (broader context):\n{memory}\n\n"
            f"Session transcript:\n{transcript}\n\n"
            "Return only the recap: 1-3 short sentences, one intent phrase, "
            "and 1-3 `-` bullets; use no other Markdown."
        )
    elif is_auto:
        user_content = (
            "Write the recap based on the transcript below.\n\n"
            f"Session transcript:\n{transcript}\n\n"
            "Return only the recap: 1-3 short sentences, one intent phrase, "
            "and 1-3 `-` bullets; use no other Markdown."
        )
    else:
        user_content = (
            "Write the recap based on the transcript below.\n\n"
            f"Session transcript:\n{transcript}\n\n"
            "Return only the recap: 1-2 short sentences, one intent phrase, "
            "and 1-3 `-` bullets; use no other Markdown."
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def infer_response_language(
    conversation: Any,
    explicit: str | None = None,
) -> str:
    """Infer whether the recap should be Chinese or English.

    Priority:
    1. Explicit override (e.g. from config ``response_language``).
    2. CJK character frequency in recent user and assistant messages.
       User messages are the primary signal, but assistant messages are
       used as a fallback when user turns are short or dominated by
       English identifiers/paths (common in coding sessions).
    3. Default to English.
    """
    if explicit and explicit in {"Chinese", "English"}:
        return explicit

    user_samples: list[str] = []
    assistant_samples: list[str] = []
    for msg in reversed(getattr(conversation, "messages", [])):
        role = getattr(msg, "role", "")
        text = _content_to_text(getattr(msg, "content", "")).strip()
        if not text:
            continue
        if role == "user":
            user_samples.append(text)
        elif role == "assistant":
            assistant_samples.append(text)
        if len(user_samples) >= 6 and len(assistant_samples) >= 6:
            break

    user_lang = _detect_language(user_samples)
    assistant_lang = _detect_language(assistant_samples)

    # If either side is clearly Chinese, prefer Chinese. This avoids false
    # negatives when user turns are dominated by English identifiers/paths
    # but the assistant has already been replying in Chinese.
    if user_lang == "Chinese" or assistant_lang == "Chinese":
        return "Chinese"

    # Only fall back to English when both sides are clearly English.
    if user_lang == "English" and assistant_lang == "English":
        return "English"

    # If one side is ambiguous, trust the side with a clear signal.
    if user_lang is not None:
        return user_lang
    if assistant_lang is not None:
        return assistant_lang

    return "English"


def _detect_language(samples: list[str]) -> str | None:
    """Return 'Chinese' or 'English' if the samples are clearly in one language."""
    if not samples:
        return None
    text = "\n".join(samples)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if cjk >= _CJK_MIN_CHARS and cjk >= latin * _CJK_TO_LATIN_RATIO:
        return "Chinese"
    if latin > 0 and cjk / max(latin, 1) < _CJK_TO_LATIN_RATIO:
        return "English"
    return None


def _serialize_transcript(conversation: Any) -> str:
    lines: list[str] = []
    for msg in getattr(conversation, "messages", []):
        if is_away_summary_message(msg):
            continue
        role = getattr(msg, "role", "")
        if role not in {"user", "assistant"}:
            continue
        text = _content_to_text(getattr(msg, "content", ""))
        if not text.strip():
            continue
        lines.append(f"{role}: {text.strip()}")
    return "\n\n".join(lines)


def _truncate_transcript(transcript: str, *, max_input_tokens: int) -> str:
    if _estimate_tokens(transcript) <= max_input_tokens:
        return transcript
    marker = "\n\n[... earlier middle of session omitted for recap budget ...]\n\n"
    if _estimate_tokens(marker) >= max_input_tokens:
        tail_chars = max(1, max_input_tokens * 4)
        while tail_chars > 1 and _estimate_tokens(transcript[-tail_chars:]) > max_input_tokens:
            tail_chars = max(1, tail_chars * 3 // 4)
        return transcript[-tail_chars:]
    budget_chars = max_input_tokens * 4
    while True:
        content_chars = max(0, budget_chars - len(marker))
        head_chars = int(content_chars * _TRUNCATION_HEAD_FRACTION)
        tail_chars = content_chars - head_chars
        head = transcript[:head_chars].rstrip()
        tail = transcript[-tail_chars:].lstrip() if tail_chars else ""
        truncated = head + marker + tail
        if _estimate_tokens(truncated) <= max_input_tokens or budget_chars <= 1:
            return truncated
        budget_chars = max(1, budget_chars * 3 // 4)


def _estimate_tokens(text: str) -> int:
    try:
        module = import_module("clawcodex_ext.utils.token_estimation")
        count_tokens = getattr(module, "count_tokens")
        return int(count_tokens(text))
    except Exception:
        return max(1, len(text) // 4)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return "" if content.strip() == _NO_CONTENT_MESSAGE else content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                value = str(text)
                if value.strip() != _NO_CONTENT_MESSAGE:
                    parts.append(value)
                continue
            kind = getattr(item, "type", None)
            if kind == "tool_use":
                name = getattr(item, "name", "")
                raw_input = getattr(item, "input", {})
                parts.append(f"[tool_use {name} {json.dumps(raw_input, default=str)}]")
                continue
            if kind == "tool_result":
                parts.append(f"[tool_result {getattr(item, 'content', '')}]")
                continue
            if isinstance(item, dict):
                if item.get("type") in (None, "text"):
                    value = str(item.get("text") or item.get("content") or "")
                    if value.strip() != _NO_CONTENT_MESSAGE:
                        parts.append(value)
                elif item.get("type") == "tool_use":
                    parts.append(f"[tool_use {item.get('name') or ''}]")
                elif item.get("type") == "tool_result":
                    parts.append(f"[tool_result {item.get('content') or ''}]")
                continue
            parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)
