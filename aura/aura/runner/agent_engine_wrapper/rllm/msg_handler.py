#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import copy
import json
from typing import List, Tuple

from transformers import PreTrainedTokenizerBase

from rllm.parser.chat_template.parser import ChatTemplateParser


def get_recent_assistant_user_messages(chat_completions_messages: List[dict]) -> Tuple[dict, List[dict]]:
    """
    Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.

    Args:
        chat_completions_messages (List[Dict]): List of message dictionaries from chat completions.

    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing:
            - The most recent assistant message (or None if not found)
            - A list of environment messages (user/tool) that occurred after the last assistant message,
              in chronological order.
    """
    env_messages = []
    assistant_message = None
    seen_assistant_message = False

    for message in reversed(chat_completions_messages):
        role = message.get("role", None)
        if role == "assistant":
            if assistant_message:
                break
            seen_assistant_message = True
            assistant_message = message
        elif role in ["user", "tool"] and not seen_assistant_message:
            env_messages.append(message)

    env_messages = list(reversed(env_messages))

    return assistant_message, env_messages


def convert_messages_to_tokens_and_masks(
    messages: List[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    parser: ChatTemplateParser,
    contains_first_msg: bool = False,
    contains_generation_msg: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Converts multiple messages to tokens and masks.
    contains_first_msg flag and contains_generation_msg flag are used to indicate
    whether the conversation is for beginning or contains the generation.
    The first and last message is assumed to be the special message respectively

    Args:
        messages (List[Dict]): The messages to convert.
        tokenizer: The tokenizer to use.
        parser: chat parser
        contains_first_msg (bool): Whether the first message is a special message.
        contains_generation_msg (bool): Whether the last message is a special message.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing all tokens and all masks.
    """
    all_msg_tokens = []
    all_msg_masks = []

    def _convert_message_to_tokens_and_masks(
        msg: dict, first_msg: bool = False, generation_msg: bool = False
    ) -> Tuple[List[int], List[int]]:
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)

        if msg["role"] == "assistant":
            if not msg_text.startswith(parser.assistant_token):
                raise Exception(f"Expected assistant token {parser.assistant_token} but got {msg_text}")
            msg_text = msg_text.replace(parser.assistant_token, "")

        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        return msg_tokens, msg_mask

    for i, message in enumerate(messages):
        message_tokens, message_mask = _convert_message_to_tokens_and_masks(
            message,
            first_msg=(contains_first_msg and i == 0),
            generation_msg=(contains_generation_msg and i == len(messages) - 1),
        )
        all_msg_tokens.extend(message_tokens)
        all_msg_masks.extend(message_mask)

    return all_msg_tokens, all_msg_masks


def preprocess_messages_for_qwen35(messages):
    """
    Convert all tool_calls.function.arguments in messages from JSON strings to dicts to adapt to
    the .items() call in the Qwen 3.5 template
    """
    new_messages = copy.deepcopy(messages)
    for msg in new_messages:
        if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to decode arguments: {args}")
    return new_messages


def parse_whole_response_token_ids(
    parser: ChatTemplateParser, messages, tools, assistant_token, enable_interleaved_think=False
):
    processed_messages = preprocess_messages_for_qwen35(messages)
    asst_start_len = len(parser.tokenizer.encode(assistant_token, add_special_tokens=False))
    first_asst_idx = next(
        (i for i, m in enumerate(processed_messages) if m["role"] == "assistant"), len(processed_messages)
    )

    prompt_str = parser.parse(
        processed_messages[:first_asst_idx], tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt_token_ids = parser.tokenizer.encode(prompt_str, add_special_tokens=False)

    if enable_interleaved_think:
        interleaved_think_msg = copy.deepcopy(processed_messages)
        for msg in interleaved_think_msg:
            if msg["role"] == "assistant" and "<think>" in msg.get("content", ""):
                msg["content"] = (
                    msg["content"].replace("<think>", "<past_thought>").replace("</think>", "</past_thought>")
                )
        full_str = parser.parse(interleaved_think_msg, tools=tools, tokenize=False, add_generation_prompt=False)
        full_str = full_str.replace("<past_thought>", "<think>").replace("</past_thought>", "</think>")
    else:
        full_str = parser.parse(processed_messages, tools=tools, tokenize=False, add_generation_prompt=False)

    full_token_ids = parser.tokenizer.encode(full_str, add_special_tokens=False)
    full_mask = [0] * len(full_token_ids)
    current_pos = len(prompt_token_ids)

    is_first = True
    for i in range(first_asst_idx + 1, len(processed_messages) + 1):
        if enable_interleaved_think:
            interleaved_think_msg = copy.deepcopy(processed_messages)
            for msg in interleaved_think_msg:
                if msg["role"] == "assistant" and "<think>" in msg.get("content", ""):
                    msg["content"] = (
                        msg["content"].replace("<think>", "<past_thought>").replace("</think>", "</past_thought>")
                    )
            sub_prompt = parser.parse(
                interleaved_think_msg[:i], tools=tools, tokenize=False, add_generation_prompt=False
            )
            sub_prompt = sub_prompt.replace("<past_thought>", "<think>").replace("</past_thought>", "</think>")
        else:
            sub_prompt = parser.parse(processed_messages[:i], tools=tools, tokenize=False, add_generation_prompt=False)

        sub_ids = parser.tokenizer.encode(sub_prompt, add_special_tokens=False)
        start_idx = current_pos
        end_idx = len(sub_ids)
        if processed_messages[i - 1]["role"] == "assistant":
            if is_first:
                actual_content_start = min(start_idx, end_idx)
                is_first = False
            else:
                actual_content_start = min(start_idx + asst_start_len, end_idx)

            # Only the assistant content within the response area is marked as 1
            for j in range(actual_content_start, end_idx):
                full_mask[j] = 1

        current_pos = end_idx

    # response_mask has automatically excluded the Prompt area (because the Prompt area corresponds to i <= first_asst_idx).
    response_token_ids = full_token_ids[len(prompt_token_ids) :]
    response_mask = full_mask[len(prompt_token_ids) :]

    if len(response_token_ids) != len(response_mask):
        raise ValueError(f"Length mismatch: ids({len(response_token_ids)}) != mask({len(response_mask)})")
    return prompt_token_ids, response_token_ids, response_mask
