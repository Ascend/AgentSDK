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
        contains_generation_msg: bool = False
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
            msg: dict,
            first_msg: bool = False,
            generation_msg: bool = False
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
            generation_msg=(contains_generation_msg and i == len(messages) - 1)
        )
        all_msg_tokens.extend(message_tokens)
        all_msg_masks.extend(message_mask)

    return all_msg_tokens, all_msg_masks