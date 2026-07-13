#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from types import SimpleNamespace
from typing import Any


class MemorySummary:
    """Lightweight chat message container for agent prompt assembly."""

    def __init__(self, config: dict[str, Any] | None = None, tokenizer=None):
        self.messages: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(
            use_summary=False,
            max_prompt_length=8192,
            chat_model_name="qwen3_4b",
            train_model_tokenizer_path=None,
        )
        self.update_configs(config or {})

    def update_configs(self, config: dict[str, Any]) -> None:
        valid_keys = vars(self.config)
        for key, value in config.items():
            if key in valid_keys:
                setattr(self.config, key, value)

    def clear_memory(self, role: str = "system", content: str = "") -> None:
        self.messages = []
        if content:
            self.messages.append({"role": role, "content": content})

    def add_message(self, message, metadata=None, insert_id: int | None = None) -> None:
        if message is None:
            return
        messages = message if isinstance(message, list) else [message]
        normalized = [dict(item) for item in messages if isinstance(item, dict)]
        if metadata is not None:
            for item in normalized:
                item["metadata"] = metadata
        if insert_id is None:
            self.messages.extend(normalized)
        else:
            for offset, item in enumerate(normalized):
                self.messages.insert(insert_id + offset, item)

    def get_prompt_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)
