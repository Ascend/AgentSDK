#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Process-local state for the Responses API."""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class StoredResponse:
    response: dict[str, Any]
    messages: list[Any]
    input_items: list[dict[str, Any]]
    conversation: str | None = None
    session_id: str | None = None


class ResponseStore:
    """Small LRU store for response chains and named conversations."""

    def __init__(self, limit: int = 128) -> None:
        self.limit = max(1, int(limit))
        self._items: OrderedDict[str, StoredResponse] = OrderedDict()
        self._conversations: dict[str, str] = {}
        self._lock = threading.RLock()

    def get(self, response_id: str) -> StoredResponse | None:
        with self._lock:
            item = self._items.get(response_id)
            if item is None:
                return None
            self._items.move_to_end(response_id)
            return item

    def put(
        self,
        response_id: str,
        response: dict[str, Any],
        messages: list[Any],
        input_items: list[dict[str, Any]] | None = None,
        *,
        conversation: str | None = None,
        session_id: str | None = None,
    ) -> None:
        with self._lock:
            previous = self._items.get(response_id)
            if (
                previous is not None
                and previous.conversation
                and previous.conversation != conversation
                and self._conversations.get(previous.conversation) == response_id
            ):
                self._conversations.pop(previous.conversation, None)
            self._items[response_id] = StoredResponse(
                response=copy.deepcopy(response),
                messages=list(messages),
                input_items=list(input_items or []),
                conversation=conversation,
                session_id=session_id,
            )
            self._items.move_to_end(response_id)
            if conversation:
                self._conversations[conversation] = response_id
            self._evict_if_needed()

    def delete(self, response_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(response_id, None) is not None
            if existed:
                stale = [name for name, stored_id in self._conversations.items() if stored_id == response_id]
                for name in stale:
                    self._conversations.pop(name, None)
            return existed

    def latest_for_conversation(self, conversation: str) -> StoredResponse | None:
        with self._lock:
            response_id = self._conversations.get(conversation)
            if not response_id:
                return None
            return self.get(response_id)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "responses": len(self._items),
                "conversations": len(self._conversations),
                "limit": self.limit,
            }

    def _evict_if_needed(self) -> None:
        while len(self._items) > self.limit:
            response_id, _ = self._items.popitem(last=False)
            stale = [name for name, stored_id in self._conversations.items() if stored_id == response_id]
            for name in stale:
                self._conversations.pop(name, None)
