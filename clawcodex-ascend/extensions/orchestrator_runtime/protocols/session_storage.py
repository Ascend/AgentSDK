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

"""SessionStorage protocol.

Persist agent sessions and restore them across restarts. A Clawcodex
backend can wrap ``clawcodex_ext.services.session_storage.SessionStorage``.
"""

from __future__ import annotations
# pylint: disable=W2301

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionMeta(Protocol):
    """Structural metadata for a persisted session."""

    session_id: str
    workspace: Path
    created_at: str  # ISO-8601


@runtime_checkable
class ConversationLike(Protocol):
    """Structural type for a serializable conversation.

    Compatible with ``clawcodex_ext.agent.conversation.Conversation``:
    ``messages``, ``provider``, ``model`` are accessible as attributes.
    """

    messages: list[Any]
    provider: str | None
    model: str | None


class SessionStorage(Protocol):
    """Persist + recover agent sessions across orchestrator restarts."""

    def save(self, session_id: str, conversation: ConversationLike) -> None: ...

    def load(self, session_id: str) -> ConversationLike | None: ...

    def list_sessions(
        self,
        workspace: Path | None = None,
    ) -> list[SessionMeta]: ...

    def session_dir(self) -> Path:
        """Return the canonical sessions directory (e.g. ``~/.codex/sessions/``)."""
        ...


__all__ = ["ConversationLike", "SessionMeta", "SessionStorage"]
