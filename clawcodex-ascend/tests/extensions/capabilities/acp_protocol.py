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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""ACP Protocol — Agent Client Protocol contract.

Defines the data model and Protocol interfaces for the Agent Client
Protocol used by Zed / Cursor / Trae IDE integrations. Pure contract
layer (Layer 2): no runtime dependencies, only signatures. Concrete
transports and servers live in ``extensions/trae/`` and future
``extensions/zed`` / ``extensions/cursor`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Protocol, runtime_checkable

__all__ = [
    "ACPMessageType",
    "ACPMessageRole",
    "ACPMessage",
    "ACPSession",
    "ACPTransport",
    "ACPServer",
    "ACPToolSpec",
]


class ACPMessageType(str, Enum):
    """ACP message-type enum (JSON-RPC method namespace)."""

    SESSION_CREATE = "session/create"
    SESSION_RESUME = "session/resume"
    SESSION_END = "session/end"
    MESSAGE_SEND = "message/send"
    MESSAGE_STREAM = "message/stream"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
    SKILL_INVOKE = "skill/invoke"
    SKILL_RESULT = "skill/result"
    ERROR = "error"


class ACPMessageRole(str, Enum):
    """Message roles (aligned with OpenAI/Anthropic chat roles)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


def _utc_now_iso() -> str:
    """Timezone-aware UTC ISO timestamp (avoids deprecated ``datetime.utcnow``)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ACPMessage:
    """ACP protocol message body (JSON-RPC over WebSocket/stdio); ``content`` may
    be str / dict / None (text or multimodal blocks).
    """

    type: ACPMessageType
    id: str = ""
    session_id: str = ""
    role: ACPMessageRole = ACPMessageRole.USER
    content: str | dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (enum -> str)."""
        return {
            "type": self.type.value,
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACPMessage":
        """Deserialize from a dict (str -> enum, tolerant of missing fields)."""
        raw_type = data.get("type", "")
        try:
            msg_type = ACPMessageType(raw_type)
        except ValueError:
            msg_type = ACPMessageType.ERROR
            data.setdefault("metadata", {})["unknown_type"] = raw_type
        raw_role = data.get("role", "user")
        try:
            role = ACPMessageRole(raw_role)
        except ValueError:
            role = ACPMessageRole.USER
        return cls(
            type=msg_type,
            id=data.get("id", ""),
            session_id=data.get("session_id", ""),
            role=role,
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            tool_results=data.get("tool_results"),
            metadata=data.get("metadata", {}) or {},
            timestamp=data.get("timestamp", _utc_now_iso()),
        )


@dataclass
class ACPSession:
    """ACP session state (held by the server, serializable)."""

    id: str
    created_at: str = field(default_factory=_utc_now_iso)
    messages: list[ACPMessage] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    workspace_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, msg: ACPMessage) -> None:
        """Record a message in this session's history."""
        self.messages.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.messages],
            "skills": list(self.skills),
            "workspace_path": self.workspace_path,
            "metadata": self.metadata,
        }


@dataclass
class ACPToolSpec:
    """ACP tool spec (used in ``tools/list`` responses)."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ACPTransport(Protocol):
    """ACP transport abstraction (stdio / WebSocket / TCP).

    Implementers frame :class:`ACPMessage` on the wire and deserialize
    in :meth:`receive`; ``None`` means stream end.
    """

    async def connect(self) -> None: ...

    async def send(self, msg: ACPMessage) -> None: ...

    async def receive(self) -> ACPMessage | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class ACPServer(Protocol):
    """ACP protocol server (accepts session requests from the IDE).

    ``process_message`` returns an async iterator for streaming
    (``message/stream``); ``invoke_skill`` is the sync skill entry.
    """

    async def handle_session(self, transport: ACPTransport) -> None: ...

    async def create_session(self, workspace_path: str) -> ACPSession: ...

    async def resume_session(self, session_id: str) -> ACPSession | None: ...

    def process_message(self, msg: ACPMessage) -> AsyncIterator[ACPMessage]: ...

    async def invoke_skill(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]: ...
