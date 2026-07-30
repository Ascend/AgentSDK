"""Pipe IPC data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipeMessageType(str, Enum):
    HEARTBEAT = "heartbeat"
    COMMAND = "command"
    REPLY = "reply"
    BROADCAST = "broadcast"
    PERMISSION_REQ = "permission_req"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_DENY = "permission_deny"
    PEER_JOIN = "peer_join"
    PEER_LEAVE = "peer_leave"
    AGENT_STREAM = "agent_stream"


@dataclass
class PipeMessage:
    type: PipeMessageType
    source_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    ttl: int = 16
    permission_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "permission_token": self.permission_token,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipeMessage:
        if not isinstance(data, dict):
            raise TypeError("PipeMessage data must be a JSON object")

        try:
            message_type = PipeMessageType(str(data["type"]))
            source_id = str(data["source_id"])
        except KeyError as exc:
            raise ValueError(f"Missing PipeMessage field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise ValueError("Invalid PipeMessage type") from exc

        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise TypeError("PipeMessage payload must be a JSON object")

        target_id = data.get("target_id")
        permission_token = data.get("permission_token")
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            type=message_type,
            source_id=source_id,
            target_id=str(target_id) if target_id is not None else None,
            payload=payload,
            timestamp=float(data.get("timestamp", time.time())),
            ttl=int(data.get("ttl", 16)),
            permission_token=str(permission_token) if permission_token is not None else None,
        )


@dataclass
class PipePeer:
    instance_id: str
    hostname: str
    pid: int
    version: str = ""
    addr: str = ""
    transport: str = "uds"
    last_seen: float = field(default_factory=time.time)
    is_master: bool = False
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "version": self.version,
            "addr": self.addr,
            "transport": self.transport,
            "last_seen": self.last_seen,
            "is_master": self.is_master,
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipePeer:
        if not isinstance(data, dict):
            raise TypeError("PipePeer data must be a JSON object")

        try:
            instance_id = str(data["instance_id"])
            hostname = str(data["hostname"])
            pid = int(data["pid"])
        except KeyError as exc:
            raise ValueError(f"Missing PipePeer field: {exc.args[0]}") from exc

        capabilities = data.get("capabilities", [])
        if not isinstance(capabilities, list):
            raise TypeError("PipePeer capabilities must be a list")

        return cls(
            instance_id=instance_id,
            hostname=hostname,
            pid=pid,
            version=str(data.get("version", "")),
            addr=str(data.get("addr", "")),
            transport=str(data.get("transport", "uds")),
            last_seen=float(data.get("last_seen", time.time())),
            is_master=bool(data.get("is_master", False)),
            capabilities=[str(item) for item in capabilities],
        )
