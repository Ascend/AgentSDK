"""CCR remote session client for ultraplan execution."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from typing_extensions import Self

from .exceptions import CCRUnavailableError
from .models import Plan


class CCREventType(str, Enum):
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    PLAN_COMPLETED = "plan.completed"
    PLAN_FAILED = "plan.failed"
    LOG = "log"


@dataclass(frozen=True)
class CCRSession:
    id: str
    plan_id: str
    status: str
    endpoint: str


@dataclass(frozen=True)
class CCREvent:
    type: CCREventType
    session_id: str
    timestamp: str
    payload: dict[str, Any]


class CCRClient:
    def __init__(self, endpoint: str, *, token: str | None = None, timeout: float = 30.0) -> None:
        if not endpoint:
            raise CCRUnavailableError("CCR endpoint is required")
        try:
            import httpx
        except Exception as exc:
            raise CCRUnavailableError("httpx is required for CCR remote sessions") from exc
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def start_session(self, plan: Plan, *, cwd: str) -> CCRSession:
        response = await self._client.post(
            f"{self._endpoint}/v1/ultraplan",
            json={"plan": plan.to_dict(), "cwd": cwd},
        )
        response.raise_for_status()
        data = response.json()
        return CCRSession(
            id=str(data["id"]),
            plan_id=str(data.get("plan_id") or plan.id),
            status=str(data.get("status") or "active"),
            endpoint=self._endpoint,
        )

    async def stream_events(self, session_id: str) -> AsyncIterator[CCREvent]:
        async with self._client.stream(
            "GET",
            f"{self._endpoint}/v1/ultraplan/{session_id}/events",
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                yield CCREvent(
                    type=CCREventType(str(data["type"])),
                    session_id=str(data.get("session_id") or session_id),
                    timestamp=str(data["timestamp"]),
                    payload=dict(data.get("payload") or {}),
                )

    async def cancel_session(self, session_id: str) -> bool:
        response = await self._client.post(f"{self._endpoint}/v1/ultraplan/{session_id}/cancel")
        return response.status_code < 400

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
