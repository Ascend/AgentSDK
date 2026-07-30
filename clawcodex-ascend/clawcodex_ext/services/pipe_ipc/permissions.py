"""Permission forwarding primitives for Pipe IPC."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .models import PipeMessage, PipeMessageType

PipeSend = Callable[[PipeMessage], Awaitable[None]]


class PipePermissionForwarder:
    def __init__(self, source_id: str, send: PipeSend) -> None:
        self._source_id = source_id
        self._send = send
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def request_permission(
        self,
        target_id: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> bool:
        request_id = str(payload.get("request_id") or uuid.uuid4())
        request_payload = dict(payload)
        request_payload["request_id"] = request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[request_id] = future

        try:
            await self._send(
                PipeMessage(
                    type=PipeMessageType.PERMISSION_REQ,
                    source_id=self._source_id,
                    target_id=target_id,
                    payload=request_payload,
                )
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return False
        finally:
            self._pending.pop(request_id, None)

    def handle_permission_response(self, message: PipeMessage) -> bool:
        if message.type not in {PipeMessageType.PERMISSION_GRANT, PipeMessageType.PERMISSION_DENY}:
            return False

        request_id = message.payload.get("request_id") or message.id
        future = self._pending.get(str(request_id))
        if future is None or future.done():
            return False

        future.set_result(message.type is PipeMessageType.PERMISSION_GRANT)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._pending)
