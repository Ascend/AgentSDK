"""Unix Domain Socket transport for Pipe IPC."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path

from .codec import PipeJsonCodec
from .models import PipeMessage, PipeMessageType

MessageHandler = Callable[[PipeMessage], Awaitable[None] | None]


class UdsPipeServer:
    def __init__(self, socket_path: Path | str, on_message: MessageHandler | None = None) -> None:
        self.socket_path = Path(socket_path)
        self.on_message = on_message
        self._server: asyncio.AbstractServer | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._peer_writers: dict[str, set[asyncio.StreamWriter]] = defaultdict(set)
        self._writer_peers: dict[asyncio.StreamWriter, str] = {}

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for writer in list(self._writers):
            await self._close_writer(writer)

        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()

    async def send(self, message: PipeMessage) -> None:
        await self._route_message(message, sender=None)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.add(writer)
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                try:
                    message = PipeJsonCodec.decode_message(raw)
                except ValueError:
                    # Drop the bad frame and keep the connection open. Raw
                    # bytes are not echoed to other peers, and the user
                    # callback is not invoked because the contract is
                    # "callback fires on a parsed message".
                    continue
                if message.type is PipeMessageType.PEER_JOIN:
                    self._peer_writers[message.source_id].add(writer)
                    self._writer_peers[writer] = message.source_id
                    writer.write(PipeJsonCodec.encode_message(message))
                    await writer.drain()
                if self.on_message is not None:
                    result = self.on_message(message)
                    if result is not None:
                        await result
                if message.type is not PipeMessageType.PEER_JOIN:
                    await self._route_message(message, sender=writer)
        finally:
            await self._close_writer(writer)

    async def _route_message(
        self,
        message: PipeMessage,
        sender: asyncio.StreamWriter | None,
    ) -> None:
        if message.target_id:
            targets = self._peer_writers.get(message.target_id, set())
        else:
            targets = self._writers

        payload = PipeJsonCodec.encode_message(message)
        stale: list[asyncio.StreamWriter] = []
        for writer in list(targets):
            if writer is sender:
                continue
            try:
                writer.write(payload)
                await writer.drain()
            except (ConnectionError, BrokenPipeError):
                stale.append(writer)

        for writer in stale:
            await self._close_writer(writer)

    async def _close_writer(self, writer: asyncio.StreamWriter) -> None:
        peer_id = self._writer_peers.pop(writer, None)
        if peer_id is not None:
            writers = self._peer_writers.get(peer_id)
            if writers is not None:
                writers.discard(writer)
                if not writers:
                    self._peer_writers.pop(peer_id, None)
        self._writers.discard(writer)
        writer.close()
        with contextlib.suppress(ConnectionError, BrokenPipeError, RuntimeError):
            await writer.wait_closed()

    @property
    def connected_count(self) -> int:
        return len(self._writers)


class UdsPipeClient:
    def __init__(
        self,
        socket_path: Path | str,
        instance_id: str,
        *,
        on_message: MessageHandler | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.instance_id = instance_id
        self.on_message = on_message
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[PipeMessage] = asyncio.Queue()
        self._join_ack: asyncio.Future[None] | None = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_unix_connection(str(self.socket_path))
        self._join_ack = asyncio.get_running_loop().create_future()
        self._reader_task = asyncio.create_task(self._read_loop())
        await self.send(
            PipeMessage(
                type=PipeMessageType.PEER_JOIN,
                source_id=self.instance_id,
                payload={"pid": os.getpid()},
            )
        )
        await asyncio.wait_for(self._join_ack, timeout=5.0)

    async def send(self, message: PipeMessage) -> None:
        if self._writer is None:
            raise RuntimeError("UdsPipeClient is not connected")
        self._writer.write(PipeJsonCodec.encode_message(message))
        await self._writer.drain()

    async def receive(self, timeout: float | None = None) -> PipeMessage:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError, RuntimeError):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def _read_loop(self) -> None:
        if self._reader is None:
            return
        while not self._reader.at_eof():
            raw = await self._reader.readline()
            if not raw:
                break
            message = PipeJsonCodec.decode_message(raw)
            if (
                message.type is PipeMessageType.PEER_JOIN
                and message.source_id == self.instance_id
                and self._join_ack is not None
                and not self._join_ack.done()
            ):
                self._join_ack.set_result(None)
                continue
            await self._queue.put(message)
            if self.on_message is not None:
                result = self.on_message(message)
                if result is not None:
                    await result
