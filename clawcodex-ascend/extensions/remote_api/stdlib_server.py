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

"""Standard-library HTTP server for ``clawcodex api serve``."""

from __future__ import annotations

# Relative imports keep the transport bound to its extension-local implementation.
# pylint: disable=relative-beyond-top-level

import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .core import RemoteAPIConfig, RemoteAPIError, RemoteAPIService


logger = logging.getLogger(__name__)


def serve(config: RemoteAPIConfig) -> None:
    """Run the blocking HTTP server."""

    service = RemoteAPIService(config)
    handler = make_handler(service)

    httpd = ThreadingHTTPServer((config.host, config.port), handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def make_handler(service: RemoteAPIService) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a service instance."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ClawCodexRemoteAPI/0.2"

        def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            self._log_request_received("GET", path)
            try:
                if path in {"/health", "/v1/health"}:
                    self._write_json(200, service.health())
                    return
                self._require_auth()
                if path == "/health/detailed":
                    self._write_json(200, service.detailed_health())
                    return
                if path == "/v1/models":
                    self._write_json(200, service.models())
                    return
                if path == "/v1/capabilities":
                    self._write_json(200, service.capabilities())
                    return
                input_items_response_id = _response_input_items_id_from_path(path)
                if input_items_response_id:
                    self._write_json(200, service.get_response_input_items(input_items_response_id))
                    return
                response_id = _response_id_from_path(path)
                if response_id:
                    self._write_json(200, service.get_response(response_id))
                    return
            except RemoteAPIError as exc:
                self._write_error(exc)
                return
            self._write_error(RemoteAPIError(404, "not found"))

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            self._log_request_received("DELETE", path)
            try:
                self._require_auth()
                response_id = _response_id_from_path(path)
                if not response_id:
                    raise RemoteAPIError(404, "not found")
                self._write_json(200, service.delete_response(response_id))
            except RemoteAPIError as exc:
                self._write_error(exc)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            self._log_request_received("POST", path)
            try:
                self._require_auth()
                if path not in {"/v1/chat/completions", "/v1/responses"}:
                    self._write_error(RemoteAPIError(404, "not found"))
                    return
                body = self._read_json_object()
                if path == "/v1/chat/completions":
                    if body.get("stream") is True:
                        prepared = service.prepare_chat_completion(body)
                        self._write_sse_stream(
                            200,
                            service.chat_completion_sse_events(body, prepared=prepared),
                        )
                    else:
                        self._write_json(200, asyncio.run(service.chat_completion(body)))
                    return
                if path == "/v1/responses":
                    if body.get("stream") is True:
                        service.validate_responses_request(body)
                        self._write_sse_stream(200, service.responses_sse_events(body))
                    else:
                        self._write_json(200, asyncio.run(service.responses(body)))
                    return
            except RemoteAPIError as exc:
                self._write_error(exc)
                return
            except Exception:
                logger.exception("Remote API POST request failed: %s", path)
                self._write_error(RemoteAPIError(500, "internal server error"))
                return

        def _require_auth(self) -> None:
            service.require_auth(self.headers.get("Authorization"))

        def _log_request_received(self, method: str, path: str) -> None:
            remote_host = self.client_address[0] if self.client_address else "unknown"
            content_length = self.headers.get("Content-Length", "0")
            logger.info(
                "Remote API request received: %s %s from %s content_length=%s",
                method,
                path,
                remote_host,
                content_length,
            )

        def _read_json_object(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise RemoteAPIError(400, "invalid Content-Length") from exc
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise RemoteAPIError(400, "request body must be valid JSON") from exc
            if not isinstance(body, dict):
                raise RemoteAPIError(400, "request body must be a JSON object")
            return body

        def _write_error(self, exc: RemoteAPIError) -> None:
            self._write_json(exc.status_code, exc.to_payload())

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_sse(self, status_code: int, frames: list[str]) -> None:
            data = "".join(frames).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_sse_stream(self, status_code: int, frames: Any) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            async def write_frames() -> None:
                try:
                    async for frame in frames:
                        self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    logger.info("Remote API SSE client disconnected")
                finally:
                    close = getattr(frames, "aclose", None)
                    if close is not None:
                        await close()

            asyncio.run(write_frames())

    return Handler


def _response_id_from_path(path: str) -> str | None:
    prefix = "/v1/responses/"
    if not path.startswith(prefix):
        return None
    response_id = path[len(prefix) :]
    if "/" in response_id:
        return None
    return response_id or None


def _response_input_items_id_from_path(path: str) -> str | None:
    prefix = "/v1/responses/"
    suffix = "/input_items"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    response_id = path[len(prefix) : -len(suffix)]
    return response_id or None
