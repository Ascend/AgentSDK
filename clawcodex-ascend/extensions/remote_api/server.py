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

"""FastAPI app for the Hermes-compatible remote API."""

from __future__ import annotations

# Relative imports keep the transport bound to its extension-local implementation.
# pylint: disable=relative-beyond-top-level

from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as exc:  # Optional transport; stdlib_server remains available.
    FastAPI = None  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]
    StarletteHTTPException = Exception  # type: ignore[assignment,misc]
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from .core import RemoteAPIConfig, RemoteAPIError, RemoteAPIService


def create_app(config: RemoteAPIConfig) -> Any:
    """Create the remote API app."""

    if FastAPI is None:
        raise RuntimeError(
            "FastAPI transport is unavailable; use stdlib_server or provision "
            "the optional fastapi/starlette dependencies"
        ) from _FASTAPI_IMPORT_ERROR

    app = FastAPI(title="ClawCodex Remote Agent API", version="0.2.0")
    app.state.remote_api_config = config
    app.state.remote_api_service = RemoteAPIService(config)

    _register_exception_handler(app)
    _register_health_routes(app)
    _register_action_routes(app)
    _register_response_routes(app)
    return app


def _register_exception_handler(app: Any) -> None:
    """Register the transport-wide HTTP exception projection."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_response(RemoteAPIError(exc.status_code, detail))


def _register_health_routes(app: Any) -> None:
    """Register health and capability discovery routes."""

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return app.state.remote_api_service.health()

    @app.get("/v1/health")
    async def v1_health() -> dict[str, Any]:
        return app.state.remote_api_service.health()

    @app.get("/health/detailed")
    async def health_detailed(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.detailed_health()
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.models()
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.get("/v1/capabilities")
    async def capabilities(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.capabilities()
        except RemoteAPIError as exc:
            return _error_response(exc)


def _register_action_routes(app: Any) -> None:
    """Register mutation and completion routes."""

    @app.post("/proactive/focus")
    async def proactive_focus(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            body = await _read_json_object(request)
            level = body.get("level")
            if not isinstance(level, str):
                raise RemoteAPIError(400, "level must be a string")
            return {"automation_state": _set_proactive_focus(level)}
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            body = await _read_json_object(request)
            if body.get("stream") is True:
                prepared = app.state.remote_api_service.prepare_chat_completion(body)
                return _sse_response(
                    app.state.remote_api_service.chat_completion_sse_events(
                        body,
                        prepared=prepared,
                    ),
                )
            return await app.state.remote_api_service.chat_completion(body)
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.post("/v1/responses")
    async def responses(request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            body = await _read_json_object(request)
            if body.get("stream") is True:
                app.state.remote_api_service.validate_responses_request(body)
                return _sse_response(
                    app.state.remote_api_service.responses_sse_events(body),
                )
            return await app.state.remote_api_service.responses(body)
        except RemoteAPIError as exc:
            return _error_response(exc)


def _register_response_routes(app: Any) -> None:
    """Register persisted Responses API lookup routes."""

    @app.get("/v1/responses/{response_id}/input_items")
    async def get_response_input_items(response_id: str, request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.get_response_input_items(response_id)
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.get("/v1/responses/{response_id}")
    async def get_response(response_id: str, request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.get_response(response_id)
        except RemoteAPIError as exc:
            return _error_response(exc)

    @app.delete("/v1/responses/{response_id}")
    async def delete_response(response_id: str, request: Request) -> Any:
        try:
            _require_auth(app.state.remote_api_service, request)
            return app.state.remote_api_service.delete_response(response_id)
        except RemoteAPIError as exc:
            return _error_response(exc)


def _require_auth(service: RemoteAPIService, request: Request) -> None:
    service.require_auth(request.headers.get("authorization"))


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise RemoteAPIError(400, "request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise RemoteAPIError(400, "request body must be a JSON object")
    return body


def _set_proactive_focus(level: str) -> str:
    try:
        from .state_reporter import set_proactive_focus
    except ImportError as exc:
        raise RemoteAPIError(500, "proactive focus is unavailable") from exc
    try:
        return set_proactive_focus(level)
    except ValueError as exc:
        raise RemoteAPIError(400, str(exc)) from exc


def _error_response(exc: RemoteAPIError) -> Any:
    if JSONResponse is None:  # Defensive: create_app normally rejects first.
        raise RuntimeError("FastAPI transport is unavailable") from _FASTAPI_IMPORT_ERROR
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


def _sse_response(iterator: Any) -> Any:
    if StreamingResponse is None:  # Defensive: create_app normally rejects first.
        raise RuntimeError("FastAPI transport is unavailable") from _FASTAPI_IMPORT_ERROR
    return StreamingResponse(
        iterator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
