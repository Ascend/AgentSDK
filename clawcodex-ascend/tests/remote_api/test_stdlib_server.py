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

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from extensions.remote_api.core import RemoteAPIConfig, RemoteAPIService
from extensions.remote_api.runner import (
    RemoteRunComplete,
    RemoteRunResult,
    RemoteTextDelta,
)
from extensions.remote_api.stdlib_server import make_handler
from clawcodex_ext.types.messages import AssistantMessage


LOGGER_NAME = make_handler.__module__


class _RunnerStub:
    calls = []

    def __init__(self, config, *, messages, instructions="", run_id=None):
        self.messages = list(messages)
        self.__class__.calls.append(self)

    async def run(self):
        return RemoteRunResult(
            text="stdlib answer",
            reason="success",
            usage={"input_tokens": 1, "output_tokens": 2},
            messages=[*self.messages, AssistantMessage(content="stdlib answer")],
            events=[RemoteTextDelta("stdlib answer")],
        )

    async def stream(self):
        yield RemoteTextDelta("stdlib answer")
        yield RemoteRunComplete(
            reason="success",
            response_text="stdlib answer",
            usage={"input_tokens": 1, "output_tokens": 2},
            messages=[*self.messages, AssistantMessage(content="stdlib answer")],
        )


@dataclass
class _HTTPResult:
    status: int
    headers: Any
    text: str

    def json(self) -> dict[str, Any]:
        return json.loads(self.text)


@contextmanager
def _server(tmp_path: Path, **config_kwargs) -> Iterator[str]:
    config = RemoteAPIConfig(workspace=tmp_path, **config_kwargs)
    service = RemoteAPIService(config)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> _HTTPResult:
    request_headers = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return _HTTPResult(
                status=response.status,
                headers=response.headers,
                text=response.read().decode("utf-8"),
            )
    except urllib.error.HTTPError as exc:
        return _HTTPResult(
            status=exc.code,
            headers=exc.headers,
            text=exc.read().decode("utf-8"),
        )


def test_stdlib_server_routes_auth_and_sse(tmp_path):
    auth = {"Authorization": "Bearer secret"}

    with _server(tmp_path, api_key="secret", model="service-model") as base_url:
        assert _request("GET", f"{base_url}/health").json()["status"] == "ok"
        assert _request("GET", f"{base_url}/v1/health").status == 200

        detailed_without_auth = _request("GET", f"{base_url}/health/detailed")
        assert detailed_without_auth.status == 401
        assert detailed_without_auth.json()["error"]["code"] == "unauthorized"

        models = _request("GET", f"{base_url}/v1/models", headers=auth).json()
        assert models["data"][0]["id"] == "service-model"
        capabilities = _request("GET", f"{base_url}/v1/capabilities", headers=auth).json()
        assert capabilities["features"]["responses_api"] is True

        with patch("extensions.remote_api.core.RemoteAgentRunner", _RunnerStub):
            chat_stream = _request(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=auth,
                body={
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            first_response = _request(
                "POST",
                f"{base_url}/v1/responses",
                headers=auth,
                body={"input": "hello", "conversation": "thread"},
            )

        assert chat_stream.status == 200
        assert chat_stream.headers["Content-Type"].startswith("text/event-stream")
        assert "chat.completion.chunk" in chat_stream.text
        assert "data: [DONE]" in chat_stream.text

        assert first_response.status == 200
        response_id = first_response.json()["id"]
        assert _request("GET", f"{base_url}/v1/responses/{response_id}", headers=auth).status == 200
        input_items = _request(
            "GET",
            f"{base_url}/v1/responses/{response_id}/input_items",
            headers=auth,
        )
        assert input_items.status == 200
        assert input_items.json()["data"][0]["content"][0]["text"] == "hello"
        deleted = _request("DELETE", f"{base_url}/v1/responses/{response_id}", headers=auth)
        assert deleted.status == 200
        assert deleted.json()["deleted"] is True
        assert _request("GET", f"{base_url}/v1/responses/{response_id}", headers=auth).status == 404


def test_stdlib_server_streaming_conversation_second_turn(tmp_path):
    _RunnerStub.calls = []

    with _server(tmp_path, api_key="") as base_url:
        with patch("extensions.remote_api.core.RemoteAgentRunner", _RunnerStub):
            first = _request(
                "POST",
                f"{base_url}/v1/responses",
                body={"input": "first", "conversation": "thread", "stream": True},
            )
            second = _request(
                "POST",
                f"{base_url}/v1/responses",
                body={"input": "second", "conversation": "thread", "stream": True},
            )

    assert first.status == 200
    assert second.status == 200
    assert "event: response.completed" in first.text
    assert "event: response.completed" in second.text
    assert "data: [DONE]" not in first.text
    assert "data: [DONE]" not in second.text
    assert _RunnerStub.calls[1].messages[0].content == "first"
    assert _RunnerStub.calls[1].messages[-1].content == "second"


def test_official_openai_sdk_parses_chat_and_responses(tmp_path):
    openai = pytest.importorskip("openai")

    with _server(tmp_path, api_key="sdk-key", model="clawcodex-agent") as base_url:
        client = openai.OpenAI(
            api_key="sdk-key",
            base_url=f"{base_url}/v1",
            timeout=5,
        )
        with patch("extensions.remote_api.core.RemoteAgentRunner", _RunnerStub):
            chat = client.chat.completions.create(
                model="clawcodex-agent",
                messages=[{"role": "user", "content": "hello"}],
            )
            chat_chunks = list(
                client.chat.completions.create(
                    model="clawcodex-agent",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
            response = client.responses.create(
                model="clawcodex-agent",
                input="hello",
            )
            response_events = list(
                client.responses.create(
                    model="clawcodex-agent",
                    input="hello",
                    stream=True,
                )
            )

    assert chat.choices[0].message.content == "stdlib answer"
    assert chat_chunks[-1].choices == []
    assert chat_chunks[-1].usage.total_tokens == 3
    assert response.output_text == "stdlib answer"
    assert response.usage.input_tokens_details.cached_tokens == 0
    assert response.usage.output_tokens_details.reasoning_tokens == 0
    assert response_events[0].type == "response.created"
    assert response_events[-1].type == "response.completed"
    assert response_events[-1].response.output_text == "stdlib answer"


def test_stdlib_server_logs_received_requests(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with _server(tmp_path, api_key="") as base_url:
            assert _request("GET", f"{base_url}/health").status == 200

    assert any("Remote API request received: GET /health" in record.message for record in caplog.records)


def test_stdlib_server_logs_unexpected_post_failure_without_client_detail(
    tmp_path,
    caplog,
):
    with (
        caplog.at_level(logging.ERROR, logger=LOGGER_NAME),
        _server(tmp_path, api_key="") as base_url,
        patch(
            "extensions.remote_api.core.RemoteAPIService.chat_completion",
            side_effect=RuntimeError("provider-secret-token"),
        ),
    ):
        response = _request(
            "POST",
            f"{base_url}/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status == 500
    assert response.json()["error"]["message"] == "internal server error"
    assert "provider-secret-token" in caplog.text
