#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Shared isolated app and session fixtures for Visualizer route tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
import pytest


def _write_json(path: Path, payload, **kwargs) -> None:
    path.write_text(json.dumps(payload, **kwargs), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


@pytest.fixture
def sessions_dir(tmp_path):
    sd = tmp_path / "sessions"
    session_id = "test-session-001"
    session_dir = sd / session_id
    session_dir.mkdir(parents=True)

    now = time.time()

    metadata = {
        "session_id": session_id,
        "title": "Test Session",
        "workspace": str(tmp_path),
        "model": "test-model",
        "provider": "test",
        "status": "completed",
        "start_time": now - 60,
        "end_time": now,
        "duration_ms": 60000,
        "turn_count": 3,
        "tool_count": 2,
    }
    _write_json(session_dir / "metadata.json", metadata, indent=2)

    transcript = [
        {"role": "user", "content": "hello", "timestamp": now - 60},
        {
            "role": "assistant",
            "content": "reading file...",
            "timestamp": now - 58,
            "tool_calls": [
                {
                    "id": "tc-001",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": {"file_path": "main.py"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc-001",
            "content": "file content",
            "timestamp": now - 57,
        },
    ]
    _write_jsonl(session_dir / "transcript.jsonl", transcript)

    return sd


def _create_minimal_session(sessions_dir: Path, session_id: str) -> Path:
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    metadata = {
        "session_id": session_id,
        "title": session_id,
        "status": "completed",
        "start_time": now - 1,
        "end_time": now,
    }
    _write_json(session_dir / "metadata.json", metadata)
    _write_jsonl(session_dir / "transcript.jsonl", [{"role": "user", "content": "hello", "timestamp": now}])
    return session_dir


@pytest.fixture
def app(sessions_dir):
    from extensions.visualizer.server import create_app

    return create_app(sessions_dir=sessions_dir, allow_import=True)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
