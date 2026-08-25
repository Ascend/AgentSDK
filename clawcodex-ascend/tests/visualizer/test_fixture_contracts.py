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


"""Generated and bundled sample-session fixture contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extensions.visualizer import fixtures as fixture_module
from extensions.visualizer.fixtures import create_demo_session


def test_demo_session_fixture_writes_parseable_contract(tmp_path) -> None:
    session_id = create_demo_session(tmp_path)
    session_dir = tmp_path / session_id
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    transcript = [
        json.loads(line) for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["session_id"] == session_id
    assert transcript and all(isinstance(item.get("content"), list) for item in transcript)
    assert (session_dir / "events.ndjson").is_file()


def test_demo_session_default_uses_temporary_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        fixture_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path),
    )

    session_id = create_demo_session()

    assert (tmp_path / session_id / "metadata.json").is_file()


def test_bundled_sample_session_is_valid_json() -> None:
    sample = Path(__file__).parents[2] / "extensions" / "visualizer" / "fixtures" / "sample_session.json"
    assert json.loads(sample.read_text(encoding="utf-8"))["sessionId"]
