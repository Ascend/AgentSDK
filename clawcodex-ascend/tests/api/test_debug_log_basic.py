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

"""Smoke test for extensions.api.debug_log — verifies NDJSON append."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from extensions.api.debug_log import append_debug_event


def test_append_debug_event_writes_ndjson() -> None:
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "debug.ndjson"
        append_debug_event(log_path, "test_stage", key="value")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["stage"] == "test_stage"
        assert row["key"] == "value"


def test_append_debug_event_none_path_noop() -> None:
    append_debug_event(None, "test_stage", key="value")
