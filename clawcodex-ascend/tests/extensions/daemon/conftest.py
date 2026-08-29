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
#

"""Shared fixtures for daemon tests (isolated tmp-path state directories)."""

from __future__ import annotations

import pytest


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "daemon"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def worker_kind() -> str:
    return "test-echo"


@pytest.fixture
def short_backoff():
    from extensions.daemon import constants

    original = constants.BACKOFF_INITIAL_MS
    constants.BACKOFF_INITIAL_MS = 50
    constants.BACKOFF_CAP_MS = 200
    constants.RAPID_FAILURE_WINDOW_MS = 500
    constants.MAX_RAPID_FAILURES = 3
    try:
        yield
    finally:
        constants.BACKOFF_INITIAL_MS = original
        constants.BACKOFF_CAP_MS = 120_000
        constants.RAPID_FAILURE_WINDOW_MS = 10_000
        constants.MAX_RAPID_FAILURES = 5
