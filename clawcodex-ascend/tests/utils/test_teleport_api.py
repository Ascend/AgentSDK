#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for ``src.utils.teleport.api``."""

from __future__ import annotations

from src.utils.teleport.api import ANTHROPIC_VERSION, get_oauth_headers


def test_anthropic_version_matches_ts() -> None:
    assert ANTHROPIC_VERSION == "2023-06-01"


def test_get_oauth_headers_shape() -> None:
    h = get_oauth_headers("tok-xyz")
    assert h == {
        "Authorization": "Bearer tok-xyz",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }


def test_get_oauth_headers_returns_fresh_dict_each_call() -> None:
    """Callers ``.update()`` the result; must not share state."""
    a = get_oauth_headers("tok")
    b = get_oauth_headers("tok")
    a["x-test"] = "mutated"
    assert "x-test" not in b


def test_get_oauth_headers_includes_token_in_authorization() -> None:
    h = get_oauth_headers("xyzABC123")
    assert h["Authorization"] == "Bearer xyzABC123"
