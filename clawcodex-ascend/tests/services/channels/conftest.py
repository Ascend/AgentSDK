#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Pytest configuration for the channels test package.

The CI sandbox intercepts DNS and returns a private-range address for any
hostname lookup. That makes ``validate_webhook_url`` reject real-looking
public hostnames used in test fixtures, even though they would resolve
correctly in production. To keep channel construction tests deterministic
and DNS-independent, we monkeypatch ``socket.getaddrinfo`` to return a
stable public address for any hostname the channels package tries to
resolve.

Tests that want to assert specific resolution behavior can override this
fixture with their own ``socket.getaddrinfo`` patch.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

# A stable, globally-routable public address (Google DNS). Avoids the
# TEST-NET documentation ranges, which Python's ``ipaddress`` flags as
# ``is_reserved`` or ``is_private`` and would cause the validator to
# reject the URL.
_FAKE_PUBLIC_IP = "8.8.8.8"


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_getaddrinfo(
        host: str,
        port: int | str | None,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        # Mimic the structure of a real ``getaddrinfo`` result so the
        # channels code path that iterates ``info[4]`` works unchanged.
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_FAKE_PUBLIC_IP, port or 443)),
            (socket.AF_INET, socket.SOCK_DGRAM, 17, "", (_FAKE_PUBLIC_IP, port or 443)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
