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

"""Optional Bearer-token authentication for the remote API."""

from __future__ import annotations

import hmac
import os

from .errors import RemoteAPIError


def resolve_api_key(configured: str | None) -> str | None:
    """Resolve the configured API key.

    ``None`` means "read environment"; an empty string explicitly disables
    auth for tests and embedded callers.
    """

    if configured is not None:
        return configured or None
    return os.getenv("CLAWCODEX_API_KEY") or os.getenv("API_SERVER_KEY") or None


def require_bearer_auth(api_key: str | None, authorization: str | None) -> None:
    """Validate a Bearer token when auth is enabled."""

    if not api_key:
        return
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise RemoteAPIError(401, "missing bearer token", code="unauthorized")
    token = authorization[len(prefix) :]
    if not hmac.compare_digest(token, api_key):
        raise RemoteAPIError(401, "invalid bearer token", code="unauthorized")
