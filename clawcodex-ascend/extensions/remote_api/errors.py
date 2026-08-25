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

"""HTTP-shaped errors for the remote API."""

from __future__ import annotations

from typing import Any


class RemoteAPIError(Exception):
    """HTTP-shaped API error."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code or _default_code(status_code)
        self.error_type = error_type

    def to_payload(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "error": {
                "message": self.detail,
                "type": self.error_type,
                "code": self.code,
            },
        }


def _default_code(status_code: int) -> str:
    if status_code == 401:
        return "unauthorized"
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limit_exceeded"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "internal_error"
    return "invalid_request"
