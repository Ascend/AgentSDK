#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Module-private failure-detail extraction helpers for ``orchestrator_*`` modules.

Centralises the operator-failure detail / status-code / error-message
extraction helpers that were previously duplicated byte-for-byte across
``orchestrator``, ``orchestrator_run``, ``orchestrator_issue``,
``orchestrator_ops``, ``orchestrator_control``, ``orchestrator_session`` and
``orchestrator_rebase``.  These helpers are imported only by sibling modules
inside the ``extensions.orchestrator`` package; the leading underscore keeps
this module private to the package and out of its public import surface.
"""

from __future__ import annotations

import json
from typing import Any


def operator_failure_detail(exc: BaseException) -> str:
    """Return a concise failure detail suitable for IM and registry records."""

    raw = " ".join(str(exc).split())
    body_detail = extract_error_message_from_body(raw)
    if body_detail:
        status_code = extract_status_code(raw)
        if raw.startswith("request_failed") and status_code:
            return f"request_failed status={status_code}: {body_detail}"
        return body_detail
    return raw or exc.__class__.__name__


def extract_status_code(text: str) -> str | None:
    for part in text.split():
        if part.startswith("status="):
            status = part.removeprefix("status=").strip()
            if status:
                return status
    return None


def extract_error_message_from_body(text: str) -> str | None:
    marker = "body="
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    body = text[marker_index + len(marker) :].strip()
    if not body:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    return extract_error_message(payload)


def extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in (
            "error_message",
            "message",
            "error_description",
            "detail",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return " ".join(error.split())
        nested = extract_error_message(error)
        if nested:
            return nested
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors:
                nested = extract_error_message(item)
                if nested:
                    return nested
    return None
