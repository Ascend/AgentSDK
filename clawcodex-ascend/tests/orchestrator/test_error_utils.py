#!/usr/bin/env python3

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

"""Unit tests for the shared orchestrator failure-detail helpers.

Covers the helpers centralised in ``extensions.orchestrator._error_utils``
after they were deduplicated across the orchestrator mixins.
"""

from __future__ import annotations

from extensions.orchestrator._error_utils import (
    extract_error_message,
    extract_error_message_from_body,
    extract_status_code,
    operator_failure_detail,
)


def test_extract_status_code_finds_first_status_token() -> None:
    assert extract_status_code("request_failed status=429 body={...}") == "429"
    assert extract_status_code("status=401 status=500") == "401"
    assert extract_status_code("no status here") is None
    assert extract_status_code("status= empty") is None


def test_extract_error_message_from_body_decodes_json_body() -> None:
    text = 'boom body={"error_message":"rate limited","code":429}'
    assert extract_error_message_from_body(text) == "rate limited"


def test_extract_error_message_from_body_handles_missing_or_bad_body() -> None:
    assert extract_error_message_from_body("no marker at all") is None
    assert extract_error_message_from_body("request_failed body=") is None
    assert extract_error_message_from_body("body=not-json") is None


def test_extract_error_message_prefers_known_keys() -> None:
    assert extract_error_message({"error_message": "a"}) == "a"
    assert extract_error_message({"message": "b"}) == "b"
    assert extract_error_message({"error_description": "c"}) == "c"
    assert extract_error_message({"detail": "d"}) == "d"
    assert extract_error_message({"message": "   padded   "}) == "padded"


def test_extract_error_message_recurses_into_error_and_errors() -> None:
    assert extract_error_message({"error": "plain"}) == "plain"
    assert extract_error_message({"error": {"detail": "nested"}}) == "nested"
    assert extract_error_message({"errors": [{"code": 1}, {"message": "second"}]}) == "second"
    # ``error`` carrying a non-dict value does not fall through to ``errors``.
    assert extract_error_message({"error": [{"message": "from list"}]}) is None


def test_extract_error_message_non_dict_payload_is_none() -> None:
    assert extract_error_message("string") is None
    assert extract_error_message(None) is None
    assert extract_error_message([]) is None
    assert extract_error_message({}) is None


def test_operator_failure_detail_prefixes_request_failed_status() -> None:
    exc = Exception('request_failed status=429 body={"error_message":"rate limited"}')
    assert operator_failure_detail(exc) == "request_failed status=429: rate limited"


def test_operator_failure_detail_returns_body_detail_or_class_name() -> None:
    assert operator_failure_detail(Exception('oops body={"detail":"server error"}')) == "server error"
    assert operator_failure_detail(Exception("   plain   text   ")) == "plain text"
    assert operator_failure_detail(ValueError("")) == "ValueError"
