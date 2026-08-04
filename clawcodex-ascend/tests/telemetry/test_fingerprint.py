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

"""Tests for compute_fingerprint."""

from __future__ import annotations

import re

from telemetry.fingerprint import compute_fingerprint


_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _raise_with_message(msg: str):
    def _f():
        raise RuntimeError(msg)

    return _f


def test_fingerprint_is_hex_16_chars():
    try:
        _raise_with_message("hello world")()
    except RuntimeError as exc:
        fp = compute_fingerprint(exc)
    assert isinstance(fp, str)
    assert len(fp) == 16
    assert _HEX16.match(fp) is not None


def test_fingerprint_is_stable_for_same_exception_shape():
    a, b = None, None
    try:
        _raise_with_message("connection refused to 10.0.0.1:443")()
    except RuntimeError as exc:
        a = compute_fingerprint(exc)
    try:
        _raise_with_message("connection refused to 192.168.1.1:443")()
    except RuntimeError as exc:
        b = compute_fingerprint(exc)
    # IP addresses are stripped; same class + same line + same message
    # shape → same fingerprint.
    assert a == b


def test_fingerprint_differs_for_different_exception_class():
    a, b = None, None
    try:
        raise ValueError("oops")
    except ValueError as exc:
        a = compute_fingerprint(exc)
    try:
        raise KeyError("oops")
    except KeyError as exc:
        b = compute_fingerprint(exc)
    assert a != b


def test_fingerprint_strips_uuids():
    a, b = None, None

    def _raise(message):
        raise RuntimeError(message)

    try:
        _raise("session 11111111-2222-3333-4444-555555555555 failed")
    except RuntimeError as exc:
        a = compute_fingerprint(exc)
    try:
        _raise("session 99999999-8888-7777-6666-555555555555 failed")
    except RuntimeError as exc:
        b = compute_fingerprint(exc)
    # Both raises share the same source line in ``_raise``; only the
    # UUID token differs. Stripping the UUID should make the hash equal.
    assert a == b


def test_fingerprint_does_not_crash_on_no_traceback():
    class _Bare:
        pass

    # Construct an exception object that has no __traceback__ set.
    exc = RuntimeError("no frame")
    fp = compute_fingerprint(exc)
    assert isinstance(fp, str) and len(fp) == 16
