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

"""Stable canonical hashing for Plan Graph JSON values."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return strict, stable JSON for a value accepted by the wire format.

    Values must already be JSON data.  Coercing arbitrary objects with
    ``default=str`` would make persisted hashes depend on implementation
    details, while non-finite floats are not portable JSON.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any, *, algorithm: str = "sha256") -> str:
    """Return ``algorithm:hexdigest`` for the canonical JSON of ``value``."""
    payload = canonical_json(value).encode("utf-8")
    if algorithm == "sha256":
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if algorithm == "sha512":
        return f"sha512:{hashlib.sha512(payload).hexdigest()}"
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


__all__ = [
    "canonical_hash",
    "canonical_json",
]
