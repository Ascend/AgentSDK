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

"""Regression tests for strict persisted-IR canonical hashing."""

from __future__ import annotations

import math

import pytest

from lkb.ir_hash import canonical_json


def test_canonical_json_rejects_non_json_object_instead_of_stringifying_it() -> None:
    with pytest.raises(TypeError):
        canonical_json({"value": object()})


def test_canonical_json_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": math.nan})
