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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Validator dict helpers aligned with ContractValidator types."""

from __future__ import annotations

from ..extractors.models import StageContract

VALIDATOR_TYPES = frozenset(
    {
        "file_exists",
        "file_size",
        "regex",
        "line_count",
        "json_schema",
        "custom",
    }
)


def contract_to_validators(contract: StageContract) -> list[dict]:
    """Map StageContract.output_files to file_exists validators."""
    return [{"type": "file_exists", "path": pattern} for pattern in contract.output_files]


def file_exists_validator(path: str) -> dict:
    return {"type": "file_exists", "path": path}
