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
# tech_v26.2.0 has not merged the package marker files (e.g.
# extensions/__init__.py) yet, so pylint cannot tell that sop_converter is a
# Python package and flags valid relative imports as E0402. Drop this tag once
# the package markers land.


# pylint: disable=undefined-loop-variable
# Backward-compatibility stub — re-exports from runtime/task_guide.py
from extensions.sop_converter.runtime.task_guide import (
    build_operation_index,
    is_entry_point,
    generate_task_guide_markdown,
    append_task_guide_to_skill_body,
    format_flat_skill_markdown,
)

__all__ = [
    "build_operation_index",
    "is_entry_point",
    "generate_task_guide_markdown",
    "append_task_guide_to_skill_body",
    "format_flat_skill_markdown",
]

from extensions.sop_converter.runtime import task_guide as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
