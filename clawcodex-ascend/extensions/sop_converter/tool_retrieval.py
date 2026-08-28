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

# pylint: disable=undefined-loop-variable
# Backward-compatibility stub — re-exports from core/tool_retrieval.py
from extensions.sop_converter.core.tool_retrieval import (
    MacroCoverage,
    RETRIEVAL_INDEX_RELATIVE_PATH,
    RETRIEVAL_INDEX_VERSION,
    ToolRetrievalIndex,
    ToolRetrievalProfile,
    index_from_routes,
    load_tool_retrieval_index,
    normalize_tool_ref,
    resolve_tool_references,
    retrieval_index_path,
    write_tool_retrieval_index,
)

__all__ = [
    "MacroCoverage",
    "RETRIEVAL_INDEX_RELATIVE_PATH",
    "RETRIEVAL_INDEX_VERSION",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "index_from_routes",
    "load_tool_retrieval_index",
    "normalize_tool_ref",
    "resolve_tool_references",
    "retrieval_index_path",
    "write_tool_retrieval_index",
]

from extensions.sop_converter.core import tool_retrieval as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
