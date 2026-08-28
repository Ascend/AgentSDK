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


"""Stage capability mapping."""

from .models import (
    Capability,
    CapabilityKind,
    ExecutionMode,
    StageAgentMap,
    StageCapabilityProfile,
)
from .mapper import StageCapabilityMapper, ensure_stage_skills


def ensure_arc_stage_skills(*args, **kwargs):
    """Lazy wrapper so ARC support is not imported until first use."""
    from .arc_mapper import ensure_arc_stage_skills as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "Capability",
    "CapabilityKind",
    "ExecutionMode",
    "StageAgentMap",
    "StageCapabilityProfile",
    "StageCapabilityMapper",
    "ensure_stage_skills",
    "ensure_arc_stage_skills",
]
