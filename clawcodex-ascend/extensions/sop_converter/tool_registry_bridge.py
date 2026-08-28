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

# Backward-compatibility stub — alias runtime/tool_registry_bridge.py
#
# The former package-root implementation imported ``.heuristics`` / ``.dependency``
# after those packages moved under ``core/``, so convert skipped all wrapper
# registration (``ImportError: cannot import name '_PRIMITIVE_TYPES'`` was the
# first failure from the star-import stub chain). Replace this module with the
# runtime implementation so CLI/tests keep working under the old import path.
from extensions.sop_converter.runtime import tool_registry_bridge as _impl
import sys

# Prefer the runtime module for subsequent ``import`` lookups.
sys.modules[__name__] = _impl
# When this file is ``importlib``-exec'd into a pre-created module object,
# that object is what callers hold onto — mirror public names onto it too.
globals().update(
    {k: v for k, v in vars(_impl).items() if k not in ("__name__", "__file__", "__package__", "__loader__", "__spec__")}
)
