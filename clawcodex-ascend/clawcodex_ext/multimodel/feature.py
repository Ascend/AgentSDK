#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Single feature-gate chokepoint for multi-model dispatch."""

from __future__ import annotations


def is_multimodel_enabled() -> bool:
    from clawcodex_ext.feature_gate import get_registry  # pylint: disable=no-name-in-module

    return get_registry().is_enabled("MULTIMODEL")


def require_multimodel_enabled() -> None:
    if not is_multimodel_enabled():
        raise RuntimeError("Multi-model mode is disabled. Enable it with `clawcodex-dev feature set MULTIMODEL --on`.")


def disabled_message() -> str:
    return "Multi-model mode is disabled. Enable it with `clawcodex-dev feature set MULTIMODEL --on`."


__all__ = ["disabled_message", "is_multimodel_enabled", "require_multimodel_enabled"]
