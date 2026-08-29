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

"""Downstream Frontend registry — plugin registration and lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawcodex_ext.frontend.protocol import FrontendPlugin


_FRONTENDS: dict[str, type[FrontendPlugin]] = {}
_INSTANCES: dict[str, FrontendPlugin] = {}


def register_frontend(cls: type[FrontendPlugin]) -> type[FrontendPlugin]:
    """Decorator to register a frontend plugin.

    Usage::

        @register_frontend
        class MyFrontend(FrontendPlugin):
            name = "myfrontend"
            display_name = "My Frontend"
            ...
    """
    _FRONTENDS[cls.name] = cls
    _INSTANCES[cls.name] = cls()  # singleton instance
    return cls


def get_frontend(name: str) -> FrontendPlugin | None:
    """Return the registered frontend instance for ``name``, or None."""
    return _INSTANCES.get(name)


def list_frontends() -> list[FrontendPlugin]:
    """Return all registered frontend instances."""
    return list(_INSTANCES.values())
