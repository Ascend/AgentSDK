#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Default adapter for :class:`PermissionContextProtocol`.

Wraps ``clawcodex_ext.permissions.types.ToolPermissionContext`` with
property aliases so the shorter Protocol field names (``is_bypass``,
``should_avoid_prompts``) map to the upstream names
(``is_bypass_permissions_mode_available``,
``should_avoid_permission_prompts``).

The upstream ``ToolPermissionContext`` already has a ``mode`` field and
a ``blocks(tool_name)`` method, so those are forwarded directly.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3.
"""

from __future__ import annotations

from typing import Any

from extensions.capabilities.permission_protocol import (
    PermissionContextProtocol,
    PermissionModeLiteral,
)

__all__ = [
    "PermissionContextAdapter",
    "default_permission_context_factory",
]


class PermissionContextAdapter(PermissionContextProtocol):
    """Wraps ``ToolPermissionContext`` to expose Protocol-compatible field names.

    The adapter stores the upstream instance and forwards most calls
    directly, with property aliases for the two fields whose names
    differ between the upstream dataclass and the Protocol.
    """

    def __init__(self, upstream: Any) -> None:
        """Wrap an upstream ``ToolPermissionContext`` instance."""
        self._upstream = upstream

    # --- direct passthrough ---

    @property
    def mode(self) -> PermissionModeLiteral:
        return self._upstream.mode  # type: ignore[no-any-return]

    # --- aliased fields ---

    @property
    def is_bypass(self) -> bool:
        return self._upstream.is_bypass_permissions_mode_available  # type: ignore[no-any-return]

    @property
    def should_avoid_prompts(self) -> bool:
        return self._upstream.should_avoid_permission_prompts  # type: ignore[no-any-return]

    # --- methods ---

    def blocks(self, tool_name: str) -> bool:
        return self._upstream.blocks(tool_name)


def default_permission_context_factory(**kwargs: Any) -> PermissionContextProtocol:
    """Construct a ``PermissionContextProtocol``-compatible instance.

    Accepts the same keyword arguments as
    ``clawcodex_ext.permissions.types.ToolPermissionContext``, plus
    ``is_bypass`` / ``should_avoid_prompts`` as aliases.

    When ``is_bypass`` is passed (and ``is_bypass_permissions_mode_available``
    is not), it is transparently mapped to the upstream field name.
    Similarly for ``should_avoid_prompts``.
    """
    from clawcodex_ext.permissions.types import ToolPermissionContext

    upstream_kw = dict(kwargs)
    # Map Protocol aliases to upstream field names
    if "is_bypass" in upstream_kw and "is_bypass_permissions_mode_available" not in upstream_kw:
        upstream_kw["is_bypass_permissions_mode_available"] = upstream_kw.pop("is_bypass")
    if "should_avoid_prompts" in upstream_kw and "should_avoid_permission_prompts" not in upstream_kw:
        upstream_kw["should_avoid_permission_prompts"] = upstream_kw.pop("should_avoid_prompts")

    upstream = ToolPermissionContext(**upstream_kw)
    return PermissionContextAdapter(upstream)
