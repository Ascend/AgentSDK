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

"""Core tool name filter for dynamic command discovery.

The default ``ToolRegistry`` ships with a fixed set of built-in tools (Read,
Write, Bash, Edit, Glob, Grep, etc.). Per the spec, those built-in
tools MUST NOT be re-exposed as ``/tool-name`` slash commands — they have
their own dedicated code paths (e.g. ``/read`` is meaningless since
``Read`` is a model-only tool). This module exposes *non-core* tools,
which are typically:

* SDK-derived tools (e.g. ``detect_modality``, ``load_dataset``)
* Agent-created tools (persisted via
  ``clawcodex_ext/agent/tool_authoring``)
* Custom tools registered by user scripts / plugins

The filter is auto-derived from ``ALL_STATIC_TOOLS`` + the optional
``extensions/tool_system_ext/registration.py::EXTENSION_TOOLS`` list, so
adding a new built-in tool automatically extends the filter — no manual
edits required.

Why name-based rather than category-based?
-----------------------------------------
The ``Tool`` dataclass does not have a public ``is_core`` flag (only
``is_mcp`` / ``is_lsp`` / ``is_enabled`` — none of which cleanly maps to
"is a built-in"). A name-based filter is robust, deterministic, and
side-effect-free. SOP-generated tools use kebab-case names that don't
collide with the PascalCase core names.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.build_tool import Tool

log = logging.getLogger(__name__)

# The set is populated lazily on first access via :func:`is_core_tool_name`.
# This avoids a hard import of ``ALL_STATIC_TOOLS`` (and the entire tool
# system graph) at module import time — important for the stability gate
# (Stage 1 imports must stay cheap).
_CORE_TOOL_NAMES_LOWER: frozenset[str] | None = None
_CORE_TOOL_NAMES_ORIG: frozenset[str] | None = None
_EXTENSION_TOOL_NAMES_LOWER: frozenset[str] | None = None
_EXTENSION_TOOL_NAMES_ORIG: frozenset[str] | None = None


def _build_core_tool_names() -> tuple[frozenset[str], frozenset[str]]:
    """Build ``(lower, original)`` name sets for the core (static) tool list.

    Returns a tuple of (lower-cased, original-cased) frozensets. The
    lower-cased set is the hot-path index (single ``name.lower() in set``
    check inside :func:`is_core_tool_name`); the original-cased set
    preserves the public API contract exposed via
    :func:`core_tool_names_snapshot` (callers expect ``"Read"`` not
    ``"read"``).

    Errors are swallowed so that a partial install (missing
    ``extensions/tool_system_ext``) still yields a working filter.
    """
    lower: set[str] = set()
    orig: set[str] = set()
    try:
        from clawcodex_ext.tool_system.tools import ALL_STATIC_TOOLS

        for tool in ALL_STATIC_TOOLS:
            try:
                orig.add(tool.name)
                lower.add(tool.name.lower())
                for alias in getattr(tool, "aliases", ()) or ():
                    orig.add(alias)
                    lower.add(alias.lower())
            except Exception:  # noqa: BLE001 — skip malformed entries
                continue
    except Exception as exc:  # noqa: BLE001
        log.debug("could not enumerate ALL_STATIC_TOOLS: %s", exc)

    # Add the three "factory" tools that ``build_default_registry`` adds
    # via ``make_*`` calls after the static + extension lists. These are
    # part of the core surface (model can call them via the standard
    # tool-use API), so re-exposing them as slash commands would be
    # redundant and potentially dangerous (e.g. ``/agent`` would bypass
    # the conversation flow).
    orig.update({"Agent", "ToolSearch", "Workflow"})
    lower.update({"agent", "toolsearch", "workflow"})
    # ``Agent`` is also exposed under the alias ``Task`` historically;
    # keep that name filtered out as well.
    orig.add("Task")
    lower.add("task")

    return frozenset(lower), frozenset(orig)


def _build_extension_tool_names() -> tuple[frozenset[str], frozenset[str]]:
    """Build ``(lower, original)`` name sets for the extension tool list."""
    lower: set[str] = set()
    orig: set[str] = set()
    try:
        from extensions.tool_system_ext.registration import EXTENSION_TOOLS  # type: ignore[import-not-found]

        for tool in EXTENSION_TOOLS:
            try:
                orig.add(tool.name)
                lower.add(tool.name.lower())
                for alias in getattr(tool, "aliases", ()) or ():
                    orig.add(alias)
                    lower.add(alias.lower())
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001  # nosec B110
        # EXTENSION_TOOLS is optional — missing means "no extensions".
        pass
    return frozenset(lower), frozenset(orig)


def _core_names_lower() -> frozenset[str]:
    global _CORE_TOOL_NAMES_LOWER, _CORE_TOOL_NAMES_ORIG
    if _CORE_TOOL_NAMES_LOWER is None or _CORE_TOOL_NAMES_ORIG is None:
        _CORE_TOOL_NAMES_LOWER, _CORE_TOOL_NAMES_ORIG = _build_core_tool_names()
    return _CORE_TOOL_NAMES_LOWER


def _core_names_orig() -> frozenset[str]:
    global _CORE_TOOL_NAMES_LOWER, _CORE_TOOL_NAMES_ORIG
    if _CORE_TOOL_NAMES_LOWER is None or _CORE_TOOL_NAMES_ORIG is None:
        _CORE_TOOL_NAMES_LOWER, _CORE_TOOL_NAMES_ORIG = _build_core_tool_names()
    return _CORE_TOOL_NAMES_ORIG


def _extension_names_lower() -> frozenset[str]:
    global _EXTENSION_TOOL_NAMES_LOWER, _EXTENSION_TOOL_NAMES_ORIG
    if _EXTENSION_TOOL_NAMES_LOWER is None or _EXTENSION_TOOL_NAMES_ORIG is None:
        _EXTENSION_TOOL_NAMES_LOWER, _EXTENSION_TOOL_NAMES_ORIG = _build_extension_tool_names()
    return _EXTENSION_TOOL_NAMES_LOWER


def _extension_names_orig() -> frozenset[str]:
    global _EXTENSION_TOOL_NAMES_LOWER, _EXTENSION_TOOL_NAMES_ORIG
    if _EXTENSION_TOOL_NAMES_LOWER is None or _EXTENSION_TOOL_NAMES_ORIG is None:
        _EXTENSION_TOOL_NAMES_LOWER, _EXTENSION_TOOL_NAMES_ORIG = _build_extension_tool_names()
    return _EXTENSION_TOOL_NAMES_ORIG


def is_core_tool_name(name: str) -> bool:
    """Return True if *name* is a core (built-in or extension) tool.

    skip-list: any tool whose name matches a built-in / extension
    name should NOT be re-exposed as ``/tool-name``. This prevents
    collisions like ``/read`` and the more dangerous ``/bash`` (which
    would create a CLI escape hatch bypassing the LLM's permission flow).
    """
    if not name:
        return False
    return name.lower() in _core_names_lower() or name.lower() in _extension_names_lower()


def is_core_tool(tool: "Tool") -> bool:
    """Return True if *tool* is a core tool by name or any alias."""
    if is_core_tool_name(tool.name):
        return True
    for alias in getattr(tool, "aliases", ()) or ():
        if is_core_tool_name(alias):
            return True
    return False


def register_core_tool_name(name: str) -> None:
    """Mark *name* as a core tool that should be skipped.

    The module-level frozensets are replaced (not mutated — frozensets
    are immutable) so concurrent readers see a consistent view. Use this
    from extension code that registers its own built-in tools.
    """
    global _CORE_TOOL_NAMES_LOWER, _CORE_TOOL_NAMES_ORIG
    if is_core_tool_name(name):
        return
    _CORE_TOOL_NAMES_LOWER = _core_names_lower() | {name.lower()}
    _CORE_TOOL_NAMES_ORIG = _core_names_orig() | {name}


def core_tool_names_snapshot() -> frozenset[str]:
    """Return a snapshot of the current core tool names, case-preserved.

    Used by tests and by callers that want a printable list (e.g. an
    admin command that prints "core tools: Read, Write, Bash"). The
    internal hot-path index (:func:`is_core_tool_name`) uses the
    lower-cased set so lookups stay O(1).
    """
    return _core_names_orig() | _extension_names_orig()
