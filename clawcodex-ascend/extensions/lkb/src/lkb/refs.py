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

"""Node references for the LKB Plan Graph.

A ``NodeRef`` is the canonical identity of any node inside a plan graph.
It triples ``(graph, kind, id)`` into a single comparable / hashable /
serializable value with strong validation so that identifiers can never
be used for path traversal or other injection attacks.

Spec §5.3 — NodeRef identity and canonical form.
Spec §9.5 — task_id → subject_ref:NodeRef migration mapping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ── validation patterns ──────────────────────────────────────────────

# graph / kind must match [A-Za-z0-9_-]+
_GRAPH_KIND_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# id allows path separators and dots for backward-compatible identifiers,
# e.g. "src/auth.py" or "pkg/sub/module.py"
# Forbidden by other checks: leading /, ".." components, "\", control chars
_ID_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
DEFAULT_PLAN_GRAPH_ID = "plan"


def _has_control_chars(value: str) -> bool:
    """Return True if *value* contains any Unicode control character."""
    for ch in value:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            return True
    return False


def _validate_component(value: str, field: str, *, allow_path: bool = False) -> None:
    """Shared validation for graph/kind/id components.

    * ``field`` is used in error messages (e.g. ``"graph"``).
    * ``allow_path`` allows ``/`` and ``.`` in backward-compatible or
      versioned ids, but still blocks path-traversal patterns.
    """
    if not isinstance(value, str):
        raise ValueError(f"NodeRef.{field} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"NodeRef.{field} must not be empty")
    if _has_control_chars(value):
        raise ValueError(f"NodeRef.{field} must not contain control characters")
    # backslash is never allowed
    if "\\" in value:
        raise ValueError(f"NodeRef.{field} must not contain '\\\\' (would enable path traversal)")
    # colon is the field separator — never allowed inside a component
    if ":" in value:
        raise ValueError(f"NodeRef.{field} must not contain ':' (reserved as field separator)")

    if allow_path:
        if not _ID_RE.match(value):
            raise ValueError(f"NodeRef.{field} must match [A-Za-z0-9_./-]+; got {value!r}")
        # path-traversal checks (split on '/')
        parts = value.split("/")
        # leading slash would produce an empty first part
        if parts[0] == "":
            raise ValueError(f"NodeRef.{field} must not start with '/' (absolute paths not allowed)")
        # trailing slash would produce an empty last part
        if parts[-1] == "":
            raise ValueError(f"NodeRef.{field} must not end with '/' (directory-style not allowed)")
        # ".." or "." as a component is dangerous
        for part in parts:
            if part == "..":
                raise ValueError(f"NodeRef.{field} must not contain '..' path component (would escape storage path)")
            if part == ".":
                raise ValueError(f"NodeRef.{field} must not contain '.' path component")
    else:
        if not _GRAPH_KIND_RE.match(value):
            raise ValueError(f"NodeRef.{field} must match [A-Za-z0-9_-]+; got {value!r}")


@dataclass(frozen=True)
class NodeRef:
    """Canonical reference to a node inside a plan graph.

    The triple ``(graph, kind, id)`` uniquely identifies a node across
    every board / graph in the system.  Graph and kind are open strings
    (not a closed Literal set) — new kinds can be introduced without
    modifying this module.

    Canonical string form: ``"graph:kind:id"``.

    Examples
    --------
    >>> NodeRef("plan", "task", "T-001").to_str()
    'plan:task:T-001'
    >>> NodeRef.from_str("plan:task:T-001").id
    'T-001'
    """

    graph: str
    kind: str
    id: str

    def __post_init__(self) -> None:
        _validate_component(self.graph, "graph", allow_path=False)
        _validate_component(self.kind, "kind", allow_path=False)
        _validate_component(self.id, "id", allow_path=True)

    # ── serialization ─────────────────────────────────────────────────

    def to_str(self) -> str:
        """Return the canonical ``"graph:kind:id"`` string form."""
        return f"{self.graph}:{self.kind}:{self.id}"

    def __str__(self) -> str:  # noqa: D401  (frozen dataclass, intuitive)
        return self.to_str()

    @classmethod
    def from_str(cls, s: str) -> "NodeRef":
        """Parse a canonical ``"graph:kind:id"`` string into a NodeRef.

        Uses the first two colons as delimiters; everything after the
        second colon is the id.  Raises ``ValueError`` if the string is
        malformed or any component fails validation.
        """
        if not isinstance(s, str):
            raise ValueError(f"NodeRef.from_str expects a string, got {type(s).__name__}")
        parts = s.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"NodeRef.from_str expects 'graph:kind:id' with at least 2 colons; got {s!r}")
        graph, kind, nid = parts
        return cls(graph=graph, kind=kind, id=nid)

    # ── helpers ───────────────────────────────────────────────────────

    @property
    def task_id(self) -> str | None:
        """Backward-compat shim: return ``self.id`` when ``kind == 'task'``.

        Existing callers that speak in terms of ``task_id`` can read
        ``ref.task_id`` without switching to the triple form immediately.
        Returns ``None`` for non-task kinds so callers get a clear signal
        when the ref is not a task node.
        """
        if self.kind == "task":
            return self.id
        return None


def plan_task_ref(task_id: str, *, graph_id: str) -> NodeRef:
    """Resolve a legacy task id through an explicit plan graph identity."""
    return NodeRef(graph=graph_id, kind="task", id=task_id)


__all__ = [
    "DEFAULT_PLAN_GRAPH_ID",
    "NodeRef",
    "plan_task_ref",
]
