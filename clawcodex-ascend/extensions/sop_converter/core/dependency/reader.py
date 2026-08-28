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

"""Tolerant reader for :class:`ToolDependencyGraph`.

Corruption / version mismatch / missing fields never raise - the
runtime consumer (task guide, system prompt) prefers an empty
graph over a noisy stack trace. A warning is logged instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .models import ToolDependencyGraph

logger = logging.getLogger(__name__)


def load_tool_dependencies(
    bundle_path: str | Path,
    *,
    filename: str = "tool-dependencies.yaml",
) -> ToolDependencyGraph | None:
    """Load ``tool-dependencies.yaml`` from ``bundle_path/.clawcodex/``."""
    base = Path(bundle_path)
    candidate = base / ".clawcodex" / filename
    if not candidate.exists():
        return None
    return load_graph_from_path(candidate)


def load_graph_from_path(path: str | Path) -> ToolDependencyGraph:
    """Load a graph from an explicit file path."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("tool-dependencies.yaml unreadable: %s", exc)
        return ToolDependencyGraph()
    return parse_graph_payload(text)


def parse_graph_payload(text: str) -> ToolDependencyGraph:
    """Parse the textual content of a tool-dependencies YAML file."""
    data = _safe_load(text)
    if data is None:
        logger.warning("tool-dependencies.yaml: empty or unparseable")
        return ToolDependencyGraph()
    if not isinstance(data, dict):
        logger.warning("tool-dependencies.yaml: top-level is not a mapping")
        return ToolDependencyGraph()
    return ToolDependencyGraph.from_dict(data)


def merge_overrides(
    graph: ToolDependencyGraph,
    override_path: str | Path,
) -> ToolDependencyGraph:
    """Load an override file and merge it into ``graph`` in place."""
    p = Path(override_path)
    if not p.exists():
        return graph
    override = load_graph_from_path(p)
    graph.merge_overrides(override)
    return graph


def _safe_load(text: str) -> dict | list | None:
    """Parse YAML with PyYAML when available, else a minimal subset."""
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return _minimal_yaml_load(text)
    except Exception as exc:
        logger.warning("PyYAML parse failed: %s", exc)
        return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> dict | list | None:
    """Tiny indent-based YAML loader - just enough for the writer."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))
    if not lines:
        return None
    root_indent = lines[0][0]
    return _parse_block(lines, 0, root_indent)[0]


def _parse_block(lines: list[tuple[int, str]], idx: int, parent_indent: int) -> tuple[Any, int]:
    if idx >= len(lines):
        return None, idx
    indent, content = lines[idx]
    if indent != parent_indent:
        return None, idx
    if content.startswith("- "):
        return _parse_seq(lines, idx, parent_indent)
    return _parse_map(lines, idx, parent_indent)


def _parse_map(lines: list[tuple[int, str]], idx: int, parent_indent: int) -> tuple[dict, int]:
    out: dict = {}
    while idx < len(lines):
        indent, content = lines[idx]
        if indent != parent_indent or content.startswith("- "):
            break
        if ":" not in content:
            idx += 1
            continue
        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not rest:
            nxt = idx + 1
            # A block value may be indented deeper, or be a block sequence
            # whose ``- `` items sit at the same indent as the key
            # (PyYAML ``default_flow_style=False`` output).
            if nxt < len(lines) and (
                lines[nxt][0] > parent_indent or (lines[nxt][0] == parent_indent and lines[nxt][1].startswith("- "))
            ):
                child, new_idx = _parse_block(lines, nxt, lines[nxt][0])
                out[key] = child
                idx = new_idx
            else:
                out[key] = None
                idx += 1
        else:
            out[key] = _coerce(rest)
            idx += 1
    return out, idx


def _parse_seq(lines: list[tuple[int, str]], idx: int, parent_indent: int) -> tuple[list, int]:
    out: list = []
    while idx < len(lines):
        indent, content = lines[idx]
        if indent != parent_indent or not content.startswith("- "):
            break
        rest = content[2:].strip()
        if not rest:
            nxt = idx + 1
            if nxt < len(lines) and lines[nxt][0] > parent_indent:
                child, new_idx = _parse_block(lines, nxt, lines[nxt][0])
                out.append(child)
                idx = new_idx
            else:
                out.append(None)
                idx += 1
        elif rest.startswith("- "):
            out.append(_coerce(rest[2:].strip()))
            idx += 1
        else:
            if ":" in rest and not rest.startswith("["):
                # ``- key: value`` — start of an inline mapping.  Parse the
                # first pair from ``rest``, then consume any following lines
                # indented deeper than the sequence item as the rest of the
                # mapping (PyYAML block style nests them under the item).
                inline_map, new_idx = _parse_inline_map(lines, idx, indent)
                out.append(inline_map)
                idx = new_idx
            else:
                out.append(_coerce(rest))
                idx += 1
    return out, idx


def _parse_inline_map(
    lines: list[tuple[int, str]],
    idx: int,
    parent_indent: int,
) -> tuple[dict, int]:
    """Parse ``- key: value`` plus its indented continuation lines as a map.

    The first ``key: value`` pair comes from the current sequence item;
    subsequent lines indented deeper than the item belong to the same
    mapping (PyYAML ``default_flow_style=False`` block output).  Returns
    the map and the index of the first unconsumed line.
    """
    out: dict = {}
    _, content = lines[idx]
    rest = content[2:].strip()
    key, _, val = rest.partition(":")
    key = key.strip()
    val = val.strip()
    nxt = idx + 1
    if not val:
        # Value-less first key: a nested block or block sequence may follow
        # (PyYAML ``default_flow_style=False`` output for list values).
        if nxt < len(lines) and (
            lines[nxt][0] > parent_indent or (lines[nxt][0] == parent_indent and lines[nxt][1].startswith("- "))
        ):
            child, new_idx = _parse_block(lines, nxt, lines[nxt][0])
            out[key] = child
            nxt = new_idx
        else:
            out[key] = None
            nxt += 1
    else:
        out[key] = _coerce(val)

    while nxt < len(lines) and lines[nxt][0] > parent_indent:
        c_indent, c_content = lines[nxt]
        if c_content.startswith("- "):
            break  # next sequence item belongs to the enclosing sequence
        if ":" not in c_content:
            nxt += 1
            continue
        c_key, _, c_val = c_content.partition(":")
        c_key = c_key.strip()
        c_val = c_val.strip()
        if c_val:
            out[c_key] = _coerce(c_val)
            nxt += 1
        else:
            # Value-less key: nested block or block sequence may follow
            n2 = nxt + 1
            if n2 < len(lines) and (
                lines[n2][0] > c_indent or (lines[n2][0] == c_indent and lines[n2][1].startswith("- "))
            ):
                child, new_idx2 = _parse_block(lines, n2, lines[n2][0])
                out[c_key] = child
                nxt = new_idx2
            else:
                out[c_key] = None
                nxt += 1
    return out, nxt


def _coerce(s: str) -> Any:
    if not s:
        return ""
    low = s.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


__all__ = [
    "load_tool_dependencies",
    "load_graph_from_path",
    "parse_graph_payload",
    "merge_overrides",
]
