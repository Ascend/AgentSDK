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

"""LODESTONE — ``LodestoneTool`` (agent-facing).

Single tool, five ``action``s — referenced by the agent when it wants
to linkify / inspect anchors without going through the ``/link`` slash
command.

* ``parse``   — return a structured anchor list for any text.
* ``resolve`` — pick a target + render in the requested sink.
* ``render``  — parse + render everything inline (returns the string).
* ``open``    — invoke the OS default URL handler.
* ``config``  — read / set ``LodestoneConfig`` fields.

When ``LODESTONE=off`` is set, ``parse`` and ``render`` return empty
shells; ``resolve`` returns ``{raw, fallback_reason: 'disabled'}``;
``open`` refuses to launch.  This is a fail-closed guard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolResult

from clawcodex_ext.services.lodestone import get_lodestone_service
from clawcodex_ext.services.lodestone import renderer as lodestone_renderer
from clawcodex_ext.services.lodestone.config import save_config
from clawcodex_ext.services.lodestone.parser import parse_anchors

log = logging.getLogger(__name__)

_ALLOWED_ACTIONS = frozenset({"parse", "resolve", "render", "open", "config"})
_ALLOWED_SINKS = frozenset({"text", "markdown", "osc8", "auto"})


def _tool_workspace_root(context: ToolContext) -> Path | None:
    root = getattr(context, "workspace_root", None)
    if root is None:
        cwd = getattr(context, "cwd", None)
        root = Path(cwd) if cwd else None
    return Path(root).resolve() if root else None


def _safe_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    action = tool_input.get("action")
    if not isinstance(action, str) or action not in _ALLOWED_ACTIONS:
        raise ToolInputError(f"action must be one of {sorted(_ALLOWED_ACTIONS)}")
    text = tool_input.get("text") or ""
    target_override = tool_input.get("target") or None
    sink = tool_input.get("sink") or None
    if sink is not None and sink not in _ALLOWED_SINKS:
        raise ToolInputError(f"sink must be one of {sorted(_ALLOWED_SINKS)}")

    svc = get_lodestone_service()
    workspace_root = _tool_workspace_root(context)

    if action == "parse":
        anchors = parse_anchors(text)
        payload = {
            "anchors": [
                {
                    "kind": a.kind,
                    "raw": a.raw,
                    "span": list(a.span) if a.span else None,
                    "file_path": a.file_path,
                    "line": a.line,
                    "column": a.column,
                    "symbol": a.symbol,
                    "git_sha": a.git_sha,
                    "tracker_key": list(a.tracker_key) if a.tracker_key else None,
                    "url": a.url,
                }
                for a in anchors
            ]
        }
        return ToolResult(name="LodestoneTool", output=payload)

    if action == "resolve":
        anchors = parse_anchors(text)
        if not anchors:
            return ToolResult(name="LodestoneTool", output={"results": []})
        results = []
        for a in anchors:
            rendered = svc.resolve_one(
                a,
                sink=sink,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                target_override=target_override,
            )
            results.append(
                {
                    "kind": a.kind,
                    "raw": a.raw,
                    "rendered": rendered.rendered,
                    "link_text": rendered.link_text,
                    "is_anchor": rendered.is_anchor,
                    "target": rendered.target.target_id if rendered.target else None,
                    "fallback_reason": rendered.fallback_reason,
                    "sink": rendered.sink,
                }
            )
        return ToolResult(name="LodestoneTool", output={"results": results})

    if action == "render":
        rendered_text = svc.resolve_text(
            text,
            workspace_root=workspace_root,
            sink=sink,  # type: ignore[arg-type]
            target_override=target_override,
        )
        return ToolResult(name="LodestoneTool", output={"rendered": rendered_text})

    if action == "open":
        anchors = parse_anchors(text)
        if not anchors:
            return ToolResult(name="LodestoneTool", output={"opened": [], "errors": ["no anchors"]})
        opened: list[str] = []
        errors: list[str] = []
        for a in anchors:
            rendered = svc.resolve_one(
                a,
                sink="text",
                workspace_root=workspace_root,
                target_override=target_override,
            )
            url = rendered.rendered if rendered.is_anchor else None
            if not url:
                errors.append(f"{a.raw}: {rendered.fallback_reason or 'no url'}")
                continue
            try:
                lodestone_renderer.open_uri(url)
            except lodestone_renderer.OpenLaunchError as exc:
                errors.append(f"{a.raw}: {exc}")
            else:
                opened.append(url)
        return ToolResult(name="LodestoneTool", output={"opened": opened, "errors": errors})

    if action == "config":
        changes = tool_input.get("changes") or {}
        if not isinstance(changes, dict):
            raise ToolInputError("changes must be an object")
        if not changes:
            return ToolResult(
                name="LodestoneTool",
                output={"config": json.loads(json.dumps(_dump(svc.config)))},
            )
        new_cfg = svc.update_config(**changes)
        try:
            save_config(new_cfg)
        except OSError as exc:
            log.warning("failed to persist lodestone config: %s", exc)
        return ToolResult(
            name="LodestoneTool",
            output={"config": json.loads(json.dumps(_dump(new_cfg)))},
        )

    raise ToolInputError(f"unknown action: {action}")


def _dump(cfg) -> dict[str, Any]:
    """Serialise a ``LodestoneConfig`` to a JSON-friendly dict."""
    from dataclasses import asdict

    return asdict(cfg)


LodestoneTool: Tool = build_tool(
    name="LodestoneTool",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ALLOWED_ACTIONS),
            },
            "text": {"type": "string"},
            "target": {"type": "string"},
            "sink": {
                "type": "string",
                "enum": sorted(_ALLOWED_SINKS),
            },
            "changes": {"type": "object"},
        },
        "required": ["action"],
    },
    call=_safe_call,
    prompt=(
        "Render and resolve deep-link anchors (file paths, git refs, tracker "
        "issues) into clickable URLs targeting the local IDE, git remote, or "
        "configured tracker."
    ),
    description=(
        "Use LodestoneTool to linkify ``path:line``, ``@sha:path``, ``#123`` "
        "or other anchors referenced in tool output before showing them to the "
        "user. ``action=render`` returns a Markdown-string; ``action=open`` "
        "launches the resolved URL on the user's machine."
    ),
    strict=True,
    max_result_size_chars=50_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    to_auto_classifier_input=lambda data: json.dumps(
        {
            "action": (data or {}).get("action", ""),
            "text": (data or {}).get("text", ""),
        },
        ensure_ascii=False,
    ),
)


__all__ = ["LodestoneTool"]
