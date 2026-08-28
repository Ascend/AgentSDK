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

"""SOP-only read-only tool for resource catalog introspection.

Uses storage-safe views from ``catalog_cli`` (``get_stored`` / no secret
restore). Registered at SOP startup — not in global ``EXTENSION_TOOLS``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from clawcodex_ext.tool_system.build_tool import Tool, build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolResult

__all__ = ["ResourceCatalogTool", "register_resource_catalog_tool"]

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"list", "get", "latest"})
_VALID_SCOPES = frozenset({"effective", "session", "bundle", "user", "all"})


def _catalog_context(context: ToolContext) -> Any:
    from extensions.sop_converter.resource_catalog import context_from_env

    bundle = getattr(context, "bundle_context", None)
    if bundle is None:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            bundle = get_active_bundle()
        except Exception as exc:
            logger.warning("No active SOP bundle context available: %s", exc)
            bundle = None
    if bundle is None:
        raise ToolInputError("resource-catalog requires an active SOP bundle (start with --agent <bundle> / SOP mode)")
    bundle_path = getattr(bundle, "bundle_path", None)
    bundle_id = str(
        getattr(bundle, "bundle_name", "")
        or getattr(bundle, "bundle_id", "")
        or (bundle_path.name if bundle_path is not None else "")
        or ""
    )
    session_id = getattr(context, "session_id", None) or None
    return context_from_env(
        bundle_path=bundle_path,
        bundle_id=bundle_id,
        session_id=session_id,
    )


def _cmd_get_by_id(
    ctx: Any,
    *,
    scope: str,
    resource_type: str,
    resource_id: str,
    resolve_payload_flag: bool,
) -> dict[str, Any] | None:
    """Like catalog_cli._cmd_get but ``resource_type`` may be empty (match by id)."""
    from extensions.sop_converter.catalog_cli import (
        _cmd_get,
        _effective_winners,
        _iter_layer_records,
        _record_row,
    )

    if resource_type:
        return _cmd_get(
            ctx,
            scope=scope,
            resource_type=resource_type,
            resource_id=resource_id,
            resolve_payload_flag=resolve_payload_flag,
        )
    scan_scope = "all" if scope == "effective" else scope
    rows = [
        (layer, loc, rec)
        for layer, loc, rec in _iter_layer_records(ctx, scan_scope, resource_type="")
        if rec.resource_id == str(resource_id)
    ]
    if scope == "effective":
        rows = _effective_winners(rows, session_id=ctx.session_id)
    if not rows:
        return None
    if scope == "all":
        rows = sorted(
            rows,
            key=lambda t: {"session": 0, "bundle": 1, "user": 2}.get(t[0], 9),
        )
    layer, loc, rec = rows[0]
    return _record_row(
        rec,
        layer,
        catalog_dir=loc.path.parent,
        resolve_payload_flag=resolve_payload_flag,
    )


def _resource_catalog_call(
    tool_input: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    from extensions.sop_converter.catalog_cli import _cmd_latest, _cmd_list

    action = str(tool_input.get("action") or "").strip().lower()
    if action not in _VALID_ACTIONS:
        raise ToolInputError(f"action must be one of {sorted(_VALID_ACTIONS)}")

    scope = str(tool_input.get("scope") or "effective").strip().lower()
    if scope not in _VALID_SCOPES:
        raise ToolInputError(f"scope must be one of {sorted(_VALID_SCOPES)}")

    resource_type = str(tool_input.get("resource_type") or "").strip()
    resource_id = str(tool_input.get("resource_id") or "").strip()
    resolve_payload = bool(tool_input.get("resolve_payload"))

    try:
        ctx = _catalog_context(context)
    except ToolInputError as exc:
        return ToolResult(
            name="resource-catalog",
            output={"error": str(exc), "error_code": "sop_bundle_required"},
            is_error=True,
        )

    try:
        if action == "list":
            payload: Any = _cmd_list(ctx, scope=scope, resource_type=resource_type)
        elif action == "get":
            if not resource_id:
                raise ToolInputError("resource_id is required for action=get")
            payload = _cmd_get_by_id(
                ctx,
                scope=scope,
                resource_type=resource_type,
                resource_id=resource_id,
                resolve_payload_flag=resolve_payload,
            )
            if payload is None:
                return ToolResult(
                    name="resource-catalog",
                    output={
                        "error": f"resource not found: {resource_id}",
                        "error_code": "resource_catalog_missing",
                    },
                    is_error=True,
                )
        else:  # latest
            if not resource_type:
                raise ToolInputError("resource_type is required for action=latest")
            payload = _cmd_latest(ctx, scope=scope, resource_type=resource_type)
            if payload is None:
                return ToolResult(
                    name="resource-catalog",
                    output={
                        "error": f"no active resource for type={resource_type}",
                        "error_code": "resource_catalog_missing",
                    },
                    is_error=True,
                )
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc

    return ToolResult(
        name="resource-catalog",
        output=json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload,
    )


ResourceCatalogTool: Tool = build_tool(
    name="resource-catalog",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get", "latest"],
                "description": "list / get / latest catalog records (storage-redacted view)",
            },
            "scope": {
                "type": "string",
                "enum": ["effective", "session", "bundle", "user", "all"],
                "description": "Catalog layer scope (default: effective)",
            },
            "resource_type": {
                "type": "string",
                "description": "Optional type filter; required for latest",
            },
            "resource_id": {
                "type": "string",
                "description": "Resource id for get (e.g. verify-bot)",
            },
            "resolve_payload": {
                "type": "boolean",
                "description": "Inline payload_ref file contents (still redacted)",
            },
        },
        "required": ["action"],
    },
    call=_resource_catalog_call,
    description=(
        "List or get SOP resource catalog records (session/bundle/user). "
        "Storage-redacted; use for inspecting persisted agents like verify-bot. "
        "Prefer this over Grep/Read for catalog discovery."
    ),
    prompt=(
        "Inspect persisted SOP resources (agents, etc.) in the resource catalog. "
        "Use action=list to browse, action=get with resource_id (e.g. verify-bot), "
        "action=latest with resource_type. Returns redacted JSON (no restored secrets). "
        "Do not Grep the workspace for agent ids — use this tool instead."
    ),
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    max_result_size_chars=40_000,
)


def register_resource_catalog_tool(registry: Any) -> None:
    """Idempotently register :data:`ResourceCatalogTool` on a tool registry."""
    if registry is None:
        return
    name = ResourceCatalogTool.name
    existing = None
    get = getattr(registry, "get", None)
    if callable(get):
        try:
            existing = get(name)
        except Exception:
            existing = None
    if existing is not None:
        return
    registry.register(ResourceCatalogTool)
