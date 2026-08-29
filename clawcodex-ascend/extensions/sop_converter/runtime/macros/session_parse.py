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

# pylint: disable=cyclic-import
"""Strict parsers for session MacroDefinition / MacroRoute (Phase 5).

Unlike ``loader.parse_macro_route``, these parsers never silently coerce
invalid enums — they raise ``MacroConvertError`` instead.
"""

from __future__ import annotations

from typing import Any

from .errors import MacroConvertError
from .models import MacroDefinition, MacroRoute

_ALLOWED_DEFINITION_KEYS = frozenset(
    {
        "version",
        "name",
        "description",
        "scope",
        "enabled",
        "workflow",
        "routing",
        "provenance",
    }
)

_ALLOWED_ROUTE_KEYS = frozenset(
    {
        "phrases",
        "keywords",
        "negative_keywords",
        "target_tool",
        "match_mode",
        "selection",
        "priority",
        "verified",
        "enabled",
        "intent_key",
        "covered_tools",
        "unavailable_policy",
        "scope",
    }
)

_VALID_MATCH_MODES = frozenset({"exact", "all", "any"})
_VALID_SELECTIONS = frozenset({"exclusive", "prefer"})
_VALID_SCOPES = frozenset({"session", "bundle", "builtin"})
_VALID_UNAVAILABLE = frozenset({"restore-covered"})


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value if str(item).strip()]


def _reject_unknown(data: dict[str, Any], allowed: frozenset[str], *, field_prefix: str) -> None:
    for key in data:
        if key not in allowed:
            raise MacroConvertError(
                "macro_unknown_field",
                f"unknown field: {key}",
                field=f"{field_prefix}{key}" if field_prefix else str(key),
            )


def parse_session_macro_route(
    data: dict[str, Any] | None,
    *,
    default_target: str = "",
) -> MacroRoute:
    """Strict MacroRoute parser — invalid enums raise, never coerce."""
    if data is None:
        raw: dict[str, Any] = {}
    elif isinstance(data, dict):
        raw = data
    else:
        raise MacroConvertError(
            "macro_schema_invalid",
            "routing must be a mapping",
            field="routing",
        )
    _reject_unknown(raw, _ALLOWED_ROUTE_KEYS, field_prefix="")

    if "match_mode" in raw:
        match_mode = str(raw.get("match_mode") or "").strip()
        if match_mode not in _VALID_MATCH_MODES:
            raise MacroConvertError(
                "macro_schema_invalid",
                f"invalid routing.match_mode: {raw.get('match_mode')!r}",
                field="routing.match_mode",
            )
    else:
        match_mode = "all"

    if "selection" in raw:
        selection = str(raw.get("selection") or "").strip()
        if selection not in _VALID_SELECTIONS:
            raise MacroConvertError(
                "macro_schema_invalid",
                f"invalid routing.selection: {raw.get('selection')!r}",
                field="routing.selection",
            )
    else:
        selection = "prefer"

    if "scope" in raw:
        scope = str(raw.get("scope") or "").strip()
        if scope not in _VALID_SCOPES:
            raise MacroConvertError(
                "macro_schema_invalid",
                f"invalid routing.scope: {raw.get('scope')!r}",
                field="routing.scope",
            )
    else:
        scope = "session"

    if "unavailable_policy" in raw:
        unavailable_policy = str(raw.get("unavailable_policy") or "").strip()
        if unavailable_policy not in _VALID_UNAVAILABLE:
            raise MacroConvertError(
                "macro_schema_invalid",
                f"invalid routing.unavailable_policy: {raw.get('unavailable_policy')!r}",
                field="routing.unavailable_policy",
            )
    else:
        unavailable_policy = "restore-covered"

    if "priority" in raw:
        try:
            priority = int(raw["priority"])
        except (TypeError, ValueError) as exc:
            raise MacroConvertError(
                "macro_schema_invalid",
                "routing.priority must be an integer",
                field="routing.priority",
            ) from exc
    else:
        priority = 100

    target = str(raw.get("target_tool") or default_target or "").strip()
    return MacroRoute(
        phrases=_as_str_list(raw.get("phrases")),
        keywords=_as_str_list(raw.get("keywords")),
        negative_keywords=_as_str_list(raw.get("negative_keywords")),
        target_tool=target,
        match_mode=match_mode,  # type: ignore[arg-type]
        selection=selection,  # type: ignore[arg-type]
        priority=priority,
        verified=bool(raw.get("verified", False)),
        enabled=raw.get("enabled", True) is not False,
        intent_key=str(raw.get("intent_key") or "").strip(),
        covered_tools=_as_str_list(raw.get("covered_tools")),
        unavailable_policy=unavailable_policy,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
    )


def parse_session_macro_definition(data: dict[str, Any], *, source: str = "") -> MacroDefinition:
    """Strict MacroDefinition parser for session scope."""
    if not isinstance(data, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro definition must be a mapping",
            manifest=source,
        )
    _reject_unknown(data, _ALLOWED_DEFINITION_KEYS, field_prefix="")

    try:
        version = int(data.get("version", 1))
    except (TypeError, ValueError) as exc:
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro version must be an integer",
            manifest=source,
            field="version",
        ) from exc
    if version != 1:
        raise MacroConvertError(
            "macro_version_unsupported",
            f"unsupported macro schema version: {version}",
            manifest=source,
            field="version",
        )

    name = str(data.get("name") or "").strip()
    if not name:
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro name is required",
            manifest=source,
            field="name",
        )

    if "scope" in data:
        scope = str(data.get("scope") or "").strip()
        if scope != "session":
            raise MacroConvertError(
                "macro_scope_unsupported",
                f"session parser requires scope=session, got {scope!r}",
                manifest=source,
                field="scope",
            )
    else:
        scope = "session"

    workflow = data.get("workflow")
    if not isinstance(workflow, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "workflow block is required",
            manifest=source,
            field="workflow",
        )

    routing = parse_session_macro_route(data.get("routing"), default_target="")
    routing.scope = "session"

    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {"kind": "session", "manifest": source}
    else:
        provenance = dict(provenance)
        if source:
            provenance.setdefault("manifest", source)

    return MacroDefinition(
        version=version,
        name=name,
        description=str(data.get("description") or ""),
        scope="session",
        enabled=data.get("enabled", True) is not False,
        workflow=workflow,
        routing=routing,
        provenance=provenance,
    )
