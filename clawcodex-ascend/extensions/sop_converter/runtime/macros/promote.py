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

"""Promote a session macro into the active bundle (Phase B)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir, save_spec
from clawcodex_ext.agent.tool_authoring.validators import ValidationError, validate_spec
from clawcodex_ext.tool_system.context import ToolContext

from .catalog import register_macro
from .convert import _to_agent_tool_spec
from .errors import MacroConvertError
from .models import MacroDefinition
from .persist import macro_definition_to_dict, macros_dir, persist_macros_atomic
from .routing import register_macro_route
from .session import (
    SessionMacroPlan,
    SessionMacroPlanStep,
    _owner_matching_snapshot,
    _raise,
)
from .validation import validate_macro_definition

logger = logging.getLogger(__name__)


def _bundle_path(context: ToolContext) -> Path:
    bundle = getattr(context, "bundle_context", None)
    if bundle is None:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            bundle = get_active_bundle()
        except ImportError:
            logger.warning("bundle_context unavailable while resolving promote bundle path")
            bundle = None
        except Exception:
            bundle = None
    path = getattr(bundle, "bundle_path", None) if bundle is not None else None
    if path is None:
        options = getattr(context, "options", None)
        path = getattr(options, "bundle_path", None) if options is not None else None
    if path is None:
        _raise("macro_promote_no_bundle", "active bundle_path is required to promote")
    return Path(path)


def _definition_to_bundle_macro(definition: MacroDefinition) -> MacroDefinition:
    data = macro_definition_to_dict(definition)
    data["scope"] = "bundle"
    provenance = dict(data.get("provenance") or {})
    provenance["promoted_from"] = "session"
    provenance["kind"] = provenance.get("kind") or "promoted"
    data["provenance"] = provenance
    routing = dict(data.get("routing") or {})
    routing["scope"] = "bundle"
    data["routing"] = routing

    from .loader import parse_macro_definition

    return parse_macro_definition(data, source=f"promote:{definition.name}")


def _build_promote_plan(
    macro: MacroDefinition,
    workflow_steps: list[Any],
    *,
    owner_session_id: str,
    expected_generation: int,
    target_path: str,
) -> SessionMacroPlan:
    steps = tuple(
        SessionMacroPlanStep(
            step_id=step.id,
            tool=step.callable_ref,
            args_template=dict(step.args),
        )
        for step in workflow_steps
    )
    routing = macro.routing
    return SessionMacroPlan(
        action="promote",
        name=macro.name,
        description=macro.description or "",
        catalog_id=f"bundle:{macro.name}",
        owner_session_id=owner_session_id,
        expected_generation=expected_generation,
        steps=steps,
        route_summary={
            "selection": routing.selection,
            "phrases": list(routing.phrases),
            "keywords": list(routing.keywords),
            "priority": routing.priority,
        },
        target_path=target_path,
    )


def promote_session_macro_to_bundle(
    context: ToolContext,
    name: str,
    *,
    replace: bool = False,
    tool_index: set[str] | None = None,
) -> dict[str, Any]:
    """Confirm + persist session macro as a bundle handwritten macro.

    Does not remove the session overlay entry.
    """
    if not getattr(context, "allow_session_macro_registration", False):
        _raise("macro_capability_denied", "session macro promote is not allowed")

    session_id = getattr(context, "session_id", None)
    if not session_id:
        _raise("macro_session_required", "session_id is required to promote")

    name_key = str(name or "").strip().lower()
    if not name_key:
        _raise("macro_schema_invalid", "name is required", error_field="name")

    snapshot = _owner_matching_snapshot(context)
    if snapshot is None:
        _raise("macro_not_found", f"session macro not found: {name}")

    definition = snapshot.definitions.get(name_key)
    if definition is None:
        _raise("macro_not_found", f"session macro not found: {name}")

    bundle_dir = _bundle_path(context)
    target = macros_dir(bundle_dir) / f"{definition.name}.yaml"
    if target.exists() and not replace:
        _raise(
            "macro_already_exists",
            f"bundle macro already exists: {target.name} (pass replace=true)",
            error_field="name",
        )

    bundle_macro = _definition_to_bundle_macro(definition)
    workflow = validate_macro_definition(
        bundle_macro,
        tool_index=tool_index,
    )
    plan = _build_promote_plan(
        bundle_macro,
        list(workflow.steps),
        owner_session_id=str(session_id),
        expected_generation=snapshot.generation,
        target_path=str(target),
    )

    confirm = getattr(context, "confirm_session_macro_plan", None)
    if confirm is None:
        _raise("macro_registration_denied", "confirm_session_macro_plan is not configured")
    try:
        approved = bool(confirm(plan))
    except Exception as exc:
        _raise(
            "macro_registration_denied",
            f"confirm_session_macro_plan failed: {exc}",
        )
    if not approved:
        _raise("macro_registration_denied", "user declined promote")

    # Re-check owner after confirm (TOCTOU).
    post = _owner_matching_snapshot(context)
    if post is None or post.owner_session_id != session_id:
        _raise("macro_stale_session", "session changed during promote confirm")
    if name_key not in post.definitions:
        _raise("macro_not_found", f"session macro disappeared: {name}")
    if post.generation != snapshot.generation:
        _raise(
            "macro_concurrent_modification",
            "session macro overlay changed during promote confirm",
        )

    spec = _to_agent_tool_spec(
        bundle_macro,
        workflow,
        bundle_dir=bundle_dir,
        register_tools=True,
    )
    try:
        validate_spec(spec)
    except ValidationError as exc:
        raise MacroConvertError(
            "macro_tool_invalid",
            f"AgentToolSpec validation failed for {bundle_macro.name}: {exc}",
            manifest=str(target),
        ) from exc

    written = persist_macros_atomic([bundle_macro], bundle_dir)
    try:
        register_macro(f"bundle:{bundle_macro.name}", workflow, replace=True)
        bundle_macro.routing.target_tool = bundle_macro.name
        bundle_macro.routing.scope = "bundle"
        register_macro_route(bundle_macro.routing, replace=True)
        save_spec(spec, tool_dir=bundle_tool_dir(bundle_dir))
    except Exception as exc:
        for path in written:
            try:
                path.unlink()
            except OSError:
                pass
        raise MacroConvertError(
            "macro_promote_failed",
            f"promote registration failed: {exc}",
            manifest=str(bundle_dir),
        ) from exc

    return {
        "promoted": True,
        "name": bundle_macro.name,
        "catalog_id": f"bundle:{bundle_macro.name}",
        "path": str(written[0]) if written else str(target),
        "session_retained": True,
    }


__all__ = ["promote_session_macro_to_bundle"]
