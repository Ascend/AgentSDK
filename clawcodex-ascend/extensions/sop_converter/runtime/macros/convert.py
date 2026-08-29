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

"""Phase 4: convert handwritten macros into bundle tools + routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ...adapters import DEFAULTS

from ..composite_runtime import CompositeWorkflowSpec
from .catalog import DEFAULT_MACRO_CATALOG, register_macro
from .errors import MacroConvertError
from .loader import discover_macro_sources, load_macro_yaml
from .models import MacroDefinition
from .persist import macro_relative_manifest, persist_macros_atomic
from .routing import DEFAULT_MACRO_ROUTE_CATALOG, register_macro_route
from .validation import validate_macro_definition
from ...core.tool_retrieval import (
    index_from_routes,
    load_tool_retrieval_index,
    write_tool_retrieval_index,
)


@dataclass
class MacroConvertResult:
    macros: list[MacroDefinition] = field(default_factory=list)
    registered_tools: dict[str, str] = field(default_factory=dict)
    written_paths: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, str]] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)


def _workflow_input_schema(spec: CompositeWorkflowSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, schema in spec.inputs.items():
        prop = dict(schema)
        prop.setdefault("type", "string")
        is_required = bool(prop.pop("required", False))
        properties[name] = prop
        if is_required:
            required.append(name)
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


def _to_agent_tool_spec(
    macro: MacroDefinition,
    workflow: CompositeWorkflowSpec,
    *,
    bundle_dir: Path | None,
    register_tools: bool,
) -> Any:
    name = macro.routing.target_tool or macro.name
    if register_tools and bundle_dir is not None:
        call_impl: dict[str, Any] = {
            "catalog_id": f"bundle:{name}",
        }
    else:
        call_impl = {"manifest": macro_relative_manifest(name)}
    aliases = []
    routing = macro.routing
    # aliases may appear in YAML routing via provenance/extra — keep empty unless present on route
    extra_aliases = getattr(routing, "aliases", None)
    if isinstance(extra_aliases, list):
        aliases = [str(a) for a in extra_aliases if str(a).strip()]
    return DEFAULTS.tool_authoring.create_spec(
        name=name,
        description=macro.description or workflow.description or name,
        input_schema=_workflow_input_schema(workflow),
        call_type="workflow",
        call_impl=call_impl,
        tags=["macro", "workflow", "sop-converter"],
        aliases=tuple(aliases),
        source="sop-converter-macro",
        bundle_id=bundle_dir.name if bundle_dir is not None else None,
    )


def convert_handwritten_macros(
    *,
    source_dir: Path | None = None,
    bundle_dir: Path | None = None,
    manifest_paths: list[Path] | None = None,
    tool_index: Iterable[str] | None = None,
    preview: bool = False,
    validate_only: bool = False,
    register_tools: bool = True,
    persist: bool = True,
) -> MacroConvertResult:
    """Load, validate, optionally persist and register handwritten macros.

    Modes (§6.4):
    - preview / validate_only: no file writes, no tool registration
    - register_tools=False: may persist normalized manifests, no active tool/route
    - normal: atomic persist + workflow AgentToolSpec + MacroRoute + MacroCatalog
    """
    result = MacroConvertResult()
    paths = discover_macro_sources(source_dir=source_dir, manifest_paths=manifest_paths)
    if not paths:
        return result

    index = {str(name) for name in (tool_index or [])}
    validated: list[tuple[MacroDefinition, CompositeWorkflowSpec]] = []

    for path in paths:
        try:
            macro = load_macro_yaml(path)
            workflow = validate_macro_definition(macro, tool_index=index)
            validated.append((macro, workflow))
            result.preview.append(
                {
                    "name": macro.name,
                    "manifest": str(path),
                    "steps": [step.id for step in workflow.steps],
                    "selection": macro.routing.selection,
                    "target_tool": macro.routing.target_tool or macro.name,
                }
            )
        except MacroConvertError as exc:
            result.diagnostics.append(exc.to_dict())
            if not preview:
                raise

    result.macros = [macro for macro, _ in validated]
    if preview or validate_only:
        return result

    prepared_specs: list[Any] = []
    if register_tools:
        if bundle_dir is None:
            raise MacroConvertError(
                "macro_bundle_required",
                "bundle --out directory is required to register macro tools",
            )
        for macro, workflow in validated:
            spec = _to_agent_tool_spec(
                macro,
                workflow,
                bundle_dir=bundle_dir,
                register_tools=True,
            )
            try:
                DEFAULTS.tool_authoring.validate_spec(spec)
            except Exception as exc:
                name = macro.routing.target_tool or macro.name
                raise MacroConvertError(
                    "macro_tool_invalid",
                    f"AgentToolSpec validation failed for {name}: {exc}",
                    manifest=str(macro.provenance.get("manifest") or name),
                ) from exc
            prepared_specs.append(spec)

    for macro, _workflow in validated:
        name = macro.routing.target_tool or macro.name
        macro.routing.target_tool = name
        macro.routing.scope = "bundle"

    if persist:
        if bundle_dir is None:
            raise MacroConvertError(
                "macro_bundle_required",
                "bundle --out directory is required to persist macros",
            )
        written = persist_macros_atomic([macro for macro, _ in validated], bundle_dir)
        result.written_paths = [str(path) for path in written]
        try:
            available_names = set(index)
            available_names.update(macro.routing.target_tool or macro.name for macro, _ in validated)
            existing_specs = DEFAULTS.tool_authoring.list_persisted_specs(
                tool_dir=DEFAULTS.tool_authoring.bundle_tool_dir(bundle_dir)
            )
            retrieval_index = index_from_routes(
                [macro.routing for macro, _ in validated],
                available_names,
                tool_specs=[*existing_specs, *prepared_specs],
                require_unique=True,
            )
            existing_retrieval = load_tool_retrieval_index(bundle_dir)
            retrieval_path = write_tool_retrieval_index(
                existing_retrieval.merge(retrieval_index),
                bundle_dir,
            )
            result.written_paths.append(str(retrieval_path))
        except Exception as exc:
            for path in written:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise MacroConvertError(
                "tool_retrieval_index_write_failed",
                f"failed to compile tool retrieval index: {exc}",
                manifest=str(bundle_dir),
            ) from exc

    if not register_tools:
        return result

    tool_dir = DEFAULTS.tool_authoring.bundle_tool_dir(bundle_dir)
    registered_ids: list[str] = []
    registered_tools: list[str] = []
    saved_spec_paths: list[Path] = []
    try:
        for (macro, workflow), spec in zip(validated, prepared_specs):
            name = macro.routing.target_tool or macro.name
            catalog_id = f"bundle:{name}"
            register_macro(catalog_id, workflow, replace=True)
            registered_ids.append(catalog_id)
            register_macro_route(macro.routing, replace=True)
            registered_tools.append(name)
            DEFAULTS.tool_authoring.save_spec(spec, tool_dir=tool_dir)
            saved_spec_paths.append(Path(tool_dir) / f"{spec.name}.json")
            result.registered_tools[macro.name] = name
    except Exception as exc:
        for catalog_id in registered_ids:
            DEFAULT_MACRO_CATALOG.unregister(catalog_id)
        for name in registered_tools:
            DEFAULT_MACRO_ROUTE_CATALOG.unregister_route(name)
        for path in saved_spec_paths:
            try:
                path.unlink()
            except OSError:
                pass
        raise MacroConvertError(
            "macro_register_failed",
            f"macro tool registration failed: {exc}",
            manifest=str(bundle_dir),
        ) from exc

    return result
