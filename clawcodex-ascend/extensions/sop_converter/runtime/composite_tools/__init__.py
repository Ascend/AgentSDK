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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Composite Tools — macro-level tools that orchestrate multiple atomic operations.

A composite tool is a higher-level abstraction that bundles together multiple
atomic ``AgentToolSpec`` operations into a single registered tool with its own
workflow.yaml sidecar.

The builtin composite tools (``agent_teams``, ``pipeline_execute``, …) serve
as the macro orchestration layer that stage agents reference.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any

from ...adapters import DEFAULTS

from extensions.sop_converter.runtime.composite_tools.builtin import (
    builtin_composite_tools,
)
from extensions.sop_converter.runtime.composite_tools.models import (
    CompositeStage,
    CompositeToolSpec,
)

logger = logging.getLogger(__name__)

# Patch/test alias — register_composite_tools calls this name so unit tests can
# ``patch("extensions.sop_converter.composite_tools.save_spec")``.
save_spec = DEFAULTS.tool_authoring.save_spec


def persist_builtin_retrieval_index(bundle_dir: Path) -> Path | None:
    """Compile metadata for persisted builtin composite macros."""

    try:
        from extensions.sop_converter.runtime.macros.routing import (
            MacroRouteCatalog,
            ensure_builtin_routes,
        )
        from extensions.sop_converter.tool_retrieval import (
            index_from_routes,
            load_tool_retrieval_index,
            write_tool_retrieval_index,
        )

        tool_dir = DEFAULTS.tool_authoring.bundle_tool_dir(bundle_dir)
        specs = DEFAULTS.tool_authoring.list_persisted_specs(tool_dir=tool_dir)
        tool_names = [spec.name for spec in specs]
        catalog = MacroRouteCatalog()
        ensure_builtin_routes(catalog)
        routes = [route for route in catalog.get_routes() if route.target_tool in tool_names]
        if not routes:
            return None
        compiled = index_from_routes(
            routes,
            tool_names,
            tool_specs=specs,
            require_unique=False,
        )
        existing = load_tool_retrieval_index(bundle_dir)
        return write_tool_retrieval_index(existing.merge(compiled), bundle_dir)
    except Exception as exc:
        logger.warning("Failed to persist builtin tool retrieval index: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def to_kebab_case(name: str) -> str:
    """Convert a dotted or CamelCase name to kebab-case."""
    import re

    name = re.sub(r"([a-z])([A-Z])", r"\1-\2", name)
    name = re.sub(r"[._]+", "-", name)
    return name.lower()


_SKIP_PLACEHOLDER_COMPOSITE_TOOLS = True


def register_composite_tools(
    *,
    persist: bool = True,
    bundle_dir: Path | None = None,
    sdk_source_dir: str = "",
) -> dict[str, str]:
    """Register all built-in composite tools as ``AgentToolSpec`` specs.

    Args:
        persist: If True, save each spec to disk via ``save_spec()``.
        bundle_dir: Optional bundle directory for per-bundle tool storage.
        sdk_source_dir: Unused (reserved for future SDK introspection).

    Returns:
        A mapping from composite tool names to kebab-case registered names
        (e.g. ``{"AgentTeams": "agent-teams"}``).
    """
    tool_dir = DEFAULTS.tool_authoring.bundle_tool_dir(bundle_dir) if bundle_dir is not None else None
    name_map: dict[str, str] = {}

    for spec in builtin_composite_tools(bundle_dir=bundle_dir):
        # Placeholder composite tools (echo stage manifests) are skipped by
        # default because "echo" is not in the bash allowlist.  Executable
        # macros such as ``invoke-existing-agent`` set ``call_impl``
        # and must still be registered.
        if _SKIP_PLACEHOLDER_COMPOSITE_TOOLS and spec.call_impl is None:
            logger.info("Skipping placeholder composite tool: %s", spec.name)
            continue

        if spec.workflow_spec is not None:
            from extensions.sop_converter.runtime.macros import register_macro

            register_macro(
                f"builtin:{to_kebab_case(spec.name)}",
                spec.workflow_spec,
                replace=True,
            )

        tool_spec = _composite_to_agent_tool_spec(spec, bundle_dir=bundle_dir)
        if persist:
            try:
                save_spec(tool_spec, tool_dir=tool_dir)
            except Exception as exc:
                logger.warning("Failed to persist composite tool %s: %s", spec.name, exc)
                continue
        name_map[spec.name] = tool_spec.name

    if persist and bundle_dir is not None and name_map:
        persist_builtin_retrieval_index(bundle_dir)

    return name_map


def _composite_to_agent_tool_spec(
    spec: CompositeToolSpec,
    *,
    bundle_dir: Path | None = None,
) -> Any:
    """Convert a ``CompositeToolSpec`` into an ``AgentToolSpec``-compatible spec.

    When ``spec.call_impl`` is set, preserves ``spec.call_type`` (defaulting to
    ``bash`` only when unset) so workflow macros such as
    ``invoke-existing-agent`` register as ``call_type=\"workflow\"``.
    Placeholder specs without ``call_impl`` still emit a stage-manifest bash
    echo for backwards compatibility.
    """
    bundle_id = bundle_dir.name if bundle_dir else None

    if spec.call_impl is not None:
        return DEFAULTS.tool_authoring.create_spec(
            name=to_kebab_case(spec.name),
            description=spec.description,
            input_schema=spec.input_schema,
            call_type=spec.call_type or "bash",
            call_impl=spec.call_impl,
            tags=spec.tags,
            aliases=spec.aliases,
            source="composite-tool",
            bundle_id=bundle_id,
            output_schema=spec.output_schema,
        )

    stages_json = json.dumps(
        [
            {
                "name": s.name,
                "description": s.description,
                "agent_ref": s.agent_ref,
                "expected_duration_s": s.expected_duration_s,
            }
            for s in spec.stages
        ],
        ensure_ascii=False,
    )

    # The default call_impl is a bash one-liner that prints the stage manifest.
    # The agent reads this to understand the macro workflow. ``shlex.quote``
    # escapes the tool name and stage JSON so embedded single quotes or other
    # shell metacharacters cannot break out of the shell command.
    call_impl = f"echo Composite tool: {shlex.quote(spec.name)} && echo Stages: {shlex.quote(stages_json)}"

    return DEFAULTS.tool_authoring.create_spec(
        name=to_kebab_case(spec.name),
        description=spec.description,
        input_schema=spec.input_schema,
        call_type="bash",
        call_impl=call_impl,
        tags=spec.tags,
        aliases=spec.aliases,
        source="composite-tool",
        bundle_id=bundle_id,
        output_schema=spec.output_schema,
    )


# ---------------------------------------------------------------------------
# Workflow YAML emission
# ---------------------------------------------------------------------------


def emit_composite_workflow_yaml(
    spec: CompositeToolSpec,
    output_dir: str | Path,
    *,
    project_name: str = "",
) -> Path | None:
    """Emit a ``workflow.yaml`` sidecar for a composite tool.

    The generated YAML describes the composite tool as a linear sequence of
    stages suitable for the orchestrator engine.

    Args:
        spec: The composite tool spec.
        output_dir: Directory to write ``workflow.yaml`` into.
        project_name: Optional project name (used as workflow name prefix).

    Returns:
        The path to the written ``workflow.yaml``, or ``None`` if no stages
        are defined.
    """
    if not spec.stages:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    workflow_name = project_name or spec.name
    stages_out: list[dict[str, Any]] = []

    for i, stage in enumerate(spec.stages):
        deps: list[int] = [i] if i > 0 else []
        stage_dict: dict[str, Any] = {
            "id": i + 1,
            "name": stage.name,
            "description": stage.description,
            "agent_ref": stage.agent_ref or to_kebab_case(stage.name),
            "depends_on": deps,
            "config": {"timeout_s": stage.expected_duration_s},
        }
        stages_out.append(stage_dict)

    data: dict[str, Any] = {
        "name": workflow_name,
        "version": "1.0",
        "description": spec.description,
        "stages": stages_out,
        "config": {"workspace": "."},
    }

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available; writing workflow.yaml as JSON")
        out_path = out / "workflow.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            # Write the human-readable header inside the open file, before the
            # JSON payload, so the file handle is still valid and the header
            # precedes the body.
            f.write(f"# Composite tool: {spec.name}\n")
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    import yaml

    out_path = out / "workflow.yaml"
    header = f"# Composite tool: {spec.name}\n"
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


__all__ = [
    "CompositeToolSpec",
    "CompositeStage",
    "register_composite_tools",
    "persist_builtin_retrieval_index",
    "emit_composite_workflow_yaml",
    "to_kebab_case",
    "save_spec",
]
