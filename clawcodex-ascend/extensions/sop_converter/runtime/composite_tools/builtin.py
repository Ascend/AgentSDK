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


"""Built-in composite tool specs.

Composite tools are macro-level operations that orchestrate multiple atomic
tools into a workflow.  They are registered alongside regular per-component
tools and can emit ``workflow.yaml`` sidecars for the orchestrator engine.

Available builtins
------------------
* ``agent_teams`` — Delegate a task to a sub-agent team.
* ``pipeline_execute`` — Execute a multi-stage pipeline.
* ``code_review`` — Run code review and auto-fix on a PR branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import CompositeStage, CompositeToolSpec
from ..composite_workflows import (
    invoke_existing_agent_workflow,
    resume_resource_workflow,
)


def builtin_composite_tools(*, bundle_dir: Path | None = None) -> list[CompositeToolSpec]:
    """Return all built-in composite tool specifications.

    These are the macro-level tools that stage agents and overview agents
    reference for cross-cutting orchestration patterns.
    """
    return [
        _agent_teams(),
        _pipeline_execute(),
        _code_review(),
        _invoke_existing_agent(bundle_dir=bundle_dir),
        _resume_resource(bundle_dir=bundle_dir),
    ]


# ---------------------------------------------------------------------------
# Individual specs
# ---------------------------------------------------------------------------


def _agent_teams() -> CompositeToolSpec:
    """Delegate a task to a sub-agent team with automatic routing."""
    return CompositeToolSpec(
        name="agent_teams",
        description=(
            "Delegate a sub-task to an agent team.  The orchestrator routes "
            "the request to the most capable sub-agent based on the task "
            "description and available skill set."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed task description for the sub-agent team.",
                },
                "target_skill": {
                    "type": "string",
                    "description": "Optional skill name to route the task to a specific agent.",
                },
                "delegation_mode": {
                    "type": "string",
                    "enum": ["auto", "explicit", "broadcast"],
                    "description": "How to delegate: auto=route by skill, explicit=use target_skill, broadcast=all agents.",
                },
            },
            "required": ["task"],
        },
        stages=[
            CompositeStage(
                name="task-analysis",
                description="Analyze the incoming task and determine the best sub-agent route.",
                agent_ref="agent_teams-analyze",
                expected_duration_s=15,
            ),
            CompositeStage(
                name="delegation",
                description="Delegate the task to the selected sub-agent.",
                agent_ref="agent_teams-delegate",
                expected_duration_s=60,
            ),
            CompositeStage(
                name="result-collection",
                description="Collect and consolidate results from sub-agents.",
                agent_ref="agent_teams-collect",
                expected_duration_s=30,
            ),
        ],
        tags=("composite", "macro", "teams", "delegation"),
        aliases=("agent-teams", "subagent-delegate"),
    )


def _pipeline_execute() -> CompositeToolSpec:
    """Execute a multi-stage pipeline with dependency ordering."""
    return CompositeToolSpec(
        name="pipeline_execute",
        description=(
            "Execute a multi-stage pipeline where each stage depends on one "
            "or more previous stages.  Stages are run in dependency order "
            "with automatic gate checks between transitions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pipeline_name": {
                    "type": "string",
                    "description": "Name of the pipeline definition to execute.",
                },
                "run_dir": {
                    "type": "string",
                    "description": "Absolute path to the pipeline run directory.",
                },
                "params": {
                    "type": "object",
                    "description": "Optional parameter overrides passed to each stage.",
                },
            },
            "required": ["pipeline_name", "run_dir"],
        },
        stages=[
            CompositeStage(
                name="pipeline-init",
                description="Initialize the pipeline run directory and validate configuration.",
                agent_ref="pipeline-init",
                expected_duration_s=10,
            ),
            CompositeStage(
                name="stage-execution",
                description="Execute pipeline stages sequentially, respecting dependency order.",
                agent_ref="pipeline-execute-stage",
                expected_duration_s=300,
            ),
            CompositeStage(
                name="post-processing",
                description="Collect artifacts, generate reports, and clean up.",
                agent_ref="pipeline-postprocess",
                expected_duration_s=30,
            ),
        ],
        tags=("composite", "macro", "pipeline"),
        aliases=("pipeline", "run-pipeline"),
    )


def _code_review() -> CompositeToolSpec:
    """Run a code review workflow with auto-fix and verification."""
    return CompositeToolSpec(
        name="code_review",
        description=(
            "Run an automated code review on the current branch, apply "
            "auto-fixable changes, run verification tests, and generate "
            "a review report.  Designed for PR review automation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Git branch to review.",
                },
                "auto_fix": {
                    "type": "boolean",
                    "description": "Apply auto-fixable changes (lint, format).",
                },
                "run_tests": {
                    "type": "boolean",
                    "description": "Run verification tests after auto-fix.",
                },
            },
            "required": ["branch"],
        },
        stages=[
            CompositeStage(
                name="diff-analysis",
                description="Analyze the diff and categorize changes.",
                agent_ref="code-review-diff",
                expected_duration_s=15,
            ),
            CompositeStage(
                name="auto-fix",
                description="Apply auto-fixable changes (lint, format, import sorting).",
                agent_ref="code-review-autofix",
                expected_duration_s=30,
            ),
            CompositeStage(
                name="verification",
                description="Run tests to verify the auto-fix didn't break anything.",
                agent_ref="code-review-verify",
                expected_duration_s=120,
            ),
            CompositeStage(
                name="report",
                description="Generate and post the review report.",
                agent_ref="code-review-report",
                expected_duration_s=10,
            ),
        ],
        tags=("composite", "macro", "review", "pr"),
        aliases=("code-review", "review-pr"),
    )


def lifecycle_tools_for_skill(
    skill_allowed_tools: list[str],
    graph: Any,
    composite_name_map: dict[str, str],
    intent_group_name: str = "agent_lifecycle",
) -> list[str]:
    """Return composite recovery tools that should be prepended to *skill*.

    If any of the skill's allowed tools intersect the tools listed in the
    named intent group (after normalizing graph keys to kebab-case), the
    matching lifecycle recovery composite tools are returned.
    """

    def _norm(key: str) -> str:
        return key.replace(".", "-").replace("_", "-").lower()

    skill_tools = {_norm(t) for t in skill_allowed_tools}
    lifecycle_match = False
    get_intent_group = getattr(graph, "get_intent_group", None)
    if callable(get_intent_group):
        group = get_intent_group(intent_group_name)
        if group is not None:
            group_tools = {_norm(t) for t in getattr(group, "tools", []) if isinstance(t, str)}
            lifecycle_match = bool(skill_tools & group_tools)

    # Some SDK factories lack a return annotation, so dependency inference
    # cannot place them in agent_lifecycle. Keep the macro discoverable when
    # the same skill clearly offers both create/build and invoke/run agent APIs.
    has_create = any("agent" in tool and ("create" in tool or "build" in tool) for tool in skill_tools)
    has_invoke = any("agent" in tool and ("invoke" in tool or "run" in tool or "call" in tool) for tool in skill_tools)
    if not lifecycle_match and not (has_create and has_invoke):
        return []

    recovery_map: dict[str, list[str]] = {
        # Agent product facade + generic resume for non-agent / typed recovery.
        # Keys must match CompositeToolSpec.name / register_composite_tools name_map.
        "agent_lifecycle": ["invoke_existing_agent", "resume_resource"],
    }
    result: list[str] = []
    for spec_name in recovery_map.get(intent_group_name, []):
        registered = composite_name_map.get(spec_name)
        if registered and registered not in result:
            result.append(registered)
    return result


def _invoke_existing_agent(*, bundle_dir: Path | None = None) -> CompositeToolSpec:
    """Invoke a previously-created SOP agent by its persisted reference.

    The tool looks up the agent in the resource catalog, materializes
    the stored SDK resource, and calls its persisted invocation contract.
    """
    del bundle_dir
    return CompositeToolSpec(
        name="invoke_existing_agent",
        description=(
            "Invoke a previously-created agent by its stable agent_id or saved name. "
            "The workflow loads its resource record, materializes "
            "the saved SDK resource, then returns the original invocation output."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_ref": {
                    "type": "string",
                    "description": "Saved agent name or stable agent_id from a prior build/create tool.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Deprecated compatibility alias for agent_ref when using a stable ID.",
                },
                "query": {
                    "type": "string",
                    "description": "Primary query / instruction passed to the agent.",
                },
                "inputs": {
                    "type": "object",
                    "description": "Optional additional inputs merged into the agent call.",
                },
            },
            "anyOf": [{"required": ["agent_ref"]}, {"required": ["agent_id"]}],
        },
        stages=[
            CompositeStage(
                name="resolve-catalog",
                description="Resolve the resource record by saved name or agent_id.",
            ),
            CompositeStage(
                name="materialize",
                description="Import the SDK module and instantiate the saved class.",
            ),
            CompositeStage(
                name="invoke",
                description="Call the agent's invoke/run method with the query.",
            ),
        ],
        tags=(
            "composite",
            "macro",
            "agent-lifecycle",
            "agent",
            "invoke",
            "existing",
            "call-by-reference",
        ),
        aliases=("run-existing-agent", "call-agent-by-id", "call-agent-by-reference"),
        call_type="workflow",
        call_impl={"catalog_id": "builtin:invoke-existing-agent"},
        output_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "output": {},
                "raw": {},
                "text": {"type": "string"},
                "method": {"type": "string"},
                "trace": {"type": "array"},
            },
            "required": ["agent_id", "output", "raw", "text", "trace"],
        },
        query_arg="query",
        workflow_spec=invoke_existing_agent_workflow(),
    )


def _resume_resource(*, bundle_dir: Path | None = None) -> CompositeToolSpec:
    """Resume any registered catalog resource by type and ref."""
    del bundle_dir
    return CompositeToolSpec(
        name="resume_resource",
        description=(
            "Resume a previously persisted catalog resource by resource_type "
            "and resource_ref. Looks up the ResourceHandler registry, materializes "
            "the opaque handle in a trusted private lane, invokes it, and returns "
            "JSON-safe output."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "description": "Registered resource_type (e.g. agent, DemoHandle).",
                },
                "resource_ref": {
                    "type": "string",
                    "description": "Stable resource id / handle from a prior create tool.",
                },
                "query": {
                    "type": "string",
                    "description": "Primary query / instruction passed to the resource.",
                },
                "inputs": {
                    "type": "object",
                    "description": "Optional additional inputs merged into the invoke call.",
                },
            },
            "required": ["resource_type", "resource_ref"],
        },
        stages=[
            CompositeStage(
                name="resolve-catalog",
                description="Resolve the resource record by type and resource_ref.",
            ),
            CompositeStage(
                name="materialize",
                description="Materialize the opaque handle via ResourceHandler.",
            ),
            CompositeStage(
                name="invoke",
                description="Invoke the resource and project JSON-safe public output.",
            ),
        ],
        tags=(
            "composite",
            "macro",
            "resource-lifecycle",
            "resource",
            "resume",
            "catalog",
            "call-by-reference",
        ),
        aliases=("resume-catalog-resource", "invoke-resource-by-ref"),
        call_type="workflow",
        call_impl={"catalog_id": "builtin:resume-resource"},
        output_schema={
            "type": "object",
            "properties": {
                "resource_type": {"type": "string"},
                "resource_ref": {"type": "string"},
                "resource_id": {"type": "string"},
                "output": {},
                "raw": {},
                "text": {"type": "string"},
                "trace": {"type": "array"},
            },
            "required": [
                "resource_type",
                "resource_ref",
                "resource_id",
                "output",
                "raw",
                "text",
                "trace",
            ],
        },
        query_arg="query",
        workflow_spec=resume_resource_workflow(),
    )
