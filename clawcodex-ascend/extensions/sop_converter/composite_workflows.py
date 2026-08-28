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


"""Standard executable composite workflow specifications."""

from __future__ import annotations

from .composite_runtime import CompositeWorkflowSpec, CompositeWorkflowStep


def invoke_existing_agent_workflow() -> CompositeWorkflowSpec:
    """Agent-facing facade over the ResourceHandler recovery chain.

    Product inputs/outputs stay agent-shaped (``agent_ref`` / ``agent_id`` /
    ``method``). Steps go through ``resolve_agent_record`` and the same
    ``materialize_resource`` / ``invoke_resource`` path as
    :func:`resume_resource_workflow`, including ``catalog_dir`` for spilled
    ``payload_ref`` records.
    """
    return CompositeWorkflowSpec(
        name="invoke-existing-agent",
        description="Invoke an SOP-created agent from its catalog record by name or ID.",
        inputs={
            "agent_ref": {"type": "string", "required": False},
            "agent_id": {"type": "string", "required": False},
            "query": {"type": "string", "required": False},
            "inputs": {"type": "object", "required": False},
        },
        steps=(
            CompositeWorkflowStep(
                id="load_agent_record",
                kind="catalog",
                callable_ref="extensions.sop_converter.resource_catalog:resolve_agent_record",
                args={
                    "agent_ref": "$input.agent_ref",
                    "agent_id": "$input.agent_id",
                    "catalog_context": "$resources.catalog",
                },
                visibility="private",
            ),
            CompositeWorkflowStep(
                id="materialize_agent",
                kind="python",
                callable_ref="extensions.sop_converter.resource_runtime:materialize_resource",
                args={
                    "record": "$private.load_agent_record.output.record",
                    "resource_type": "agent",
                    "catalog_dir": "$private.load_agent_record.output.location.path.parent",
                },
                visibility="private",
            ),
            CompositeWorkflowStep(
                id="invoke_agent",
                kind="python",
                callable_ref="extensions.sop_converter.resource_runtime:invoke_resource",
                args={
                    "record": "$private.load_agent_record.output.record",
                    "resource_type": "agent",
                    "query": "$input.query",
                    "inputs": "$input.inputs",
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "resource_id": {"type": "string"},
                        "text": {"type": "string"},
                        "raw": {},
                        "output": {},
                        "method": {"type": "string"},
                    },
                    "required": ["text", "raw", "output"],
                },
            ),
        ),
        outputs={
            "agent_id": "$steps.invoke_agent.output.agent_id",
            "output": "$steps.invoke_agent.output.output",
            "raw": "$steps.invoke_agent.output.raw",
            "text": "$steps.invoke_agent.output.text",
            "method": "$steps.invoke_agent.output.method",
        },
        trusted=True,
    )


def resume_resource_workflow() -> CompositeWorkflowSpec:
    """Return the generic resume workflow over ResourceHandler rows."""
    return CompositeWorkflowSpec(
        name="resume-resource",
        description=(
            "Resume any registered catalog resource by resource_type and "
            "resource_ref, then invoke it through its ResourceHandler."
        ),
        inputs={
            "resource_type": {"type": "string", "required": True},
            "resource_ref": {"type": "string", "required": True},
            "query": {"type": "string", "required": False},
            "inputs": {"type": "object", "required": False},
        },
        steps=(
            CompositeWorkflowStep(
                id="load_resource_record",
                kind="catalog",
                callable_ref="extensions.sop_converter.resource_catalog:resolve_record",
                args={
                    "resource_ref": "$input.resource_ref",
                    "resource_type": "$input.resource_type",
                    "catalog_context": "$resources.catalog",
                },
                visibility="private",
            ),
            CompositeWorkflowStep(
                id="materialize_resource",
                kind="python",
                callable_ref="extensions.sop_converter.resource_runtime:materialize_resource",
                args={
                    "record": "$private.load_resource_record.output.record",
                    "resource_type": "$input.resource_type",
                    "catalog_dir": "$private.load_resource_record.output.location.path.parent",
                },
                visibility="private",
            ),
            CompositeWorkflowStep(
                id="invoke_resource",
                kind="python",
                callable_ref="extensions.sop_converter.resource_runtime:invoke_resource",
                args={
                    "record": "$private.load_resource_record.output.record",
                    "resource_type": "$input.resource_type",
                    "query": "$input.query",
                    "inputs": "$input.inputs",
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "resource_type": {"type": "string"},
                        "resource_ref": {"type": "string"},
                        "resource_id": {"type": "string"},
                        "text": {"type": "string"},
                        "raw": {},
                        "output": {},
                    },
                    "required": [
                        "resource_type",
                        "resource_ref",
                        "resource_id",
                        "text",
                        "raw",
                        "output",
                    ],
                },
            ),
        ),
        outputs={
            "resource_type": "$steps.invoke_resource.output.resource_type",
            "resource_ref": "$steps.invoke_resource.output.resource_ref",
            "resource_id": "$steps.invoke_resource.output.resource_id",
            "output": "$steps.invoke_resource.output.output",
            "raw": "$steps.invoke_resource.output.raw",
            "text": "$steps.invoke_resource.output.text",
        },
        trusted=True,
    )


__all__ = ["invoke_existing_agent_workflow", "resume_resource_workflow"]
