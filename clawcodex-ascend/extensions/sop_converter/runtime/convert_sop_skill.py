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

"""ConvertPOSToAgent skill — converts an SOP into a reusable Agent.

This is the execution layer (Skill) for the SOP conversion pattern.
It takes SDK specifications and business requirements, then produces:
  1. An AgentDefinition with grouped Skills
  2. SKILL.md files for each Skill
  3. Optional agent persistence file for long-term use

Usage:
    /convert-sop-to-agent <sdk_spec> [--requirements "<requirements>"]

Examples:
    /convert-sop-to-agent https://openapi.example.com/spec.json
    /convert-sop-to-agent docker,kubectl,k8s_apply --requirements "CI/CD pipeline"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core import (
    SdkParser,
    MappingRule,
)
from .skill_grouper import SkillGrouper
from .agent_builder import AgentBuilder
from .tool_registry_bridge import register_http_tools

logger = logging.getLogger(__name__)

# Default mapping rules for common SDK patterns
_DEFAULT_RULES: list[MappingRule] = [
    MappingRule("docker_build", "docker_build", "build_image", "Build Docker image"),
    MappingRule("docker_tag", "docker_tag", "build_image", "Tag Docker image"),
    MappingRule("docker_push", "docker_push", "build_image", "Push Docker image"),
    MappingRule("k8s_apply", "k8s_apply", "deploy_service", "Apply Kubernetes manifest"),
    MappingRule("k8s_delete", "k8s_delete", "deploy_service", "Delete Kubernetes resource"),
    MappingRule("k8s_get", "k8s_get", "deploy_service", "Get Kubernetes resource"),
    MappingRule("health_check", "health_check", "deploy_service", "Check service health"),
    MappingRule("rollback", "rollback", "deploy_service", "Rollback deployment"),
    MappingRule("slack_send", "slack_send", "notify_team", "Send Slack notification"),
    MappingRule("email_send", "email_send", "notify_team", "Send email notification"),
    MappingRule("s3_upload", "s3_upload", "upload_artifact", "Upload to S3"),
    MappingRule("s3_download", "s3_download", "upload_artifact", "Download from S3"),
    MappingRule("spark_submit", "spark_submit", "run_spark", "Submit Spark job"),
    MappingRule("etl_run", "etl_run", "run_etl", "Run ETL pipeline"),
    MappingRule("train_model", "train_model", "train_model", "Train ML model"),
    MappingRule("predict", "predict", "run_inference", "Run inference"),
]


def convert_sop_to_agent(
    sdk_spec: str | dict[str, Any],
    requirements: str = "",
    agent_name: str = "",
    agent_description: str = "",
    model: str | None = None,
    mapping_rules: list[MappingRule] | None = None,
    *,
    register_http: bool = True,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert an SOP spec into a reusable Agent.

    Args:
        sdk_spec: SDK specification (OpenAPI dict, URL, or method list string).
        requirements: Business requirements for skill grouping.
        agent_name: Name for the resulting agent (auto-generated if empty).
        agent_description: Description for the resulting agent.
        model: Optional model override for the agent.
        mapping_rules: Optional custom mapping rules.
        register_http: If True (default), register HTTP tools for OpenAPI specs.

    Returns:
        Dict with agent definition, skill specs, and any warnings.
    """
    rules = mapping_rules or _DEFAULT_RULES

    # Step 1: Parse SDK → atomic tools
    parser = SdkParser(sdk_spec)
    methods = parser.parse()

    if not methods:
        return {
            "status": "error",
            "error": "No SDK methods parsed from spec",
        }

    # Step 2: Group tools → Skills
    grouper = SkillGrouper(methods, mapping_rules=rules)
    skills = grouper.group(requirements)

    if not skills:
        return {
            "status": "error",
            "error": "No skills produced from grouping",
        }

    # Step 3: Register HTTP tools for OpenAPI specs
    name_map: dict[str, str] = {}
    is_openapi = any(m.http_method is not None for m in methods)
    if register_http and is_openapi:
        try:
            name_map = register_http_tools(methods, persist=True, bundle_dir=bundle_dir)
            # Update skill allowed_tools with kebab-case names
            for skill in skills:
                updated_tools = []
                for tool in skill.allowed_tools:
                    updated_tools.append(name_map.get(tool, tool))
                skill.allowed_tools = updated_tools
        except Exception as exc:
            logger.warning("Failed to register HTTP tools: %s", exc)

    # Step 4: Build Agent from Skills
    resolved_name = agent_name or _generate_agent_name(requirements)
    resolved_desc = agent_description or (f"Agent for: {requirements}" if requirements else "Agent converted from SDK")
    resolved_model = model

    builder = AgentBuilder(
        skills=skills,
        agent_name=resolved_name,
        agent_description=resolved_desc,
        model=resolved_model,
        mapping_rules=rules,
    )
    result = builder.build()

    # Step 5: Persist agent for long-term use
    try:
        from extensions.sop_converter.agent_builder import persist_converted_agent

        persist_converted_agent(result.agent, skills)
        persist_status = "saved"
    except Exception as exc:
        logger.warning("Failed to persist converted agent: %s", exc)
        persist_status = f"save_failed: {exc}"

    return {
        "status": "converted",
        "agent_type": result.agent.agent_type,
        "agent_description": result.agent.when_to_use,
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "tools": s.allowed_tools,
            }
            for s in skills
        ],
        "model": result.agent.model,
        "tools": result.agent.tools or [],
        "skill_files": [str(p) for p in result.skill_files],
        "persist_status": persist_status,
        "warnings": result.warnings,
        "is_openapi": is_openapi,
        "registered_http_tools": len(name_map),
    }


def _generate_agent_name(requirements: str) -> str:
    """Generate a valid agent name from requirements."""
    import re

    name = requirements.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"^-+|-+$", "", name)
    if not name:
        name = "converted-agent"
    if len(name) > 40:
        name = name[:40]
    return name


def get_prompt_for_command(args: str) -> str:
    """Build the skill prompt with substitution support."""
    if not args:
        return _SKILL_PROMPT

    parts = args.split("::")
    sdk_spec = parts[0].strip()

    requirements = ""
    if len(parts) >= 2:
        requirements = parts[1].strip()

    agent_name = ""
    agent_description = ""
    if len(parts) >= 3:
        agent_name = parts[2].strip()

    result = convert_sop_to_agent(
        sdk_spec=sdk_spec,
        requirements=requirements,
        agent_name=agent_name,
        agent_description=agent_description,
    )

    return _format_result(result)


def _format_result(result: dict[str, Any]) -> str:
    """Format the conversion result as a readable string."""
    if result["status"] == "error":
        return f"Conversion failed: {result.get('error', 'unknown error')}"

    lines = [
        f"✅ Converted SOP: **{result['agent_type']}**",
        f"\nDescription: {result['agent_description']}",
        f"\nModel: {result.get('model', 'default')}",
        f"\n## Skills ({len(result['skills'])})",
    ]

    for skill in result["skills"]:
        lines.append(f"\n### {skill['name']}")
        lines.append(f"{skill['description']}")
        lines.append(f"Tools: {', '.join(skill['tools'])}")

    lines.append(f"\n## Tools ({len(result.get('tools', []))})")
    lines.append(", ".join(result.get("tools", [])))

    if result.get("is_openapi"):
        lines.append("\n📡 OpenAPI Mode: Yes")
        lines.append(f"Registered HTTP tools: {result.get('registered_http_tools', 0)}")

    lines.append(f"\nPersistence: {result['persist_status']}")
    lines.append(f"Skill files: {', '.join(result.get('skill_files', []))}")

    if result.get("warnings"):
        lines.append(f"\nWarnings: {'; '.join(result['warnings'])}")

    return "\n".join(lines)


_SKILL_PROMPT = """\
# ConvertSOP Skill

Convert a Standard Operating Procedure (SOP) into a reusable Agent.

## Input Format
```
/convert-sop-to-agent <sdk_spec> [--requirements "<requirements>"]
```

## Arguments
- `sdk_spec`: OpenAPI URL, JSON spec string, or comma-separated method list
- `requirements`: Business context (e.g., "CI/CD pipeline", "data processing")

## Three-Layer Mapping
1. **SOP** (Standard Operating Procedure) → **Agent**
2. **workflow steps** → **Skill**
3. **SDK interfaces** → **atomic tools**

## Example
```
/convert-sop-to-agent docker_build,docker_tag,docker_push,k8s_apply,health_check --requirements "CI/CD pipeline"
```

## Output
- Agent definition with grouped skills
- SKILL.md files for each skill
- Agent persisted to ~/.clawcodex/agents/<name>.json
"""
