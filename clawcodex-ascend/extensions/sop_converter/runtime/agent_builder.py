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
# tech_v26.2.0 has not yet merged package-marker files such as
# extensions/__init__.py, so pylint cannot recognize sop_converter as a Python
# package and flags legitimate relative imports as E0402. Once the package
# markers are merged, remove this tag to restore content identical to source_ref.


"""AgentBuilder — builds an AgentDefinition from grouped Skills.

Fills the Agent definition template using Skill specs and metadata.
The resulting AgentDefinition can be registered and persisted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extensions.capabilities.agent_definition_protocol import (
    AgentDefinitionProtocol,
    AgentToolConstants,
)
from ..core.source_parser import SourceComponent
from .agent_md_writer import AgentMarkdownWriter, AgentComponentInfo
from .skill_grouper import SkillSpec, MappingRule
from ..core.templates import SKILL_TEMPLATE
from .task_guide import append_task_guide_to_skill_body, generate_task_guide_markdown
from ..adapters import DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class AgentBuildResult:
    """Result of building an Agent from SOP conversion."""

    agent: AgentDefinitionProtocol
    skill_files: list[Path] = field(default_factory=list)
    markdown_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentBuilder:
    """Build an AgentDefinition from SkillSpecs and metadata.

    Takes grouped SkillSpecs plus agent metadata (name, description, model,
    tools, memory_scope) and fills the Agent definition template.
    """

    def __init__(
        self,
        skills: list[SkillSpec],
        *,
        agent_name: str,
        agent_description: str,
        model: str | None = None,
        tools: list[str] | None = None,
        memory_scope: list[str] | None = None,
        persistent: bool = True,
        mapping_rules: list[MappingRule] | None = None,
        source_components: list[SourceComponent] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._skills = skills
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._model = model
        self._tools = tools
        self._memory_scope = memory_scope or []
        self._persistent = persistent
        self._mapping_rules = mapping_rules
        self._source_components = source_components or []
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()
        self._result: AgentBuildResult | None = None
        self._built_format: str | None = None

    def build(self, format: str = "agent_definition") -> AgentBuildResult:  # pylint: disable=redefined-builtin
        """Build the AgentDefinition and optionally persist Skill files.

        Args:
            format: Output format — ``"agent_definition"`` (default, old behavior),
                    ``"markdown"``, or ``"both"``.

        Returns:
            AgentBuildResult with the built AgentDefinition and generated file paths.
        """
        valid_formats = {"agent_definition", "markdown", "both"}
        if format not in valid_formats:
            raise ValueError(f"Invalid format {format!r}. Must be one of {valid_formats}")

        if self._result is not None:
            if self._built_format != format:
                raise ValueError(
                    f"AgentBuilder already built with format {self._built_format!r}; "
                    f"cannot rebuild with {format!r}. Create a new AgentBuilder instead."
                )
            return self._result

        domain_tools = self._tools or self._collect_tools()
        domain_tools.sort()
        agent = self._build_agent(domain_tools)

        skill_files, warnings = self._write_skill_files()

        md_files: list[Path] = []
        if format in ("markdown", "both"):
            try:
                md_files = self._write_agent_markdown(domain_tools)
            except Exception as exc:
                warnings.append(f"Failed to write agent markdown: {exc}")

        self._result = AgentBuildResult(
            agent=agent,
            skill_files=skill_files,
            warnings=warnings,
        )
        self._result.markdown_files = md_files
        self._built_format = format
        return self._result

    def _collect_tools(self) -> list[str]:
        """Collect all tools from grouped skills."""
        tools: dict[str, bool] = {}
        for spec in self._skills:
            for tool in spec.allowed_tools:
                tools[tool] = True
        return list(tools.keys())

    def _resolve_allowed_tools(self, domain_tools: list[str]) -> list[str]:
        """Choose the agent tool allowlist, capping inline tool display."""
        if len(domain_tools) > AgentToolConstants.MAX_INLINE_TOOL_DISPLAY:
            return AgentToolConstants.registered_proxy_base_tools()
        return domain_tools

    def _build_agent(self, domain_tools: list[str]) -> AgentDefinitionProtocol:
        """Build the AgentDefinition from grouped skills and metadata."""
        return DEFAULTS.agent_definition_factory(
            agent_type=self._agent_name,
            when_to_use=self._agent_description,
            tools=self._resolve_allowed_tools(domain_tools),
            skills=[s.name for s in self._skills],
            source="dynamic",
            base_dir="dynamic",
            model=self._model,
        )

    def _write_skill_files(self) -> tuple[list[Path], list[str]]:
        """Persist one SKILL.md per grouped skill; return (paths, warnings)."""
        skill_files: list[Path] = []
        warnings: list[str] = []
        for spec in self._skills:
            try:
                path = _write_skill_file(
                    spec,
                    mapping_rules=self._mapping_rules,
                    source_components=self._source_components,
                    bundle=self._output_dir,
                )
                skill_files.append(path)
            except Exception as exc:
                warnings.append(f"Failed to write skill file for {spec.name}: {exc}")
        return skill_files, warnings

    def _write_agent_markdown(self, domain_tools: list[str]) -> list[Path]:
        """Generate agent markdown files using AgentMarkdownWriter.

        Returns list of generated file paths.
        """
        writer = AgentMarkdownWriter()
        md_files: list[Path] = []

        agent_def = {
            "name": self._agent_name,
            "description": self._agent_description,
            "model": self._model,
            "tools": domain_tools,
            "skills": [s.name for s in self._skills],
        }
        agent_path = writer.write_agent(agent_def, self._output_dir, bundle=self._output_dir)
        md_files.append(agent_path)

        # Write skills
        lifecycle_graph = _build_lifecycle_graph(self._source_components)
        lifecycle_tools_for_skill = _load_lifecycle_tools(lifecycle_graph)
        lifecycle_deps_ref = _lifecycle_deps_ref(self._output_dir)

        skill_dicts = [
            _skill_markdown_dict(
                spec,
                source_components=self._source_components,
                output_dir=self._output_dir,
                lifecycle_graph=lifecycle_graph,
                lifecycle_tools_for_skill=lifecycle_tools_for_skill,
                lifecycle_deps_ref=lifecycle_deps_ref,
            )
            for spec in self._skills
        ]
        skill_paths = writer.write_skills(skill_dicts, self._output_dir, bundle=self._output_dir)
        md_files.extend(skill_paths)

        # Build overview from grouped skills (not raw source_components).
        overview_info = _overview_components(self._skills)
        if len(overview_info) > 1:
            overview_path = writer.write_overview_agent(
                name="clawcodex-overview",
                description=f"Overview agent for {self._agent_name}",
                component_agents=overview_info,
                workflow_stages=[],
                output_dir=self._output_dir,
                model=self._model or "default",
            )
            md_files.append(overview_path)

        return md_files


def _overview_components(skills: list[SkillSpec]) -> list[AgentComponentInfo]:
    """Build overview agent components from grouped skills."""
    return [
        AgentComponentInfo(
            name=f"{skill.name}-agent",
            description=skill.description,
            capabilities=skill.allowed_tools[:5],
            invoke_pattern=f"@{skill.name}-agent {{task}}",
        )
        for skill in skills
    ]


def _lifecycle_deps_ref(output_dir: Path) -> str:
    """Return the relative lifecycle-deps path when the manifest exists."""
    if (output_dir / ".clawcodex" / "tool-dependencies.yaml").exists():
        return ".clawcodex/tool-dependencies.yaml"
    return ""


def _skill_markdown_dict(
    spec: SkillSpec,
    *,
    source_components: list[SourceComponent],
    output_dir: Path,
    lifecycle_graph: Any | None,
    lifecycle_tools_for_skill: Any | None,
    lifecycle_deps_ref: str,
) -> dict[str, Any]:
    """Build the writer payload dict for one skill."""
    guide = ""
    if source_components:
        guide = generate_task_guide_markdown(spec, source_components, bundle=output_dir)
    return {
        "name": spec.name,
        "description": spec.description,
        "allowed_tools": _apply_lifecycle_tools(
            spec,
            lifecycle_graph,
            lifecycle_tools_for_skill,
        ),
        "parameters": [],
        "source_code": "",
        "task_guide": guide,
        "lifecycle_deps": lifecycle_deps_ref,
    }


def _build_lifecycle_graph(source_components: list[SourceComponent]) -> Any | None:
    """Detect the lifecycle dependency graph from source components."""
    if not source_components:
        return None
    try:
        from extensions.sop_converter.core.dependency.models import ToolDependencyGraph

        return ToolDependencyGraph.detect_from_components(source_components)
    except Exception as exc:
        logger.warning("Failed to detect lifecycle graph: %s", exc)
        return None


def _load_lifecycle_tools(lifecycle_graph: Any | None) -> Any | None:
    """Import the lifecycle tool computation helper, if the graph is available."""
    if lifecycle_graph is None:
        return None
    try:
        from extensions.sop_converter.runtime.composite_tools.builtin import (
            lifecycle_tools_for_skill,
        )

        return lifecycle_tools_for_skill
    except Exception as exc:
        logger.warning("Failed to import lifecycle_tools_for_skill: %s", exc)
        return None


def _apply_lifecycle_tools(
    spec: SkillSpec,
    lifecycle_graph: Any | None,
    lifecycle_tools_for_skill: Any | None,
) -> list[str]:
    """Prepend lifecycle tools to a skill's allowed tools, when computable."""
    if lifecycle_graph is None or lifecycle_tools_for_skill is None:
        return list(spec.allowed_tools)
    try:
        extras = lifecycle_tools_for_skill(
            list(spec.allowed_tools),
            lifecycle_graph,
            {
                "invoke_existing_agent": "invoke-existing-agent",
                "resume_resource": "resume-resource",
            },
        )
    except Exception as exc:
        logger.warning("Failed to compute lifecycle tools for skill %s: %s", spec.name, exc)
        return list(spec.allowed_tools)
    return _merge_lifecycle_extras(spec.allowed_tools, extras)


def _merge_lifecycle_extras(allowed_tools: list[str], extras: list[str]) -> list[str]:
    """Prepend extra tools that are not already allowed."""
    merged = list(allowed_tools)
    for tool_name in extras:
        if tool_name not in merged:
            merged.insert(0, tool_name)
    return merged


def _write_skill_file(
    spec: SkillSpec,
    *,
    mapping_rules: list[MappingRule] | None = None,
    source_components: list[SourceComponent] | None = None,
    bundle: str | Path | None = None,
) -> Path:
    """Write a SKILL.md file from a SkillSpec."""
    rules = mapping_rules or []
    rule = next((r for r in rules if r.skill_name == spec.name), None)

    skill_dir = Path.home() / ".clawcodex" / "skills" / spec.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    frontmatter_lines = _skill_frontmatter_lines(spec, rule, bundle)

    body = SKILL_TEMPLATE.format(
        skill_name=spec.name,
        description=spec.description,
        description_lower=spec.description.lower(),
        tool_count=len(spec.allowed_tools),
    )
    if source_components:
        body = append_task_guide_to_skill_body(body, spec, source_components, bundle=bundle)

    content = "\n".join(frontmatter_lines) + "\n\n" + body
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def _skill_frontmatter_lines(
    spec: SkillSpec,
    rule: MappingRule | None,
    bundle: str | Path | None,
) -> list[str]:
    """Build the SKILL.md frontmatter lines for a SkillSpec.

    Prefers the mapping rule's description when present to avoid duplicate
    ``when_to_use`` keys in the generated frontmatter.
    """
    lines = [
        "---",
        f"name: {spec.name}",
        f"description: {spec.description}",
        "user-invocable: true",
    ]
    when_to_use = _preferred_when_to_use(spec, rule)
    if when_to_use:
        lines.append(f"when_to_use: {when_to_use}")
    _append_frontmatter_list(lines, "arguments", spec.argument_names)
    _append_frontmatter_list(lines, "allowed-tools", spec.allowed_tools)
    if bundle is not None and (Path(bundle) / ".clawcodex" / "tool-dependencies.yaml").exists():
        lines.append("lifecycle-deps: .clawcodex/tool-dependencies.yaml")
    lines.append("---")
    return lines


def _preferred_when_to_use(spec: SkillSpec, rule: MappingRule | None) -> str | None:
    """Return the effective ``when_to_use`` text, preferring the mapping rule.

    Only one of *spec.when_to_use* or *rule.description* is returned so the
    generated frontmatter never contains duplicate ``when_to_use`` keys.
    """
    if rule and rule.description:
        return rule.description
    return spec.when_to_use


def _append_frontmatter_list(lines: list[str], key: str, items: list[str] | None) -> None:
    """Append a ``key:`` + indented ``- item`` block when items are present."""
    if not items:
        return
    lines.append(f"{key}:")
    for item in items:
        lines.append(f"  - {item}")


def write_agent_markdown(agent: AgentDefinitionProtocol, path: Path) -> None:
    """Write an AgentDefinitionProtocol as a markdown file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _agent_markdown_lines(agent)
    path.write_text("\n".join(lines), encoding="utf-8")


def _agent_markdown_lines(agent: AgentDefinitionProtocol) -> list[str]:
    """Build markdown lines (frontmatter + body) for an AgentDefinition."""
    lines = [
        "---",
        f"name: {agent.agent_type}",
        f"description: {agent.when_to_use}",
    ]
    if agent.model:
        lines.append(f"model: {agent.model}")
    if agent.tools:
        lines.append(f"tools: [{', '.join(agent.tools)}]")
    if agent.skills:
        lines.append(f"skills: [{', '.join(agent.skills)}]")
    if agent.memory:
        lines.append(f"memory: {agent.memory}")
    lines.append("---")
    lines.append("")
    lines.append(agent.when_to_use or "")
    return lines


@dataclass
class AgentPersistenceSpec:
    """JSON-serializable agent spec for persistence."""

    name: str
    description: str
    model: str | None = None
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    memory_scope: list[str] = field(default_factory=list)
    persistent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "tools": self.tools,
            "skills": self.skills,
            "memory_scope": self.memory_scope,
            "persistent": self.persistent,
        }

    @classmethod
    def from_agent(cls, agent: AgentDefinitionProtocol) -> AgentPersistenceSpec:
        return cls(
            name=agent.agent_type,
            description=agent.when_to_use,
            model=agent.model,
            tools=agent.tools or [],
            skills=agent.skills or [],
            memory_scope=[agent.memory] if agent.memory else [],
        )

    def save(self, agents_dir: Path | None = None) -> Path:
        agents_dir = agents_dir or (Path.home() / ".clawcodex" / "agents")
        agents_dir.mkdir(parents=True, exist_ok=True)
        path = agents_dir / f"{self.name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str, agents_dir: Path | None = None) -> AgentPersistenceSpec | None:
        agents_dir = agents_dir or (Path.home() / ".clawcodex" / "agents")
        path = agents_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                model=data.get("model"),
                tools=data.get("tools", []),
                skills=data.get("skills", []),
                memory_scope=data.get("memory_scope", []),
                persistent=data.get("persistent", True),
            )
        except (json.JSONDecodeError, KeyError):
            return None


def persist_converted_agent(
    agent: AgentDefinitionProtocol,
    skills: list[SkillSpec],
    agents_dir: Path | None = None,
) -> AgentPersistenceSpec:
    """Persist a converted Agent to disk for long-term use."""
    spec = AgentPersistenceSpec.from_agent(agent)
    spec.save(agents_dir=agents_dir)

    for skill_spec in skills:
        try:
            _write_skill_file(skill_spec)
        except Exception as exc:
            logger.warning("Failed to write skill file for %s: %s", skill_spec.name, exc)

    return spec
