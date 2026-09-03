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


"""Agent definition generator for per-stage agents."""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from extensions.sop_converter.runtime.agent_md_writer import AgentMarkdownWriter, WorkflowStage
from extensions.sop_converter.runtime.skill_grouper import SkillSpec

from ..ast_helpers import extract_callee_names, to_kebab
from ..capability.models import ExecutionMode, StageAgentMap
from ..extractors.models import ExtractedStage, WorkflowGraph
from ..mapping import build_workflow_stages_with_map
from .overview_gen import enrich_workflow_stages
from .skill_gen import write_stage_skill
from .artifact_semantics import output_descriptions
from .tool_gen import stage_agent_tool_names, tools_for_profile

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
# Maximum number of trailing snake_case segments used to build kebab-case
# matching variants when scoping a stage's tools to its implementation file.
_MAX_NAME_SEGMENTS = 4


def _resolve_stage_impl_path(source_dir: Path, stage: ExtractedStage) -> Path | None:
    """Find the implementation file for a stage (same logic as mapper)."""
    if stage.file_path:
        p = source_dir / stage.file_path
        if p.is_file():
            return p
    for sub in ("stage_impls", "stages", "pipeline"):
        candidate = source_dir / sub / f"{stage.name.replace('-', '_')}.py"
        if candidate.is_file():
            return candidate
        candidate = source_dir / sub / f"{stage.name}.py"
        if candidate.is_file():
            return candidate
    return None


def coarse_agent_skills(
    grouped_skills: list[SkillSpec],
    workflow_graph: WorkflowGraph | None,
) -> list[SkillSpec]:
    """Skills that map to coarse SDK agents — excludes per-stage synthesized skills."""
    if workflow_graph is None:
        return grouped_skills
    stage_names = {stage.name for stage in workflow_graph.stages}
    return [skill for skill in grouped_skills if skill.name not in stage_names]


def stage_agent_existing_names(
    grouped_skills: list[SkillSpec],
    workflow_graph: WorkflowGraph,
    *,
    overview_agent_name: str | None = None,
) -> set[str]:
    """Names already claimed before resolving per-stage agent markdown.

    Coarse domain agents, the overview agent, and summary ``{stage}-agent``
    slots from grouped skills (which collide with stage agent base names).
    """
    names = {f"{s.name}-agent" for s in coarse_agent_skills(grouped_skills, workflow_graph)}
    if overview_agent_name:
        names.add(overview_agent_name)
    stage_names = {s.name for s in workflow_graph.stages}
    for skill in grouped_skills:
        if skill.name in stage_names:
            names.add(f"{skill.name}-agent")
    return names


class AgentDefinitionGenerator:
    """Generate per-stage agent markdown by execution_mode."""

    def __init__(self, writer: AgentMarkdownWriter | None = None) -> None:
        self._writer = writer or AgentMarkdownWriter()
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
        )

    def _resolve_agent_name(
        self,
        stage_name: str,
        project_prefix: str,
        existing_names: set[str],
    ) -> str:
        base = f"{stage_name}-agent"
        if base not in existing_names:
            return base
        prefixed = f"{project_prefix}-{stage_name}-agent"
        if prefixed not in existing_names:
            return prefixed
        suffix = 2
        while f"{project_prefix}-{stage_name}-{suffix}-agent" in existing_names:
            suffix += 1
        return f"{project_prefix}-{stage_name}-{suffix}-agent"

    def finalize_stage_agent_names(
        self,
        graph: WorkflowGraph,
        agent_map: StageAgentMap,
        *,
        project_name: str = "project",
        existing_agent_names: set[str] | None = None,
    ) -> None:
        """Resolve final stage agent names into ``profile.mapped_agent``.

        Uses the same rules as :meth:`generate_stage_agents` (base
        ``{stage}-agent``, then ``{project_name}-{stage}-agent`` on collision).
        Call before emitting workflow.yaml or overview so references match
        on-disk agent markdown.
        """
        used_names = set(existing_agent_names or [])
        for stage in graph.stages:
            profile = agent_map.profile_for_stage(stage.id)
            if not profile:
                continue
            if profile.mapping_confidence <= 0.0 and not profile.mapped_skill:
                continue

            if profile.mapped_agent and profile.mapped_agent not in used_names:
                agent_name = profile.mapped_agent
            else:
                agent_name = self._resolve_agent_name(stage.name, project_name, used_names)
            profile.mapped_agent = agent_name
            used_names.add(agent_name)
            if profile.mapped_skill:
                agent_map.skill_to_agent[profile.mapped_skill] = agent_name

    def _bridge_tool_name(self, project_name: str) -> str:
        from ..bridge.mcp_adapter import bridge_tool_name

        return bridge_tool_name(project_name)

    @staticmethod
    def _scope_tools(
        source_dir: Path,
        stage: ExtractedStage,
        profile_tools: list[str],
    ) -> list[str]:
        """Filter *profile_tools* to only those called in the stage impl file.

        Falls back to the full list when the impl file is missing / unparseable,
        or when the intersection is empty (stage only calls internal helpers).
        """
        impl_path = _resolve_stage_impl_path(source_dir, stage)
        if impl_path is None:
            return profile_tools

        callee_names = extract_callee_names(impl_path)
        if not callee_names:
            return profile_tools

        # Convert callee snake_case names → kebab-case for matching
        callee_kebab = {to_kebab(n) for n in callee_names}
        # Also collect snake→kebab variants of just the terminal component
        for name in callee_names:
            parts = name.split("_")
            for n in range(1, min(_MAX_NAME_SEGMENTS, len(parts) + 1)):
                callee_kebab.add("-".join(parts[-n:]))

        scoped = [t for t in profile_tools if t.replace("-", "_") in callee_names or t in callee_kebab]
        return scoped if scoped else profile_tools

    def generate_stage_agents(
        self,
        graph: WorkflowGraph,
        agent_map: StageAgentMap,
        output_dir: Path,
        *,
        project_name: str = "project",
        bridge_script: str | None = None,
        write_skills: bool = False,
        existing_agent_names: set[str] | None = None,
        composite_tools: list[str] | None = None,
        fallback_tool_name: str | None = None,
    ) -> list[Path]:
        """Render stage agent markdown files into output_dir.

        Parameters
        ----------
        composite_tools:
            Registered composite (macro) tool kebab-case names to inject into
            every stage agent's frontmatter ``tools`` list.  These are higher-
            level orchestration tools (e.g. ``invoke-existing-agent``,
            ``pipeline-execute``) that stage agents may need to invoke.
        fallback_tool_name:
            Optional SDK-specific tool name to mention when a stage has no
            registered tools.  The generator does not invent a tool name when
            this value is omitted.
        """
        output_dir = Path(output_dir)
        source_dir = Path(graph.source_dir)
        self.finalize_stage_agent_names(
            graph,
            agent_map,
            project_name=project_name,
            existing_agent_names=existing_agent_names,
        )
        paths: list[Path] = []
        bridge_tool = self._bridge_tool_name(project_name) if bridge_script else None

        for stage in graph.stages:
            profile = agent_map.profile_for_stage(stage.id)
            if not profile:
                continue
            if profile.mapping_confidence <= 0.0 and not profile.mapped_skill:
                logger.warning("Skipping unmapped stage %s (id=%s)", stage.name, stage.id)
                continue

            agent_name = profile.mapped_agent
            if not agent_name:
                continue

            # Scope tools to those actually called in the stage implementation
            scoped_tools = self._scope_tools(source_dir, stage, profile.recommended_tools)
            profile.recommended_tools = scoped_tools
            tools = stage_agent_tool_names(tools_for_profile(profile, bridge_tool=bridge_tool))

            # Gap 2: inject composite macro tools into every stage agent
            # frontmatter so they can reference higher-level orchestration
            # tools (pipeline-execute, invoke-existing-agent, …).
            if composite_tools:
                for ct in composite_tools:
                    if ct not in tools:
                        tools.append(ct)

            contract = graph.contracts.get(stage.id)
            input_files = contract.input_files if contract else []
            output_files = contract.output_files if contract else []
            wrapper_command = (
                f"python {bridge_script} --stage-id {stage.id} --project-dir <workspace>"
                if bridge_script
                else f"# bridge not generated; stage {stage.id}"
            )

            stage_skill_name = f"{stage.name}-skill"

            stage_dod = contract.dod if contract else ""
            primary_tool = tools[0] if tools else fallback_tool_name
            tool_search_step = (
                f"ToolSearch 找到主工具：`{primary_tool}`" if primary_tool else "ToolSearch 查找与当前阶段匹配的主工具"
            )
            ctx = {
                "agent_name": agent_name,
                "description": stage.description or f"Agent for stage {stage.label}",
                "stage_label": stage.label,
                "stage_description": stage.description,
                "stage_skill_name": stage_skill_name,
                "tools": tools,
                "skills": [stage_skill_name],
                "steps": [
                    f'调用 `Skill(skill="{stage_skill_name}")`（阻塞，仅一次）',
                    tool_search_step,
                    f'调用主工具：`stage`="{stage.label}" 或 `stage_id`={stage.id}，`run_dir`/`project_dir`=<run_dir>',
                    "验证输出契约",
                ],
                "input_files": input_files,
                "output_files": output_files,
                "contract_dod": stage_dod,
                "output_descriptions": output_descriptions(output_files, stage_dod=stage_dod),
                "bridge_tool": bridge_tool,
                "wrapper_command": wrapper_command,
            }

            mode = profile.execution_mode
            if mode == ExecutionMode.WRAPPER:
                template_name = "agent_wrapper.md.j2"
            elif mode == ExecutionMode.HYBRID:
                template_name = "agent_hybrid.md.j2"
            else:
                template_name = "agent_native.md.j2"

            body = self._jinja.get_template(template_name).render(**ctx)
            agents_dir = output_dir / ".claude" / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            out_path = agents_dir / f"{agent_name}.md"
            out_path.write_text(body, encoding="utf-8")
            paths.append(out_path)

            if write_skills:
                skill_path = write_stage_skill(stage, graph, output_dir, tools=scoped_tools)
                if skill_path:
                    paths.append(skill_path)

        return paths

    def enrich_workflow_stages(
        self,
        graph: WorkflowGraph,
        agent_map: StageAgentMap,
        *,
        skill_agent_map: dict[str, str],
    ) -> list[WorkflowStage]:
        base = build_workflow_stages_with_map(graph, agent_map, skill_agent_map=skill_agent_map)
        return enrich_workflow_stages(graph, agent_map, base)
