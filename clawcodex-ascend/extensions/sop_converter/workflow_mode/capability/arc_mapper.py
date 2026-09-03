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


"""ARC (AutoResearchClaw) stage to skill mapping helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from extensions.sop_converter.runtime.skill_grouper import SkillSpec
from extensions.sop_converter.core.source_parser import SourceComponent

from ..extractors.pattern import resolve_pipeline_dir
from ..extractors.models import ExtractedStage, WorkflowGraph

logger = logging.getLogger(__name__)

_EXECUTOR_BASENAMES = frozenset({"executor.py", "executor"})


def _to_kebab(name: str) -> str:
    return name.replace("_", "-").lower()


def _norm_rel(path: Path, source_dir: Path) -> str:
    return path.resolve().relative_to(source_dir.resolve()).as_posix()


def build_entry_function_impl_index(pipeline_dir: Path) -> dict[str, str]:
    """Map ``_execute_*`` entry function names to repo-relative impl module paths."""
    index: dict[str, str] = {}
    pipeline_dir = pipeline_dir.resolve()
    search_dirs = [pipeline_dir / "stage_impls", pipeline_dir]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for py_file in sorted(search_dir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("def ", "async def ")):
                    continue
                name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                if name.startswith("_execute_"):
                    index.setdefault(name, py_file.as_posix())
    return index


def resolve_arc_stage_impl_path(
    source_dir: Path,
    pipeline_dir: Path,
    stage: ExtractedStage,
) -> Path | None:
    """Locate the stage implementation file for an ARC stage."""
    if not stage.entry_function:
        return None

    source_dir = source_dir.resolve()
    pipeline_dir = pipeline_dir.resolve()
    entry_fn = stage.entry_function

    impl_dir = pipeline_dir / "stage_impls"
    if impl_dir.is_dir():
        for py_file in sorted(impl_dir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if f"def {entry_fn}" in text or f"async def {entry_fn}" in text:
                return py_file

    executor = pipeline_dir / "executor.py"
    if executor.is_file():
        try:
            text = executor.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if f"def {entry_fn}" in text or f"async def {entry_fn}" in text:
            return executor

    return None


def arc_stage_impl_rel_path(
    source_dir: Path,
    pipeline_dir: Path,
    stage: ExtractedStage,
) -> str | None:
    impl = resolve_arc_stage_impl_path(source_dir, pipeline_dir, stage)
    if impl is None:
        return None
    try:
        return _norm_rel(impl, source_dir)
    except ValueError:
        return impl.as_posix()


def is_executor_module_path(file_path: str | None) -> bool:
    if not file_path:
        return False
    return Path(file_path.replace("\\", "/")).name in _EXECUTOR_BASENAMES


def _qualified_operation_names(comp: SourceComponent) -> list[str]:
    names: list[str] = []
    for op in comp.operations:
        if op.class_name:
            names.append(f"{op.class_name}.{op.name}")
        elif "." in comp.name and not op.name.startswith(comp.name):
            names.append(f"{comp.name}.{op.name}")
        else:
            names.append(op.name)
    return names


def _path_matches_impl(comp_path: str, impl_rel: str) -> bool:
    comp_norm = comp_path.replace("\\", "/").lower()
    impl_norm = impl_rel.replace("\\", "/").lower()
    if impl_norm in comp_norm or comp_norm.endswith(impl_norm):
        return True
    stem = Path(impl_rel).stem.lower()
    return stem in comp_norm and "stage_impl" in comp_norm


def collect_tools_for_stage(
    stage: ExtractedStage,
    components: list[SourceComponent],
    skills: list[SkillSpec],
    *,
    impl_rel: str | None,
) -> list[str]:
    """Collect SDK tool names most relevant to a single ARC stage."""
    tools: list[str] = []
    entry_fn = stage.entry_function or ""
    tail = entry_fn[len("_execute_") :] if entry_fn.startswith("_execute_") else entry_fn

    if impl_rel:
        for comp in components:
            if not _path_matches_impl(comp.file_path, impl_rel):
                continue
            tools.extend(_qualified_operation_names(comp))

    if entry_fn:
        for comp in components:
            for op in comp.operations:
                op_tail = op.name.replace("_", "-").lower()
                tail_kebab = tail.replace("_", "-").lower() if tail else ""
                if entry_fn in op.name or (tail_kebab and tail_kebab in op_tail):
                    qn = f"{op.class_name}.{op.name}" if op.class_name else op.name
                    if comp.name and "." not in qn and not op.class_name:
                        qn = f"{comp.name}.{op.name}"
                    tools.append(qn)

    if not tools:
        for skill in skills:
            for tool in skill.allowed_tools:
                low = tool.lower()
                if "execute_stage" in low or "execute-stage" in low:
                    tools.append(tool)
                elif entry_fn and entry_fn.replace("_", "-") in low:
                    tools.append(tool)

    seen: set[str] = set()
    ordered: list[str] = []
    for tool in tools:
        if tool not in seen:
            seen.add(tool)
            ordered.append(tool)
    return ordered


def finalize_arc_stage_tools(tools: list[str], project_name: str) -> list[str]:
    """Ensure ARC stage skills expose both pipeline and bridge execute-stage tools."""
    from ..bridge.mcp_adapter import bridge_tool_name

    pipeline_tool = "researchclaw-pipeline-execute-stage"
    bridge_tool = bridge_tool_name(project_name)
    ordered: list[str] = []
    seen: set[str] = set()
    for tool in [*tools, pipeline_tool, bridge_tool]:
        if tool and tool not in seen:
            seen.add(tool)
            ordered.append(tool)
    return ordered


def ensure_arc_stage_skills(
    graph: WorkflowGraph,
    components: list[SourceComponent],
    skills: list[SkillSpec],
    source_dir: Path,
) -> list[SkillSpec]:
    """Synthesize per-stage skills when ARC coarse grouping hides stage names."""
    pipeline_dir = resolve_pipeline_dir(source_dir)
    if pipeline_dir is None:
        return skills

    source_dir = source_dir.resolve()
    existing = {s.name for s in skills}
    existing |= {_to_kebab(s.name) for s in skills}
    impl_index = build_entry_function_impl_index(pipeline_dir)

    synthesized: list[SkillSpec] = []
    for stage in graph.stages:
        if stage.name in existing:
            continue

        impl_rel: str | None = None
        if stage.entry_function and stage.entry_function in impl_index:
            impl_path = Path(impl_index[stage.entry_function])
            try:
                impl_rel = _norm_rel(impl_path, source_dir)
            except ValueError:
                impl_rel = impl_path.as_posix()
        if impl_rel is None:
            impl_rel = arc_stage_impl_rel_path(source_dir, pipeline_dir, stage)

        tools = collect_tools_for_stage(
            stage,
            components,
            skills,
            impl_rel=impl_rel,
        )
        tools = finalize_arc_stage_tools(tools, source_dir.name)
        label = stage.label or stage.name.replace("-", " ").title()
        synthesized.append(
            SkillSpec(
                name=stage.name,
                description=f"ResearchClaw pipeline stage: {label}",
                allowed_tools=tools,
            )
        )
        logger.debug(
            "Synthesized ARC stage skill %s (%d tools, impl=%s)",
            stage.name,
            len(tools),
            impl_rel,
        )

    if not synthesized:
        return skills
    return skills + synthesized
