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

"""Dynamic cross-domain orchestration routes for SOP overview agents.

Generated at ``sop convert`` into ``ORCHESTRATION_ROUTES.md`` and injected into
overview system prompts at runtime. All titles, params, and suppression rules are
derived from parsed SDK metadata (docstrings, parameters, dependencies,
interactive-CLI heuristics) — no SDK-specific tool or operation names.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core.intent_tags import collect_intent_phrases, format_search_suggestions
from .skill_grouper import SkillSpec
from ..core.source_parser import SourceComponent, SourceOperation
from .task_guide import (
    _format_required_params_note,
    _looks_like_interactive_terminal,
    build_operation_index,
    is_entry_point,
    _resolve_operation,
    _select_task_guide_entries,
    _task_guide_rank_key,
)
from ..core.tool_dependencies import ToolOperationDeps

logger = logging.getLogger(__name__)

ORCHESTRATION_ROUTES_NAME = "ORCHESTRATION_ROUTES.md"
ORCHESTRATION_ROUTES_MAX = 40
_SINGLE_STEP_CLI_ROUTE_LABEL = "[单步 CLI]"

_TASK_GUIDE_ROW_RE = re.compile(
    r"^\|\s*(?P<intent>[^|]+?)\s*\|\s*`(?P<tool>[^`]+)`\s*\|\s*(?P<search>[^|]+?)\s*\|\s*(?P<note>[^|]+?)\s*\|\s*$"
)

_INTERACTIVE_ROUTE_NOTE = "交互式终端 — Agent 无 TTY 时按「交互式终端停损」引导用户到真实终端运行"

_PROGRAMMATIC_TERMINAL_PREFIXES = ("run_", "start_", "execute_", "invoke_", "launch_")


@dataclass(frozen=True)
class OrchestrationStep:
    agent: str
    flow: str
    param_hint: str = ""


@dataclass
class OrchestrationRoute:
    title: str
    steps: list[OrchestrationStep] = field(default_factory=list)
    note: str = ""
    tool_refs: list[str] = field(default_factory=list)


def skill_name_to_agent(skill_name: str) -> str:
    base = skill_name.removesuffix("-skill") if skill_name.endswith("-skill") else skill_name
    return f"{base}-agent"


def build_tool_to_agent_map(skills: list[SkillSpec]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for skill in skills:
        agent = skill_name_to_agent(f"{skill.name}-skill")
        for tool in skill.allowed_tools:
            mapping[tool] = agent
    return mapping


def _short_tool_label(tool: str) -> str:
    parts = tool.split("-")
    if len(parts) >= 4:
        return "-".join(parts[-3:])
    elif len(parts) >= 2:
        return "-".join(parts[-2:])
    return tool


def _title_from_operation(op: SourceOperation) -> str:
    phrases = collect_intent_phrases(op)
    if phrases:
        title = phrases[0].strip()
        if title:
            return title[:100]
    desc = (op.description or op.name).strip()
    first = re.split(r"[。\n.!]", desc, maxsplit=1)[0].strip()
    if first:
        return first[:100]
    return op.name.replace("_", " ")


def _title_from_task_guide_row(row: dict[str, str]) -> str:
    intent = (row.get("intent") or "").strip()
    if intent:
        return intent.split("；", 1)[0].strip()[:100]
    tool = row.get("tool") or ""
    return _short_tool_label(tool).replace("-", " ")


def _param_hint_from_operation(comp: SourceComponent, op: SourceOperation) -> str:
    note = _format_required_params_note(comp, op).strip()
    return note


def _param_hint_from_task_guide_note(note: str) -> str:
    text = (note or "").strip()
    match = re.search(r"必填\s*(.+?)(?:。|$)", text)
    if match:
        return match.group(1).strip()
    return ""


def _toolsearch_query_from_row(search: str, tool: str) -> str:
    first = (search or "").split(",", 1)[0].strip().strip('"')
    if first and len(first) <= 80:
        return first
    return _short_tool_label(tool).replace("-", " ")


def _parse_task_guide_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_guide = False
    for line in body.splitlines():
        if line.strip() == "## 任务指南":
            in_guide = True
            continue
        if in_guide and line.startswith("## "):
            break
        if not in_guide or not line.startswith("|"):
            continue
        if line.startswith("|--") or "用户意图" in line:
            continue
        match = _TASK_GUIDE_ROW_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "intent": match.group("intent").strip(),
                    "tool": match.group("tool").strip(),
                    "search": match.group("search").strip(),
                    "note": match.group("note").strip(),
                }
            )
    return rows


def _load_skills_from_bundle_dirs(bundle_path: Path, workspace_root: Path | None = None) -> list[SkillSpec]:
    from .bundle_skills import _bundle_skill_search_dirs

    ws = workspace_root or bundle_path.parent
    specs: list[SkillSpec] = []
    seen: set[str] = set()
    for base in _bundle_skill_search_dirs(bundle_path, ws):
        for md in sorted(base.glob("*-skill.md")):
            try:
                content = md.read_text(encoding="utf-8")
            except OSError:
                continue
            skill_name = md.stem
            if skill_name in seen:
                continue
            seen.add(skill_name)
            tools: list[str] = []
            in_tools = False
            desc = ""
            for line in content.splitlines():
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
                if line.strip() == "allowed-tools:":
                    in_tools = True
                    continue
                if in_tools:
                    if line.startswith("  - "):
                        tools.append(line[4:].strip())
                    elif line.strip() and not line.startswith((" ", "\t", "-")):
                        in_tools = False
            if not tools:
                from .bundle_skills import _parse_allowed_tools_from_frontmatter_text

                tools = _parse_allowed_tools_from_frontmatter_text(content)
            base_name = skill_name.removesuffix("-skill") if skill_name.endswith("-skill") else skill_name
            specs.append(
                SkillSpec(
                    name=base_name,
                    description=desc,
                    allowed_tools=tools,
                )
            )
    return specs


def _is_single_step_cli_route(route: OrchestrationRoute) -> bool:
    return len(route.steps) == 1 and route.note == _INTERACTIVE_ROUTE_NOTE


def _format_route_heading(route: OrchestrationRoute) -> str:
    if _is_single_step_cli_route(route):
        return f"### {_SINGLE_STEP_CLI_ROUTE_LABEL} {route.title}"
    return f"### {route.title}"


def _agents_with_single_step_cli_routes(routes: list[OrchestrationRoute]) -> set[str]:
    return {route.steps[0].agent for route in routes if _is_single_step_cli_route(route)}


def _is_programmatic_terminal_tool(
    tool_ref: str,
    components: list[SourceComponent] | None,
) -> bool:
    if not components or not tool_ref:
        return False
    index = build_operation_index(components)
    resolved = _resolve_operation(tool_ref, index)
    if resolved is None:
        return False
    comp, op = resolved
    if _looks_like_interactive_terminal(comp, op):
        return False
    if not is_entry_point(comp, op):
        return False
    return any(op.name.startswith(prefix) for prefix in _PROGRAMMATIC_TERMINAL_PREFIXES)


def _should_suppress_multi_step_route(
    route: OrchestrationRoute,
    *,
    cli_agents: set[str],
    components: list[SourceComponent] | None,
) -> bool:
    """Drop cross-domain API chains when an overlapping skill already exposes a CLI one-shot."""
    route_agents = {step.agent for step in route.steps}
    if len(route.steps) < 2 or len(route_agents) < 2:
        return False
    if not cli_agents or not route_agents & cli_agents:
        return False
    terminal_tool = route.tool_refs[-1] if route.tool_refs else ""
    return _is_programmatic_terminal_tool(terminal_tool, components)


def _task_guide_rows_from_bundle(bundle_path: Path, skills: list[SkillSpec]) -> dict[str, list[dict[str, str]]]:
    from .bundle_skills import _bundle_skill_search_dirs

    rows_by_skill: dict[str, list[dict[str, str]]] = {}
    for base in _bundle_skill_search_dirs(bundle_path, bundle_path.parent):
        for md in sorted(base.glob("*-skill.md")):
            try:
                content = md.read_text(encoding="utf-8")
            except OSError:
                continue
            skill_key = md.stem.removesuffix("-skill") if md.stem.endswith("-skill") else md.stem
            parsed = _parse_task_guide_rows(content)
            if parsed:
                rows_by_skill[skill_key] = parsed
    for skill in skills:
        rows_by_skill.setdefault(skill.name, [])
    return rows_by_skill


def _cli_orchestrator_routes(
    skills: list[SkillSpec],
    components: list[SourceComponent] | None,
    *,
    bundle_path: Path | None = None,
) -> list[OrchestrationRoute]:
    routes: list[OrchestrationRoute] = []
    if components:
        index = build_operation_index(components)
        for skill in skills:
            ranked: list[tuple[tuple[int, int, int, str], str, SourceComponent, SourceOperation]] = []
            seen: set[str] = set()
            for tool_ref in skill.allowed_tools:
                resolved = _resolve_operation(tool_ref, index)
                if resolved is None:
                    continue
                comp, op = resolved
                op_key = f"{comp.name}:{op.class_name}:{op.name}"
                if op_key in seen or not is_entry_point(comp, op):
                    continue
                if not _looks_like_interactive_terminal(comp, op):
                    continue
                seen.add(op_key)
                ranked.append((_task_guide_rank_key(comp, op, tool_ref), tool_ref, comp, op))
            ranked.sort(key=lambda item: item[0], reverse=True)
            for tool_ref, comp, op in _select_task_guide_entries(ranked, max_entries=2):
                agent = skill_name_to_agent(f"{skill.name}-skill")
                query = format_search_suggestions(op, comp_name=comp.name).split(",", 1)[0].strip()
                if not query:
                    query = op.name.replace("_", " ")
                routes.append(
                    OrchestrationRoute(
                        title=_title_from_operation(op),
                        tool_refs=[tool_ref],
                        steps=[
                            OrchestrationStep(
                                agent=agent,
                                flow=(
                                    f'``Skill`` → ``ToolSearch(query="{query}")`` → ``{_short_tool_label(tool_ref)}``'
                                ),
                                param_hint=_param_hint_from_operation(comp, op),
                            )
                        ],
                        note=_INTERACTIVE_ROUTE_NOTE,
                    )
                )
        return routes

    rows_by_skill: dict[str, list[dict[str, str]]] = {}
    if bundle_path is not None:
        rows_by_skill = _task_guide_rows_from_bundle(bundle_path, skills)

    for skill in skills:
        for row in rows_by_skill.get(skill.name, ()):
            if "交互式终端" not in row.get("note", ""):
                continue
            tool = row["tool"]
            agent = skill_name_to_agent(f"{skill.name}-skill")
            query = _toolsearch_query_from_row(row["search"], tool)
            routes.append(
                OrchestrationRoute(
                    title=_title_from_task_guide_row(row),
                    tool_refs=[tool],
                    steps=[
                        OrchestrationStep(
                            agent=agent,
                            flow=(f'``Skill`` → ``ToolSearch(query="{query}")`` → ``{_short_tool_label(tool)}``'),
                            param_hint=_param_hint_from_task_guide_note(row.get("note", "")),
                        )
                    ],
                    note=_INTERACTIVE_ROUTE_NOTE,
                )
            )
    return routes


def _cross_domain_dependency_routes(
    skills: list[SkillSpec],
    tool_deps_index: dict[str, ToolOperationDeps],
) -> list[OrchestrationRoute]:
    tool_to_agent = build_tool_to_agent_map(skills)
    routes: list[OrchestrationRoute] = []

    for tool, deps in tool_deps_index.items():
        if not deps.requires:
            continue
        chain_tools: list[str] = []
        visited: set[str] = set()
        cursor = tool
        truncated = False
        while cursor:
            if cursor in visited:
                truncated = True
                logger.warning(
                    "cross-domain dependency chain truncated at %s (cycle in tool_deps_index)",
                    cursor,
                )
                break
            visited.add(cursor)
            chain_tools.insert(0, cursor)
            req = tool_deps_index.get(cursor)
            if req is None or not req.requires:
                break
            cursor = req.requires[0]

        agents: list[str] = []
        steps: list[OrchestrationStep] = []
        for chain_tool in chain_tools:
            agent = tool_to_agent.get(chain_tool)
            if agent is None:
                break
            if agent not in agents:
                agents.append(agent)
            query = _short_tool_label(chain_tool).replace("-", " ")
            steps.append(
                OrchestrationStep(
                    agent=agent,
                    flow=(f'``Skill`` → ``ToolSearch(query="{query}")`` → ``{_short_tool_label(chain_tool)}``'),
                    param_hint="上一步工具返回值" if steps else "",
                )
            )
        if len(agents) < 2 or len(steps) < 2:
            continue

        title = f"跨域：{' → '.join(_short_tool_label(t) for t in chain_tools)}"
        routes.append(
            OrchestrationRoute(
                title=title,
                tool_refs=list(chain_tools),
                steps=steps,
                note="依赖链被截断（疑似循环引用）" if truncated else "",
            )
        )

    return routes


def _route_sort_key(route: OrchestrationRoute) -> tuple[int, int, str]:
    return (0 if _is_single_step_cli_route(route) else 1, len(route.steps), route.title.lower())


def discover_orchestration_routes(
    skills: list[SkillSpec],
    *,
    components: list[SourceComponent] | None = None,
    tool_deps_index: dict[str, ToolOperationDeps] | None = None,
    bundle_path: Path | None = None,
) -> list[OrchestrationRoute]:
    if not skills:
        return []

    cli_routes = _cli_orchestrator_routes(skills, components, bundle_path=bundle_path)
    cli_agents = _agents_with_single_step_cli_routes(cli_routes)

    routes: list[OrchestrationRoute] = list(cli_routes)

    if tool_deps_index:
        for dep_route in _cross_domain_dependency_routes(skills, tool_deps_index):
            if _should_suppress_multi_step_route(
                dep_route,
                cli_agents=cli_agents,
                components=components,
            ):
                continue
            routes.append(dep_route)

    deduped: list[OrchestrationRoute] = []
    seen_titles: set[str] = set()
    for route in routes:
        key = route.title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(route)

    deduped.sort(key=_route_sort_key)
    return deduped[:ORCHESTRATION_ROUTES_MAX]


def _group_routes_by_first_step(routes: list[OrchestrationRoute]) -> list[OrchestrationRoute]:
    if len(routes) < 2:
        return routes

    grouped: dict[tuple[str, str], list[OrchestrationRoute]] = {}
    for route in routes:
        if route.steps:
            first_step = route.steps[0]
            key = (first_step.agent, first_step.flow)
        else:
            key = ("", "")
        grouped.setdefault(key, []).append(route)

    result: list[OrchestrationRoute] = []
    for key, group in grouped.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            first_step = group[0].steps[0] if group[0].steps else None
            all_second_steps: list[tuple[str, str, str]] = []
            seen_second: set[str] = set()

            for r in group:
                if len(r.steps) >= 2:
                    second = r.steps[1]
                    second_key = f"{second.agent}:{second.flow}"
                    if second_key not in seen_second:
                        seen_second.add(second_key)
                        all_second_steps.append((second.agent, second.flow, second.param_hint))

            if first_step and all_second_steps:
                combined_title = (
                    f"跨域：{_short_tool_label(group[0].tool_refs[0])} → 多个目标（任选其一）"
                    if group[0].tool_refs
                    else "跨域：多目标路由（任选其一）"
                )
                combined_steps = [first_step]
                for agent, flow, hint in all_second_steps[:8]:
                    combined_steps.append(
                        OrchestrationStep(
                            agent=agent,
                            flow=f"【任选】{flow}",
                            param_hint=hint,
                        )
                    )
                if len(all_second_steps) > 8:
                    combined_steps.append(
                        OrchestrationStep(
                            agent="",
                            flow=f"【任选】（另有 {len(all_second_steps) - 8} 个目标，见完整路由表）",
                            param_hint="",
                        )
                    )

                combined_tool_refs: list[str] = []
                for r in group:
                    combined_tool_refs.extend(r.tool_refs)

                result.append(
                    OrchestrationRoute(
                        title=combined_title,
                        steps=combined_steps,
                        tool_refs=combined_tool_refs,
                        note="以下第二步为互斥选项，根据任务选择其中一个目标",
                    )
                )
            else:
                result.extend(group)

    return result


def generate_orchestration_routes_markdown(
    skills: list[SkillSpec],
    *,
    components: list[SourceComponent] | None = None,
    tool_deps_index: dict[str, ToolOperationDeps] | None = None,
    bundle_path: Path | None = None,
) -> str:
    routes = discover_orchestration_routes(
        skills,
        components=components,
        tool_deps_index=tool_deps_index,
        bundle_path=bundle_path,
    )

    cli_routes = [r for r in routes if _is_single_step_cli_route(r)]
    multi_step_routes = [r for r in routes if not _is_single_step_cli_route(r)]
    grouped_multi_step = _group_routes_by_first_step(multi_step_routes)
    routes = cli_routes + grouped_multi_step

    lines = [
        "# 跨域编排路由",
        "",
        "由 `sop convert` 根据 Skill 任务指南与工具 ``x-sop-dependencies`` 自动生成。",
        "Overview **按表委派**对应域 Agent；子代理 prompt 须包含上一步产出与参数提示。",
        "",
    ]
    if not routes:
        lines.extend(
            [
                "_（无跨域路由；单域任务直接委派对应 ``*-agent``）_",
                "",
            ]
        )
        return "\n".join(lines)

    for route in routes:
        lines.append(_format_route_heading(route))
        lines.append("")
        lines.append("| 步骤 | 委派 | 子代理内流程 | 参数提示 |")
        lines.append("|------|------|--------------|----------|")
        for idx, step in enumerate(route.steps, start=1):
            param = step.param_hint or "—"
            agent_display = f"``@{step.agent}``" if step.agent else "—"
            lines.append(f"| {idx} | {agent_display} | {step.flow} | {param} |")
        if route.note:
            lines.append("")
            lines.append(f"_{route.note}_")
        lines.append("")

    lines.extend(
        [
            "## Overview 委派要点",
            "",
            '1. **逐步委派** — 每步 ``Agent(subagent_type="<domain>-agent", prompt="...")``；'
            "prompt 写明 Skill、ToolSearch 查询、参数及上一步 JSON/路径",
            "2. **禁止** overview 自己调用域 Skill / ToolSearch / SDK 工具",
            f"3. **单步交互式 CLI 路由**（标题前缀 ``{_SINGLE_STEP_CLI_ROUTE_LABEL}``，备注「交互式终端」）"
            "优先于跨域 programmatic API 链",
            "4. bundle 已注册 composite macro 工具时，优先一次调用（见 skill 任务指南）",
            "",
        ]
    )
    return "\n".join(lines)


def write_orchestration_routes(
    bundle_dir: Path,
    skills: list[SkillSpec],
    *,
    components: list[SourceComponent] | None = None,
    tool_deps_index: dict[str, ToolOperationDeps] | None = None,
) -> Path:
    bundle_dir = bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    content = generate_orchestration_routes_markdown(
        skills,
        components=components,
        tool_deps_index=tool_deps_index,
    )
    path = bundle_dir / ORCHESTRATION_ROUTES_NAME
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def format_orchestration_routes_block(
    bundle_path: Path | str | None,
    *,
    workspace_root: Path | str | None = None,
    inline_content: bool = False,
    max_inline_chars: int = 8000,
) -> str:
    if not bundle_path:
        return ""
    bundle = Path(bundle_path).resolve()
    path = bundle / ORCHESTRATION_ROUTES_NAME

    if not inline_content:
        if not path.is_file():
            return f"""\
### 跨域编排（sop convert 生成）

- bundle 内文件：``{path}``（尚未生成）
- 跨工具链顺序见各工具 JSON schema 的 ``x-sop-dependencies``
"""
        return f"""\
### 跨域编排（sop convert 生成）

- bundle 内文件：``{path}``
- 跨域委派前 **Read** ``{ORCHESTRATION_ROUTES_NAME}`` 查路由表与「参数提示」（正文不内嵌全文，避免挤占上下文）
- 委派子 Agent 时 prompt **必须**写入上一步工具返回的路径/对象与表中「参数提示」
- 表中存在**单步交互式 CLI** 路由时，**禁止**改用跨域 programmatic API 链完成同一类用户任务
"""

    header = f"""\
### 跨域编排（sop convert 生成）

- bundle 内文件：``{path}`` — Overview 路由时**优先使用下方路由表**
- 委派子 Agent 时 prompt **必须**写入上一步工具返回的路径/对象与表中「参数提示」
- 表中存在**单步交互式 CLI** 路由时，**禁止**改用跨域 programmatic API 链完成同一类用户任务
"""

    body = ""
    if path.is_file():
        try:
            body = path.read_text(encoding="utf-8").strip()
        except OSError:
            body = ""
    else:
        skills = _load_skills_from_bundle_dirs(bundle, Path(workspace_root) if workspace_root else None)
        if skills:
            body = generate_orchestration_routes_markdown(
                skills,
                bundle_path=bundle,
            ).strip()

    if not body:
        return header + "\n- 跨工具链顺序见各工具 JSON schema 的 ``x-sop-dependencies``\n"

    if len(body) > max_inline_chars:
        body = body[: max_inline_chars - 40].rstrip() + "\n\n…（全文见 Read ORCHESTRATION_ROUTES.md）"

    return f"{header}\n{body}"
