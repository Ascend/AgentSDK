# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Generate a compact SDK module map during ``sop convert``.

Gives overview / domain agents a pre-built index of which module handles which
capability, so they do not need to grep the entire source tree for routing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from extensions.sop_converter.runtime.skill_grouper import GroupStrategy, SkillSpec
from extensions.sop_converter.core.source_parser import SourceComponent, SourceOperation
from extensions.sop_converter.runtime.task_guide import _resolve_operation, build_operation_index, is_entry_point

_IO_OVERVIEW_MARKER = "<!-- sop-convert: strategy=io_relation -->"
_IO_TYPE_DESC_RE = re.compile(r"Operations sharing types:\s*(.+)", re.I)


def _module_key(comp: SourceComponent) -> str:
    path = comp.file_path.replace("\\", "/")
    parts = [p for p in path.split("/") if p and not p.endswith(".py")]
    if parts:
        return "/".join(parts[: min(3, len(parts))])
    return comp.name.split(".")[0] if comp.name else "root"


def _top_level_package(comp: SourceComponent) -> str:
    name = comp.name.split(".")[0] if comp.name else ""
    if name:
        return name
    path = comp.file_path.replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else "sdk"


def _entry_ops(comp: SourceComponent, *, limit: int = 4) -> list[SourceOperation]:
    ranked: list[tuple[int, SourceOperation]] = []
    for op in comp.operations:
        if not is_entry_point(comp, op):
            continue
        score = 1 if op.description else 0
        if op.name.startswith(("run_", "start_", "open_", "launch_", "init", "create_")):
            score += 2
        ranked.append((score, op))
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return [op for _score, op in ranked[:limit]]


def _skill_base_name(skill_name: str) -> str:
    """Strip ``-skill`` / ``_merged-skill`` suffix to get the skill base name.

    Examples:
        ``harness_merged-skill`` -> ``harness``
        ``core_merged-skill`` -> ``core``
        ``foundation-skill`` -> ``foundation``
        ``harness-skill`` -> ``harness``
    """
    name = skill_name.lower()
    for suffix in ("_merged-skill", "-skill", "_skill"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # ``harness_merged`` -> ``harness`` (drop _merged mid-fix as well)
    if name.endswith("_merged"):
        name = name[: -len("_merged")]
    return name


def _skill_for_component(comp_name: str, skills: list[SkillSpec]) -> str | None:
    """Match a component path against skill names by path segment.

    A skill ``harness_merged-skill`` matches any component whose path contains
    the segment ``harness`` (case-insensitive).

    Falls back to substring match for skill names that don't reduce to a clean
    base (e.g. ``agent_evolving_merged-skill`` -> base ``agent_evolving``).
    """
    comp_lower = comp_name.lower()
    # Split on path separators, dots and hyphens so dotted module names
    # (``openjiuwen.agent_teams``) and hyphenated segments are handled too.
    comp_segments = {seg for seg in re.split(r"[/_.-]", comp_lower) if seg}
    for skill in skills:
        base = _skill_base_name(skill.name)
        if not base:
            continue
        # Normalize hyphens to underscores for cross-notation matching:
        # skill ``agent-teams`` must match component ``agent_teams``.
        base_normalized = base.replace("-", "_")
        # Match by path segment (most reliable for ``harness``, ``core``, etc.)
        if base in comp_segments or base_normalized in comp_segments:
            return f"{skill.name}-agent"
        # Fallback: substring match on the reduced base name
        if base in comp_lower or base_normalized in comp_lower:
            return f"{skill.name}-agent"
    return None


def _extract_io_type_anchors(skill: SkillSpec) -> list[str]:
    """Parse dominant parameter types from an IO_RELATION skill."""
    desc = (skill.description or "").strip()
    match = _IO_TYPE_DESC_RE.match(desc)
    if match:
        return [t.strip() for t in match.group(1).split(",") if t.strip()]
    if skill.name.startswith("io_group_"):
        slug = skill.name.removeprefix("io_group_")
        if slug.endswith("_utility"):
            return ["utility"]
        return [part for part in slug.split("_") if part]
    return []


def _sample_modules_for_skill(
    skill: SkillSpec,
    index: dict[str, tuple[SourceComponent, SourceOperation]],
    *,
    limit: int = 4,
) -> list[str]:
    seen: set[str] = set()
    modules: list[str] = []
    for tool in skill.allowed_tools:
        resolved = _resolve_operation(tool, index)
        if resolved is None:
            continue
        comp, _op = resolved
        module = _module_key(comp)
        if module in seen:
            continue
        seen.add(module)
        modules.append(module)
        if len(modules) >= limit:
            break
    return modules


def _is_io_overview_content(body: str) -> bool:
    head = body[:600]
    return _IO_OVERVIEW_MARKER in head or "SDK 类型路由总览 (IO 分组)" in head


def generate_io_sdk_overview_markdown(
    components: list[SourceComponent],
    *,
    skills: list[SkillSpec] | None = None,
    sdk_source_dir: Path | str | None = None,
    max_tools: int = 80,
    max_entry_apis: int = 40,
) -> str:
    """Build IO-grouping ``SDK_OVERVIEW.md`` — route by tool/type, not module path."""
    skills = skills or []
    index = build_operation_index(components)

    lines = [
        _IO_OVERVIEW_MARKER,
        "# SDK 类型路由总览 (IO 分组)",
        "",
        "由 `sop convert --strategy io` 自动生成。Overview / 域 Agent **优先读本文件**做路由，"
        "禁止对 SDK 源码树做无边界 Glob/Grep。",
        "",
        "> **重要**：本 bundle 使用 **IO 参数类型分组**。子 Agent 按共享参数类型聚类，"
        "**不按源码路径划分**。同一模块的不同方法可能落在不同 `@io_group_*-agent`。"
        "**禁止** 按目录/包名臆测 Agent；请使用下方 **工具 → Agent** 或 **入口 API** 表。",
        "",
    ]
    if sdk_source_dir:
        lines.append(f"- **SDK 源码根**: `{Path(sdk_source_dir).resolve()}`")
        lines.append("")

    lines.extend(
        [
            "## 域 Agent 速查（按类型锚点）",
            "",
            "| 域 Agent | 类型锚点 | 工具数 | 覆盖模块（示例） |",
            "|----------|----------|--------|------------------|",
        ]
    )
    for skill in skills:
        anchors = ", ".join(f"`{t}`" for t in _extract_io_type_anchors(skill)) or "—"
        modules = ", ".join(f"`{m}`" for m in _sample_modules_for_skill(skill, index)) or "—"
        lines.append(f"| `@{skill.name}-agent` | {anchors} | {len(skill.allowed_tools)} | {modules} |")
    lines.append("")

    entry_rows: list[tuple[str, str, str, str]] = []
    for skill in skills:
        agent = f"@{skill.name}-agent"
        for tool in skill.allowed_tools:
            resolved = _resolve_operation(tool, index)
            if resolved is None:
                continue
            comp, op = resolved
            if not is_entry_point(comp, op):
                continue
            summary = (op.description or op.name).split("\n", 1)[0].strip()
            if len(summary) > 90:
                summary = summary[:87] + "..."
            entry_rows.append((tool, summary, agent, _module_key(comp)))

    if entry_rows:
        lines.extend(["## 入口 API → Agent", ""])
        lines.append("| 工具 / API | 说明 | 模块 | 路由 Agent |")
        lines.append("|------------|------|------|------------|")
        for tool, summary, agent, module in entry_rows[:max_entry_apis]:
            lines.append(f"| `{tool}` | {summary} | `{module}` | `{agent}` |")
        if len(entry_rows) > max_entry_apis:
            lines.append("")
            lines.append(f"_（另有 {len(entry_rows) - max_entry_apis} 个入口 API，见各 Skill 任务指南）_")
        lines.append("")

    tool_rows: list[tuple[str, str]] = []
    for skill in skills:
        agent = f"@{skill.name}-agent"
        for tool in skill.allowed_tools:
            tool_rows.append((tool, agent))
    tool_rows.sort(key=lambda row: row[0].lower())

    if tool_rows:
        lines.extend(["## 工具 → Agent", ""])
        lines.append("| 工具 | 路由 Agent |")
        lines.append("|------|------------|")
        for tool, agent in tool_rows[:max_tools]:
            lines.append(f"| `{tool}` | `{agent}` |")
        if len(tool_rows) > max_tools:
            lines.append("")
            lines.append(f"_（另有 {len(tool_rows) - max_tools} 个工具，见各 Skill 的 allowed-tools 或 ToolSearch）_")
        lines.append("")

    lines.extend(
        [
            "## 使用说明（Overview — IO 分组）",
            "",
            "1. 用户提到具体 API/方法名 → 查「入口 API」或「工具 → Agent」，直接 `Agent(subagent_type=...)` 委派",
            "2. 用户意图模糊 → 对照「域 Agent 速查」的**类型锚点**与**覆盖模块**，选最相关的 `@io_group_*-agent`",
            "3. 多个 io_group 都合理时，**向用户确认**后再委派",
            "4. **禁止** 假设 `harness/`、`core/` 等路径对应固定 Agent；IO 分组下路径与 Agent **无稳定 1:1 关系**",
            "5. 跨工具编排见 `ORCHESTRATION_ROUTES.md` 与各工具 schema 的 `x-sop-dependencies`",
            "",
        ]
    )
    return "\n".join(lines)


def generate_sdk_overview_markdown(
    components: list[SourceComponent],
    *,
    skills: list[SkillSpec] | None = None,
    sdk_source_dir: Path | str | None = None,
    max_modules: int = 40,
    group_strategy: GroupStrategy | None = None,
) -> str:
    """Build ``SDK_OVERVIEW.md`` content from parsed components."""
    if group_strategy == GroupStrategy.IO_RELATION:
        return generate_io_sdk_overview_markdown(
            components,
            skills=skills,
            sdk_source_dir=sdk_source_dir,
        )
    skills = skills or []
    by_package: dict[str, list[SourceComponent]] = defaultdict(list)
    for comp in components:
        by_package[_top_level_package(comp)].append(comp)

    lines = [
        "# SDK 模块总览",
        "",
        "由 `sop convert` 自动生成。Overview / 域 Agent **优先读本文件**做路由，禁止对 SDK 源码树做无边界 Glob/Grep。",
        "",
    ]
    if sdk_source_dir:
        lines.append(f"- **SDK 源码根**: `{Path(sdk_source_dir).resolve()}`")
        lines.append("")

    lines.extend(
        [
            "## 域 Agent 速查",
            "",
            "| 域 Agent | Skill | 说明 |",
            "|----------|-------|------|",
        ]
    )
    for skill in skills:
        sample = ", ".join(f"`{t}`" for t in skill.allowed_tools[:2])
        if len(skill.allowed_tools) > 2:
            sample += ", …"
        desc = (skill.description or "").strip()
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"| `@{skill.name}-agent` | `{skill.name}-skill` | {desc or sample or '—'} |")
    lines.append("")

    lines.extend(["## 模块 → 能力", ""])
    module_rows: list[tuple[str, str, SourceComponent]] = []
    for pkg, comps in sorted(by_package.items()):
        for comp in comps:
            module_rows.append((pkg, _module_key(comp), comp))
    module_rows.sort(key=lambda row: (row[0], row[1]))

    shown = 0
    current_pkg = ""
    for pkg, module_path, comp in module_rows:
        if shown >= max_modules:
            lines.append("_（更多模块已省略，详见各 Skill 任务指南）_")
            break
        if pkg != current_pkg:
            lines.append(f"### `{pkg}`")
            lines.append("")
            current_pkg = pkg

        agent = _skill_for_component(comp.name, skills)
        agent_cell = f"`@{agent}`" if agent else "—"
        desc = (comp.description or comp.name).strip()
        if len(desc) > 100:
            desc = desc[:97] + "..."

        lines.append(f"#### `{module_path}`")
        lines.append("")
        lines.append(f"- **说明**: {desc}")
        lines.append(f"- **路由 Agent**: {agent_cell}")
        lines.append(f"- **源文件**: `{comp.file_path}`")

        entries = _entry_ops(comp)
        if entries:
            lines.append("- **常见入口 API**:")
            for op in entries:
                summary = (op.description or op.name).split("\n", 1)[0].strip()
                if len(summary) > 90:
                    summary = summary[:87] + "..."
                lines.append(f"  - `{op.name}` — {summary}")
        lines.append("")
        shown += 1

    lines.extend(
        [
            "## 使用说明（Overview）",
            "",
            "1. 用户意图模糊时，先在本表找到最相关的 **域 Agent** 或 **入口 API**",
            "2. 若多个候选仍无法区分，**向用户确认**后再 `Agent(subagent_type=...)` 委派",
            "3. 子 Agent 内按 Skill「任务指南」→ ToolSearch → SDK 工具；"
            "跨工具编排见各工具 schema 的 `x-sop-dependencies`",
            "",
        ]
    )
    return "\n".join(lines)


def write_sdk_overview(
    bundle_dir: Path,
    components: list[SourceComponent],
    *,
    skills: list[SkillSpec] | None = None,
    sdk_source_dir: Path | str | None = None,
    group_strategy: GroupStrategy | None = None,
) -> Path:
    bundle_dir = bundle_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    content = generate_sdk_overview_markdown(
        components,
        skills=skills,
        sdk_source_dir=sdk_source_dir,
        group_strategy=group_strategy,
    )
    path = bundle_dir / "SDK_OVERVIEW.md"
    path.write_text(content, encoding="utf-8")
    return path


def format_sdk_overview_block(
    bundle_path: Path | str | None,
    *,
    inline_content: bool = False,
    max_inline_chars: int = 10000,
) -> str:
    """Prompt block for overview agents — points at ``SDK_OVERVIEW.md`` (opt-in embed)."""
    if not bundle_path:
        return ""
    path = Path(bundle_path).resolve() / "SDK_OVERVIEW.md"
    if not path.is_file():
        return ""

    try:
        body = path.read_text(encoding="utf-8").strip()
    except OSError:
        body = ""

    if _is_io_overview_content(body):
        if not inline_content:
            return f"""\
## SDK 类型路由总览（IO 分组，sop convert 生成）

- bundle 内文件：``{path}``
- 需要查「工具 → Agent」或「入口 API」时 **Read** ``{path.name}``（正文不内嵌全文，避免挤占上下文）
- **IO 分组**：按**工具名 / 类型锚点**路由，**禁止**按源码路径臆测 Agent
- 用户意图不明确且多个 ``@io_group_*-agent`` 都合理时，**向用户确认**后再委派
- 跨工具编排见 ``ORCHESTRATION_ROUTES.md`` 与各工具 schema 的 ``x-sop-dependencies``
"""
        header = f"""\
## SDK 类型路由总览（IO 分组，sop convert 生成）

- bundle 内文件：``{path}`` — 路由时**优先使用下方摘要**，必要时再 Read 全文
- **IO 分组**：按**工具名 / 类型锚点**路由，**禁止**按源码路径臆测 Agent
- 用户意图不明确且多个 ``@io_group_*-agent`` 都合理时，**向用户确认**后再委派
- 跨工具编排见 ``ORCHESTRATION_ROUTES.md`` 与各工具 schema 的 ``x-sop-dependencies``
"""
    else:
        if not inline_content:
            return f"""\
## SDK 模块总览（sop convert 生成）

- bundle 内文件：``{path}``
- 需要按模块/域 Agent/入口 API 路由时 **Read** ``{path.name}``（正文不内嵌全文，避免挤占上下文）
- **禁止**无目标广搜 SDK 源码
- 用户意图不明确且总览表中有 2+ 合理候选时，**向用户确认**选项后再委派
- 工具链顺序（先 A 产出 B，再调 C）见各工具 JSON schema 的 ``x-sop-dependencies``
"""
        header = f"""\
## SDK 模块总览（sop convert 生成）

- bundle 内文件：``{path}`` — 路由时**优先使用下方摘要**，必要时再 Read 全文
- 用户意图不明确且总览表中有 2+ 合理候选时，**向用户确认**选项后再委派
- 工具链顺序（先 A 产出 B，再调 C）见各工具 JSON schema 的 ``x-sop-dependencies``
"""

    if not body:
        return header

    if len(body) > max_inline_chars:
        body = body[: max_inline_chars - 40].rstrip() + "\n\n…（全文见 Read SDK_OVERVIEW.md）"

    return f"{header}\n{body}"
