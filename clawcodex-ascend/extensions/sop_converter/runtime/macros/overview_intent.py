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

"""Overview-agent macro intent routing (phrase → domain agent → macro tool).

Handwritten / bundle macros must not be broadcast onto every domain Skill.
Each macro is owned by exactly one skill (see ``assign_macros_to_owner_skills``);
overview gets a compact intent table so it can delegate without source search.
Domain agent capability listings intentionally omit macro tool names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _agent_base(agent_name: str) -> str:
    name = str(agent_name or "").strip()
    if name.endswith("-agent"):
        return name[: -len("-agent")]
    return name


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _name_match_score(agent_name: str, tool_ref: str) -> int:
    """Longer agent-base substring match inside a tool id scores higher."""
    base = _normalize_token(_agent_base(agent_name))
    tool = _normalize_token(tool_ref)
    if not base or not tool or base not in tool:
        return 0
    return len(base)


def resolve_macro_delegate_agent(
    *,
    target_tool: str,
    covered_tools: Sequence[str] = (),
    agent_names: Sequence[str] = (),
    agent_tools: Mapping[str, Iterable[str]] | None = None,
) -> str | None:
    """Pick the best domain agent for a macro.

    Preference order (scored):
    1. Agent base name appears in a ``covered_tools`` ref (longest wins)
    2. Agent lists a covered atomic tool
    3. Agent lists the macro ``target_tool``
    """
    names = [str(n).strip() for n in agent_names if str(n).strip()]
    if not names:
        return None

    tools_index: dict[str, set[str]] = {}
    if agent_tools:
        for name, tools in agent_tools.items():
            tools_index[str(name)] = {str(t).strip() for t in tools if str(t).strip()}

    covered = [str(t).strip() for t in covered_tools if str(t).strip()]
    target = str(target_tool or "").strip()
    best_name: str | None = None
    best_score = -1

    for name in names:
        score = 0
        owned = tools_index.get(name, set())
        for ct in covered:
            score = max(score, _name_match_score(name, ct) * 10)
            if ct in owned:
                score += 100
        if target:
            score = max(score, _name_match_score(name, target) * 5)
            if target in owned:
                score += 10
        if score > best_score:
            best_score = score
            best_name = name

    if best_score <= 0:
        # Fall back: first agent that owns the macro tool, else None.
        if target:
            for name in names:
                if target in tools_index.get(name, set()):
                    return name
        return None
    return best_name


def _skill_from_agent_name(skills: Sequence[Any], agent_name: str) -> Any | None:
    base = _agent_base(agent_name)
    for skill in skills:
        name = str(getattr(skill, "name", "") or "").strip()
        if name == base or name == f"{base}-skill" or name.removesuffix("-skill") == base:
            return skill
    return None


def pick_owner_skill(
    skills: Sequence[Any],
    *,
    target_tool: str,
    covered_tools: Sequence[str] = (),
) -> Any | None:
    """Return the single SkillSpec that should own ``target_tool``."""
    skills = [s for s in skills if getattr(s, "name", None)]
    if not skills:
        return None

    agent_names = []
    agent_tools: dict[str, set[str]] = {}
    for skill in skills:
        base = str(skill.name).removesuffix("-skill")
        agent = f"{base}-agent"
        agent_names.append(agent)
        agent_tools[agent] = {str(t).strip() for t in (getattr(skill, "allowed_tools", None) or []) if str(t).strip()}

    agent = resolve_macro_delegate_agent(
        target_tool=target_tool,
        covered_tools=covered_tools,
        agent_names=agent_names,
        agent_tools=agent_tools,
    )
    if agent:
        owned = _skill_from_agent_name(skills, agent)
        if owned is not None:
            return owned

    # Fallback: skill that already exposes the most covered atomic tools.
    covered = [str(t).strip() for t in covered_tools if str(t).strip()]
    best = None
    best_n = -1
    for skill in skills:
        allowed = {str(t).strip() for t in (skill.allowed_tools or []) if str(t).strip()}
        n = sum(1 for ct in covered if ct in allowed)
        if n > best_n:
            best, best_n = skill, n
    if best_n > 0:
        return best
    return None


def assign_macros_to_owner_skills(
    skills: Sequence[Any],
    macros: Sequence[Any],
) -> dict[str, str]:
    """Attach each macro tool to exactly one owning skill.

    Returns ``{macro_tool: skill_name}``. Macros with no resolvable owner are
    skipped (never broadcast onto every skill).
    """
    ownership: dict[str, str] = {}
    for macro in macros or []:
        route = getattr(macro, "routing", None)
        target = str(
            (getattr(route, "target_tool", None) if route is not None else None) or getattr(macro, "name", "") or ""
        ).strip()
        if not target:
            continue
        covered = []
        if route is not None:
            covered = [str(t).strip() for t in (getattr(route, "covered_tools", None) or []) if str(t).strip()]
        owner = pick_owner_skill(
            skills,
            target_tool=target,
            covered_tools=covered,
        )
        if owner is None:
            continue
        allowed = getattr(owner, "allowed_tools", None)
        if not isinstance(allowed, list):
            continue
        if target not in allowed:  # pylint: disable=unsupported-membership-test
            allowed.insert(0, target)
        ownership[target] = str(owner.name)
    return ownership


def list_bundle_domain_agents(bundle_path: Path | str | None) -> list[str]:
    """Return ``*-agent`` names from ``<bundle>/.claude/agents/``, excluding overview."""
    if not bundle_path:
        return []
    agents_dir = Path(bundle_path) / ".claude" / "agents"
    if not agents_dir.is_dir():
        return []
    names: list[str] = []
    for path in sorted(agents_dir.glob("*-agent.md")):
        name = path.stem
        if "overview" in name:
            continue
        names.append(name)
    return names


def _agent_tools_from_component_agents(
    component_agents: Sequence[Any] | None,
) -> tuple[list[str], dict[str, set[str]]]:
    names: list[str] = []
    tools: dict[str, set[str]] = {}
    if not component_agents:
        return names, tools
    for agent in component_agents:
        name = str(getattr(agent, "name", "") or "").strip()
        if not name:
            continue
        names.append(name)
        caps = getattr(agent, "capabilities", None) or []
        tools[name] = {str(t).strip() for t in caps if str(t).strip()}
    return names, tools


def _macro_description(macros_dir: Path, target_tool: str) -> str:
    path = macros_dir / f"{target_tool}.yaml"
    if not path.is_file():
        path = macros_dir / f"{target_tool}.yml"
    if not path.is_file():
        return ""
    try:
        import yaml
    except ImportError:
        return ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return ""
    except yaml.YAMLError:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("description") or "").strip()


def format_overview_macro_intent_block(
    bundle_path: Path | str | None,
    *,
    component_agents: Sequence[Any] | None = None,
) -> str:
    """Build the overview macro-tool-intent routing table, or ``\"\"`` if none."""
    if not bundle_path:
        return ""

    bundle = Path(bundle_path).resolve()
    macros_dir = bundle / ".clawcodex" / "macros"
    if not macros_dir.is_dir():
        return ""

    try:
        from extensions.sop_converter.bundle_context import load_bundle_macro_routes
    except ImportError:
        return ""

    try:
        routes = load_bundle_macro_routes(bundle)
    except OSError:
        return ""

    enabled = [r for r in routes if getattr(r, "enabled", True) and str(getattr(r, "target_tool", "") or "").strip()]
    if not enabled:
        return ""

    names, tools = _agent_tools_from_component_agents(component_agents)
    if not names:
        names = list_bundle_domain_agents(bundle)

    # Highest priority first for stable, useful ordering.
    enabled.sort(
        key=lambda r: (
            -int(getattr(r, "priority", 100)),
            str(getattr(r, "target_tool", "")),
        )
    )

    rows: list[str] = []
    seen_targets: set[str] = set()
    for route in enabled:
        target = str(route.target_tool).strip()
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)

        covered = [str(t).strip() for t in (route.covered_tools or []) if str(t).strip()]
        agent = resolve_macro_delegate_agent(
            target_tool=target,
            covered_tools=covered,
            agent_names=names,
            agent_tools=tools,
        )
        if not agent:
            # Still emit the row with a generic hint so overview knows the macro exists.
            agent = "（任选含该宏的域 Agent）"

        phrases = [str(p).strip() for p in (route.phrases or []) if str(p).strip()]
        intent = "；".join(phrases[:6]) if phrases else target
        keywords = [str(k).strip() for k in (route.keywords or []) if str(k).strip()]
        if keywords and not phrases:
            intent = "；".join(keywords[:6])

        desc = _macro_description(macros_dir, target)
        note = desc[:80] if desc else f"宏 `{target}`"
        skill = agent.removesuffix("-agent") + "-skill" if agent.endswith("-agent") else ""
        prompt_hint = (
            f"Skill({skill}) → ToolSearch(select:{target}) → 调用宏；传入用户给出的路径/参数"
            if skill
            else f"ToolSearch(select:{target}) → 调用宏"
        )
        rows.append(f"| {intent} | `{target}` | `@{agent}` | {prompt_hint}；{note} |")

    if not rows:
        return ""

    lines = [
        "## 宏工具意图（overview 路由优先）",
        "",
        "用户意图命中下表（含「手写宏 / 处理文本|图像|多模态数据」等）时：",
        "",
        '1. **立即** ``Agent(subagent_type="<域 Agent>", prompt="...")`` 委派',
        "2. 子代理 prompt **必须**写明：宏工具名、"
        '``Skill → ToolSearch(query="select:<宏名>") → 调用宏``、用户给出的 input/output 路径',
        "3. **禁止** overview 自己 Skill / ToolSearch / 调宏",
        "4. **禁止** Explore / general-purpose / Grep / Glob / Bash 搜索 SDK 或 `.clawcodex/macros` 源码",
        "",
        "| 用户意图（示例） | 宏工具 | 委派 Agent | 子代理 prompt 要点 |",
        "|----------------|--------|------------|-------------------|",
        *rows,
        "",
    ]
    return "\n".join(lines)
