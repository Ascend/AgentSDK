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

"""Load SOP bundle skills into the runtime skill registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BundleSkillLoadResult:
    skill_names: list[str]
    skill_dirs: list[Path]
    tool_names: list[str]


def resolve_bundle_skill_workspace(bundle_path: Path, workspace_root: Path) -> Path:
    """Find the project root that owns flat SOP skills for *bundle_path*."""
    bundle = bundle_path.resolve()
    ws = workspace_root.resolve()

    if _resolve_project_skills_dir(bundle, ws) is not None:
        return ws

    current = bundle.parent
    while True:
        if _resolve_project_skills_dir(bundle, current) is not None:
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    current = ws
    while True:
        if _resolve_project_skills_dir(bundle, current) is not None:
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return ws


def _bundle_id_names(bundle_path: Path) -> list[str]:
    names = [bundle_path.name]
    try:
        from extensions.sop_converter.bundle_manifest import read_bundle_manifest

        manifest = read_bundle_manifest(bundle_path)
        if manifest is not None and manifest.bundle_id:
            bundle_id = manifest.bundle_id.strip()
            if bundle_id and bundle_id not in names:
                names.append(bundle_id)
    except ImportError:
        pass
    return names


def _compact_bundle_name(name: str) -> str:
    """Normalize a versioned bundle id (``<prefix>_v<version>``) to its
    un-versioned form (``<prefix>_<version>``) so project ``skills/`` dirs
    written without the version still match.  SDK-agnostic: applies to any
    bundle id, not a specific SDK name.
    """
    marker = "_v"
    idx = name.find(marker)
    if idx > 0 and name[idx + len(marker) : idx + len(marker) + 1].isdigit():
        return name[:idx] + "_" + name[idx + len(marker) :]
    return name


def _resolve_project_skills_dir(bundle_path: Path, workspace_root: Path) -> Path | None:
    """Locate ``skills/<bundle>/`` even when folder name differs from bundle dir."""
    ws = workspace_root.resolve()
    for name in _bundle_id_names(bundle_path):
        candidate = ws / "skills" / name
        if candidate.is_dir():
            return candidate

    skills_root = ws / "skills"
    if not skills_root.is_dir():
        return None

    flat_dirs = [d for d in skills_root.iterdir() if d.is_dir() and any(d.glob("*-skill.md"))]
    if not flat_dirs:
        return None

    targets = {_compact_bundle_name(n).lower() for n in _bundle_id_names(bundle_path)}
    hits = [d for d in flat_dirs if _compact_bundle_name(d.name).lower() in targets]
    if len(hits) == 1:
        return hits[0]

    prefix = _bundle_id_names(bundle_path)[0].split("_")[0]
    prefixed = [d for d in flat_dirs if d.name.startswith(prefix)]
    if len(prefixed) == 1:
        return prefixed[0]

    return None


def _bundle_skill_search_dirs(bundle_path: Path, workspace_root: Path) -> list[Path]:
    """Directories that may contain loadable skill markdown for a POS bundle."""
    bundle = bundle_path.resolve()
    ws = resolve_bundle_skill_workspace(bundle, workspace_root)
    dirs: list[Path] = [
        bundle / ".atomcode" / "skills",
        bundle / ".clawcodex" / "skills",
    ]
    project_skills = _resolve_project_skills_dir(bundle, ws)
    if project_skills is not None:
        dirs.append(project_skills)
    return [d for d in dirs if d.is_dir()]


def _parse_allowed_tools_from_frontmatter_text(content: str) -> list[str]:
    """Line-wise fallback when YAML frontmatter parsing yields no allowed-tools."""
    lines = content.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return []

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return []

    tools: list[str] = []
    in_list = False
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if stripped.startswith("allowed-tools:"):
            in_list = True
            continue
        if not in_list:
            continue
        if line.startswith("  - "):
            tool = line[4:].strip()
            if tool:
                tools.append(tool)
            continue
        if stripped.startswith("- "):
            tool = stripped[2:].strip()
            if tool:
                tools.append(tool)
            continue
        if stripped and not line.startswith((" ", "\t")):
            in_list = False
    return tools


def _load_flat_skill_markdown(path: Path) -> None:
    """Register a single flat ``*-skill.md`` (frontmatter name + body)."""
    from ..adapters import DEFAULTS
    from clawcodex_ext.skills.loader import (
        create_skill_command,
        parse_skill_frontmatter_fields,
        _dynamic_skills,
    )

    try:
        content = path.read_text(encoding="utf-8")
        parsed = DEFAULTS.frontmatter_parser(content)
        frontmatter = parsed.frontmatter
        body = parsed.body
    except (OSError, UnicodeDecodeError, Exception) as exc:
        logger.debug("skip flat skill %s: %s", path, exc)
        return

    skill_name = str(frontmatter.get("name") or path.stem)
    fields = parse_skill_frontmatter_fields(frontmatter, body, skill_name)
    allowed_tools = list(fields["allowed_tools"])
    if not allowed_tools:
        allowed_tools = _parse_allowed_tools_from_frontmatter_text(content)
    skill = create_skill_command(
        skill_name=skill_name,
        display_name=fields["display_name"],
        description=fields["description"],
        has_user_specified_description=fields["has_user_specified_description"],
        markdown_content=body,
        allowed_tools=allowed_tools,
        argument_hint=fields["argument_hint"],
        argument_names=fields["argument_names"],
        when_to_use=fields["when_to_use"],
        version=fields["version"],
        model=fields["model"],
        disable_model_invocation=fields["disable_model_invocation"],
        user_invocable=fields["user_invocable"],
        source="projectSettings",
        base_dir=str(path.parent),
        loaded_from="project",
        execution_context=fields["execution_context"],
        agent=fields["agent"],
        paths=fields["paths"],
        effort=fields["effort"],
        hooks=fields.get("hooks"),
        shell=fields.get("shell"),
    )
    _dynamic_skills[skill.name] = skill
    logger.info("Registered SOP flat skill: %s from %s", skill.name, path)


def register_bundle_skills(bundle_path: Path, workspace_root: Path) -> BundleSkillLoadResult:
    """Load skills for a POS agent bundle into ``_dynamic_skills``.

    Clears previously registered SOP flat skills so only the active bundle
    remains in the dynamic registry (L2 isolation).
    """
    from clawcodex_ext.skills.loader import (  # noqa: SLF001
        _dynamic_skills,
        _dynamic_skills_by_workspace,
        _workspace_key,
        load_skills_from_skills_dir,
    )

    try:
        from extensions.sop_converter.bundle_context import get_active_bundle, set_active_bundle

        active = get_active_bundle()
        if active is not None and active.bundle_path != bundle_path.resolve():
            set_active_bundle(None)
    except ImportError:
        pass

    search_dirs = _bundle_skill_search_dirs(bundle_path, workspace_root)
    workspace_key = _workspace_key(workspace_root)

    # Drop prior bundle flat skills; standard SKILL.md trees are reloaded below.
    for name in list(_dynamic_skills):
        skill = _dynamic_skills[name]
        base = str(getattr(skill, "base_dir", "") or "")
        if base and any(base.startswith(str(d.resolve())) for d in search_dirs):
            del _dynamic_skills[name]
        elif name.endswith("-skill"):
            del _dynamic_skills[name]

    workspace_skills = _dynamic_skills_by_workspace.setdefault(workspace_key, {})
    for name in list(workspace_skills):
        skill = workspace_skills[name]
        base = str(getattr(skill, "base_dir", "") or "")
        if base and any(base.startswith(str(d.resolve())) for d in search_dirs):
            del workspace_skills[name]
        elif name.endswith("-skill"):
            del workspace_skills[name]

    registered: list[str] = []
    tool_names: dict[str, bool] = {}

    for base in search_dirs:
        for skill in load_skills_from_skills_dir(str(base), "projectSettings", loaded_from="project"):
            _dynamic_skills[skill.name] = skill
            workspace_skills[skill.name] = skill
            registered.append(skill.name)
            for tool in skill.allowed_tools or []:
                if isinstance(tool, str) and tool:
                    tool_names[tool] = True

        for md in sorted(base.glob("*-skill.md")):
            if not md.is_file():
                continue
            before = set(_dynamic_skills.keys())
            _load_flat_skill_markdown(md)
            new_names = set(_dynamic_skills.keys()) - before
            for name in new_names:
                skill = _dynamic_skills.get(name)
                if skill is not None:
                    workspace_skills[name] = skill
            registered.extend(sorted(new_names))
            for name in new_names:
                skill = _dynamic_skills.get(name)
                if skill is None:
                    continue
                for tool in skill.allowed_tools or []:
                    if isinstance(tool, str) and tool:
                        tool_names[tool] = True

    registered = sorted(set(registered))

    skill_tool_count = len(tool_names)
    try:
        from extensions.sop_converter.bundle_context import collect_tool_names_from_bundle_specs

        for name in collect_tool_names_from_bundle_specs(bundle_path):
            tool_names[name] = True
        if not skill_tool_count and tool_names:
            logger.info(
                "Bundle %s: skill frontmatter had no allowed-tools; using %d persisted tool spec names as allowlist",
                bundle_path.name,
                len(tool_names),
            )
    except ImportError:
        pass

    resolved_tool_names = sorted(tool_names)

    if registered:
        try:
            from ..adapters import DEFAULTS

            DEFAULTS.clear_sop_caches()
        except Exception as exc:  # nosec
            logger.warning("Failed to clear SOP caches after bundle skill registration: %s", exc)

    return BundleSkillLoadResult(
        skill_names=registered,
        skill_dirs=search_dirs,
        tool_names=resolved_tool_names,
    )
