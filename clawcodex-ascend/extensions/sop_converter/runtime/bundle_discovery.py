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

"""Discover POS agent bundles in a workspace for SOP auto-activation."""

from __future__ import annotations

import logging
from pathlib import Path

from .bundle_skills import _bundle_skill_search_dirs

logger = logging.getLogger(__name__)


def _is_sop_skill_name(name: str) -> bool:
    return isinstance(name, str) and name.endswith("-skill")


def _has_bundle_manifest(path: Path) -> bool:
    """True when *path* carries a ``bundle.json`` written by ``sop convert``.

    The manifest is the canonical, SDK-agnostic marker of a POS bundle root
    (see ``extensions/sop_converter/core/bundle_manifest.py``); probing it
    never raises on a missing or malformed file.
    """
    try:
        from extensions.sop_converter.bundle_manifest import read_bundle_manifest
    except ImportError:
        return False
    return read_bundle_manifest(path) is not None


def _looks_like_pos_bundle(path: Path) -> bool:
    """Content-based POS bundle detection — no SDK-name convention required.

    A directory is a POS bundle root when it carries conversion artifacts:
    a ``bundle.json`` manifest or persisted agent-tool specs.  Flat
    ``*-skill.md`` files alone do not qualify — ordinary project skill
    folders share that shape, so discovery deliberately ignores it.
    """
    if not path.is_dir():
        return False
    if _has_bundle_manifest(path):
        return True
    from ..adapters import DEFAULTS

    return bool(DEFAULTS.tool_authoring.iter_bundle_tool_dirs(path))


def list_workspace_bundle_candidates(workspace: Path) -> list[Path]:
    """Return candidate bundle roots under *workspace* (deduplicated by name).

    Candidates are identified purely by conversion artifacts (manifest /
    persisted tool specs), never by a directory-name prefix, so discovery
    stays SDK-agnostic.
    """
    ws = workspace.resolve()
    by_name: dict[str, Path] = {}

    clawcodex_root = ws / ".clawcodex"
    if clawcodex_root.is_dir():
        for child in sorted(clawcodex_root.iterdir()):
            if not child.is_dir():
                continue
            if _looks_like_pos_bundle(child):
                by_name.setdefault(child.name, child)

    skills_root = ws / "skills"
    if skills_root.is_dir():
        for child in sorted(skills_root.iterdir()):
            if not child.is_dir():
                continue
            preferred = by_name.get(child.name) or child
            if _looks_like_pos_bundle(preferred) or _looks_like_pos_bundle(child):
                by_name[child.name] = preferred

    return sorted(by_name.values(), key=lambda p: p.name)


def _skill_hits_in_bundle(
    bundle_path: Path,
    workspace: Path,
    skill_names: list[str],
) -> int:
    search_dirs = _bundle_skill_search_dirs(bundle_path, workspace)
    hits = 0
    for skill in skill_names:
        if not _is_sop_skill_name(skill):
            continue
        md_name = skill if skill.endswith(".md") else f"{skill}.md"
        if any((d / md_name).is_file() for d in search_dirs):
            hits += 1
    return hits


def discover_workspace_bundle(
    workspace: Path,
    *,
    agent_skills: list[str] | None = None,
) -> Path | None:
    """Pick the best POS bundle for an overview agent in *workspace*."""
    candidates = list_workspace_bundle_candidates(workspace)
    if not candidates:
        return None

    skills = [s for s in (agent_skills or []) if _is_sop_skill_name(s)]
    if not skills:
        return candidates[0] if len(candidates) == 1 else None

    best_path: Path | None = None
    best_hits = 0
    for bundle_path in candidates:
        hits = _skill_hits_in_bundle(bundle_path, workspace, skills)
        if hits > best_hits:
            best_hits = hits
            best_path = bundle_path

    if best_path is not None and best_hits > 0:
        return best_path
    return candidates[0] if len(candidates) == 1 else None


def overview_has_sop_skills(agent: dict) -> bool:
    skills = agent.get("skills") or []
    return any(_is_sop_skill_name(s) for s in skills if isinstance(s, str))
