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
#
"""Compatibility projections of the canonical skill discovery paths.

These helpers do not discover or cache skills. They expose the path layout used
by clawcodex_ext.skills.loader for integrations that still import
extensions.skills_ext.paths.
"""

from __future__ import annotations


import os
from pathlib import Path

from clawcodex_ext.skills import loader as canonical_loader


def _get_global_config_dir() -> Path:
    """Return the canonical Claude user configuration directory.

    NOTE: delegates to the canonical loader's private ``_get_global_config_dir``.
    ``clawcodex_ext.skills.loader`` exposes no public equivalent that returns a
    ``Path`` (``get_skills_path`` only returns a concatenated ``str``); this
    extension and the loader live in the same repository/version, so the
    private call is version-locked and safe. Revisit if the loader grows a
    public accessor.
    """

    return canonical_loader._get_global_config_dir()


def _get_managed_file_path() -> Path:
    """Return the canonical managed configuration directory.

    NOTE: same rationale as :func:`_get_global_config_dir` — no public
    ``Path``-returning equivalent exists in the canonical loader.
    """

    return canonical_loader._get_managed_file_path()


def get_clawcodex_skills_dir() -> Path:
    """Return the legacy ClawCodex user skill directory."""

    return Path.home() / ".clawcodex" / "skills"


def get_clawcodex_managed_skills_dir() -> Path | None:
    """Return the optional ClawCodex managed override directory."""

    value = os.environ.get("CLAWCODEX_MANAGED_SKILLS_DIR")
    return Path(value).expanduser().resolve() if value else None


def get_clawcodex_user_skills_dirs() -> list[Path]:
    """Return ClawCodex-specific user roots used by the canonical loader.

    NOTE: ``_legacy_user_skill_dirs``/``_legacy_project_skill_dirs`` are
    intentionally called — the canonical loader documents them as
    "Compatibility seams retained for downstream plugins/tests", which is
    exactly this extension's role. Calls are version-locked within the
    same repository.
    """

    roots = canonical_loader._legacy_user_skill_dirs(None)
    clawcodex_roots = {get_clawcodex_skills_dir().expanduser().resolve()}
    env_root = os.environ.get("CLAWCODEX_SKILLS_DIR")
    if env_root:
        clawcodex_roots.add(Path(env_root).expanduser().resolve())
    return [root for root in roots if root in clawcodex_roots]


def get_clawcodex_project_skills_dir(
    project_root: str | Path | None,
) -> Path | None:
    """Return the legacy project skill directory, if a root was supplied."""

    if project_root is None:
        return None
    dirs = canonical_loader._legacy_project_skill_dirs(project_root)
    return dirs[0] if dirs else None


def _append_unique(target: list[str], path: str | Path) -> None:
    normalized = str(Path(path).expanduser().resolve())
    if normalized not in target:
        target.append(normalized)


def resolve_skills_paths(
    project_root: str | Path | None = None,
    user_skills_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Project the discovery roots owned by the canonical loader.

    The returned mapping is informational only. Calling it never loads skills or
    creates another cache.
    """

    result: dict[str, list[str]] = {"user": [], "project": [], "managed": []}

    _append_unique(
        result["user"],
        canonical_loader.get_skills_path("userSettings"),
    )
    for root in canonical_loader._legacy_user_skill_dirs(user_skills_dir):
        _append_unique(result["user"], root)

    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        _append_unique(result["project"], root / ".claude" / "skills")
        for legacy_root in canonical_loader._legacy_project_skill_dirs(root):
            _append_unique(result["project"], legacy_root)

    _append_unique(
        result["managed"],
        canonical_loader.get_skills_path("policySettings"),
    )
    managed_override = get_clawcodex_managed_skills_dir()
    if managed_override is not None:
        _append_unique(result["managed"], managed_override)

    return result
