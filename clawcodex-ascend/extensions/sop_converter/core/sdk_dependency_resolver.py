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

"""Resolve third-party dependencies declared by an SDK source tree."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[import-not-found]


@dataclass(frozen=True)
class SdkDependencySpec:
    """SDK dependency declarations discovered during ``sop convert``."""

    requirements: tuple[str, ...]
    source: str
    raw_path: str


def resolve_sdk_dependencies(sdk_source_dir: str | Path) -> SdkDependencySpec:
    """Resolve runtime dependencies from ``pyproject.toml`` or ``requirements.txt``.

    Priority is:
    1. ``[project].dependencies`` in ``pyproject.toml``
    2. ``requirements.txt``
    3. empty dependency set
    """

    root = Path(sdk_source_dir).expanduser().resolve()

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        deps = _parse_pyproject_dependencies(pyproject)
        if deps:
            return SdkDependencySpec(
                requirements=tuple(deps),
                source="pyproject.toml",
                raw_path=str(pyproject),
            )
    else:
        logger.debug("No pyproject.toml in %s; falling back to requirements.txt", root)

    requirements = root / "requirements.txt"
    if requirements.is_file():
        deps = _parse_requirements_txt(requirements)
        if deps:
            return SdkDependencySpec(
                requirements=tuple(deps),
                source="requirements.txt",
                raw_path=str(requirements),
            )
    else:
        logger.debug("No requirements.txt in %s; returning empty dependency set", root)

    return SdkDependencySpec(requirements=(), source="empty", raw_path=str(root))


def _parse_pyproject_dependencies(path: Path) -> list[str]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Failed to read or parse %s: %s", path, exc)
        return []

    project = data.get("project")
    if not isinstance(project, dict):
        logger.debug("No [project] table in %s; no dependencies declared", path)
        return []

    deps = project.get("dependencies")
    if not isinstance(deps, list):
        logger.debug("[project].dependencies missing or not a list in %s", path)
        return []

    sdk_names = {_normalise_name(path.parent.name)}
    project_name = project.get("name")
    if isinstance(project_name, str) and project_name.strip():
        sdk_names.add(_normalise_name(project_name))

    result: list[str] = []
    seen: set[str] = set()
    for dep in deps:
        if not isinstance(dep, str):
            continue
        dep = dep.strip()
        if not dep:
            continue
        dep_name = _requirement_name(dep)
        if dep_name and _normalise_name(dep_name) in sdk_names:
            continue
        if dep not in seen:
            result.append(dep)
            seen.add(dep)
    if not result:
        logger.debug("No usable dependencies in %s (empty or all self-references filtered)", path)
    return result


def _parse_requirements_txt(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return []

    result: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line).strip()
        if not line:
            continue
        if _is_requirements_directive(line):
            continue
        if line not in seen:
            result.append(line)
            seen.add(line)
    if not result:
        logger.debug("No dependencies found in %s", path)
    return result


def _strip_inline_comment(line: str) -> str:
    """Strip comments that are separated from the requirement by whitespace."""

    return re.split(r"\s+#", line, maxsplit=1)[0]


def _is_requirements_directive(line: str) -> bool:
    return line.startswith(
        (
            "-",
            "--",
        )
    )


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if not match:
        return ""
    return match.group(1)


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
