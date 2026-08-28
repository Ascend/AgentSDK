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

"""CLI entrypoint discovery helpers for generated workflow bridges."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]


def split_cli_prefix(cli_prefix: str | None) -> list[str]:
    """Split a configured CLI prefix into argv tokens.

    Uses non-POSIX splitting on Windows so backslash paths are preserved.
    """

    if not cli_prefix:
        return []
    return shlex.split(cli_prefix, posix=os.name != "nt")


def discover_cli_prefix(
    source_dir: Path,
    project_name: str,
    *,
    override: str | None = None,
) -> str | None:
    """Return a CLI prefix from override or matching ``[project.scripts]``.

    When neither is available, returns ``None`` so the bridge generator can
    fall back to executing the source file directly (or skip CLI mode).
    """

    if override and override.strip():
        return override.strip()

    pyproject = Path(source_dir) / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, TypeError):
        return None

    scripts = data.get("project", {}).get("scripts") or {}
    if not isinstance(scripts, dict):
        return None
    if project_name in scripts:
        return project_name
    return None
