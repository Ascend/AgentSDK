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

from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Optional, Sequence


def create_skill(
    *,
    directory: str | Path,
    name: str,
    description: str,
    when_to_use: Optional[str] = None,
    allowed_tools: Optional[Sequence[str]] = None,
    arguments: Optional[Sequence[str]] = None,
    user_invocable: bool = True,
    disable_model_invocation: bool = False,
    context: str = "inline",
    agent: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
    effort: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
    body: str = "",
) -> Path:
    base = Path(directory).expanduser().resolve()
    skill_dir = _resolve_skill_dir(base, name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    fm: dict[str, Any] = {
        "description": description,
        "user-invocable": user_invocable,
        "disable-model-invocation": disable_model_invocation,
    }
    if when_to_use is not None:
        fm["when_to_use"] = when_to_use
    if allowed_tools:
        fm["allowed-tools"] = list(allowed_tools)
    if arguments:
        fm["arguments"] = list(arguments)
    if context and context != "inline":
        fm["context"] = context
    if agent is not None:
        fm["agent"] = agent
    if version is not None:
        fm["version"] = version
    if model is not None:
        fm["model"] = model
    if effort is not None:
        fm["effort"] = effort
    if paths:
        fm["paths"] = list(paths)

    content = _render_frontmatter(fm) + "\n" + (body or "")
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def _resolve_skill_dir(base: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("skill name must resolve inside the skills directory")

    windows_path = PureWindowsPath(name)
    posix_path = PurePosixPath(name.replace("\\", "/"))
    if any(
        (
            windows_path.is_absolute(),
            bool(windows_path.drive),
            posix_path.is_absolute(),
            not posix_path.parts,
            posix_path == PurePosixPath("."),
            ".." in posix_path.parts,
        )
    ):
        raise ValueError("skill name must resolve inside the skills directory")

    skill_dir = base.joinpath(*posix_path.parts).resolve(strict=False)
    try:
        skill_dir.relative_to(base)
    except ValueError as exc:
        raise ValueError("skill name must resolve inside the skills directory") from exc
    return skill_dir


def _render_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value), ensure_ascii=False)


def _render_frontmatter(fm: Mapping[str, Any]) -> str:
    lines: list[str] = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {_render_yaml_scalar(item)}")
        else:
            lines.append(f"{k}: {_render_yaml_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)
