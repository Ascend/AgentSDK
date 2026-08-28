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

"""Atomic YAML writer for :class:`ToolDependencyGraph`.

Falls back to a hand-rolled emitter - the structure is shallow enough
that a manual serialiser stays readable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .models import ToolDependencyGraph

logger = logging.getLogger(__name__)


def _yaml_dump(data: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except ImportError:
        logger.warning("PyYAML unavailable; falling back to built-in YAML-subset emitter")
        return _yaml_subset_dump(data)
    except Exception as exc:  # pragma: no cover - e.g. RepresenterError on exotic values
        logger.warning("PyYAML dump failed: %s; falling back to built-in YAML-subset emitter", exc)
        return _yaml_subset_dump(data)


def _yaml_subset_dump(data: dict[str, Any]) -> str:
    """Hand-rolled block-style YAML emitter (a YAML subset).

    Used when PyYAML is unavailable so the ``.yaml`` output stays valid
    block-style YAML that the reader's minimal loader can parse back,
    instead of JSON masquerading as YAML.
    """

    def scalar(value: Any) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if text == "" or text.lower() in ("null", "true", "false", "~") or re.search(r"[:#\[\]{}&*!|>'\"%@`]", text):
            return json.dumps(text, ensure_ascii=False)
        return text

    def emit_map(mapping: dict[str, Any], indent: int) -> list[str]:
        pad = "  " * indent
        lines: list[str] = []
        for key, value in mapping.items():
            if isinstance(value, dict):
                if not value:
                    lines.append(f"{pad}{key}: {{}}")
                    continue
                lines.append(f"{pad}{key}:")
                lines.extend(emit_map(value, indent + 1))
            elif isinstance(value, list):
                if not value:
                    lines.append(f"{pad}{key}: []")
                    continue
                lines.append(f"{pad}{key}:")
                lines.extend(emit_seq(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {scalar(value)}")
        return lines

    def emit_seq(items: list[Any], indent: int) -> list[str]:
        pad = "  " * indent
        lines: list[str] = []
        for item in items:
            if isinstance(item, dict):
                first = True
                for key, value in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(value, dict):
                        if not value:
                            lines.append(f"{prefix}{key}: {{}}")
                            continue
                        lines.append(f"{prefix}{key}:")
                        lines.extend(emit_map(value, indent + 2))
                    elif isinstance(value, list):
                        if not value:
                            lines.append(f"{prefix}{key}: []")
                            continue
                        lines.append(f"{prefix}{key}:")
                        lines.extend(emit_seq(value, indent + 2))
                    else:
                        lines.append(f"{prefix}{key}: {scalar(value)}")
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.extend(emit_seq(item, indent + 1))
            else:
                lines.append(f"{pad}- {scalar(item)}")
        return lines

    return "\n".join(emit_map(data, 0)) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("atomic write failed for %s (%s); falling back", path, exc)
        path.write_text(content, encoding="utf-8")


def write_tool_dependencies(
    graph: ToolDependencyGraph,
    path: str | Path,
    *,
    project_name: str = "",
) -> Path:
    """Persist ``graph`` to ``path`` as ``tool-dependencies.yaml``."""
    out = Path(path)
    header = "# tool-dependencies.yaml - SOP bundle dependency graph\n# auto-generated: extensions/sop_converter/dependency\n"
    if project_name:
        header += f"# project: {project_name}\n"
    body = _yaml_dump(graph.to_dict())
    _atomic_write_text(out, header + "\n" + body)
    return out


__all__ = ["write_tool_dependencies"]
