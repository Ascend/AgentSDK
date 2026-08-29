#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Persistence for agent-created tools — saves/loads specs to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec

TOOL_DIR = Path.home() / ".clawcodex" / "agent-tools"
TOOL_DIR.mkdir(parents=True, exist_ok=True)

BUNDLES_ROOT = TOOL_DIR / "bundles"


def bundle_tool_dir(bundle_path: Path) -> Path:
    """Primary L3 storage: ``<bundle>/agent-tools``."""
    return bundle_path.resolve() / "agent-tools"


def legacy_bundle_tool_dir(bundle_name: str) -> Path:
    """Centralized per-bundle fallback under ``~/.clawcodex/agent-tools/bundles/``."""
    return BUNDLES_ROOT / bundle_name


def iter_bundle_tool_dirs(bundle_path: Path) -> list[Path]:
    """Search paths for persisted tools belonging to *bundle_path*."""
    bundle_path = bundle_path.resolve()
    dirs: list[Path] = []
    for candidate in (
        bundle_tool_dir(bundle_path),
        legacy_bundle_tool_dir(bundle_path.name),
    ):
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def scripts_dir_for(tool_dir: Path) -> Path:
    path = tool_dir / "scripts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_tool_dir(*, bundle_path: Path | None = None, bundle_id: str | None = None) -> Path:
    """Resolve the directory used to persist tool JSON for a bundle."""
    if bundle_path is not None:
        return bundle_tool_dir(bundle_path)
    if bundle_id:
        return legacy_bundle_tool_dir(bundle_id)
    return TOOL_DIR


def _spec_to_dict(spec: AgentToolSpec) -> dict[str, Any]:
    data = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "call_type": spec.call_type,
        "call_impl": spec.call_impl,
        "tags": list(spec.tags),
        "aliases": list(spec.aliases),
        "source": spec.source,
    }
    if spec.bundle_id:
        data["bundle_id"] = spec.bundle_id
    if spec.stateful_wrapper:
        data["stateful_wrapper"] = True
    if spec.output_schema is not None:
        data["output_schema"] = spec.output_schema
    return data


def _dict_to_spec(d: dict[str, Any]) -> AgentToolSpec:
    return AgentToolSpec(
        name=d["name"],
        description=d["description"],
        input_schema=d["input_schema"],
        call_type=d["call_type"],
        call_impl=d["call_impl"],
        tags=tuple(d.get("tags", ())),
        aliases=tuple(d.get("aliases", ())),
        source=d.get("source", "agent-created"),
        bundle_id=d.get("bundle_id"),
        stateful_wrapper=bool(d.get("stateful_wrapper", False)),
        output_schema=d.get("output_schema"),
    )


def save_spec(spec: AgentToolSpec, *, tool_dir: Path | None = None) -> None:
    """Persist a tool spec to disk."""
    base = tool_dir or resolve_tool_dir(bundle_id=spec.bundle_id)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{spec.name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_spec_to_dict(spec), f, indent=2, ensure_ascii=False)


def load_spec(name: str, *, tool_dir: Path | None = None) -> AgentToolSpec | None:
    """Load a tool spec from disk, returning None if not found."""
    base = tool_dir or TOOL_DIR
    path = base / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return _dict_to_spec(json.load(f))


def delete_spec(name: str, *, tool_dir: Path | None = None) -> bool:
    """Remove a persisted spec. Returns True if it existed."""
    base = tool_dir or TOOL_DIR
    path = base / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_persisted_specs(*, tool_dir: Path | None = None) -> list[AgentToolSpec]:
    """Load tool specs from a directory (defaults to legacy global dir)."""
    base = tool_dir or TOOL_DIR
    specs: list[AgentToolSpec] = []
    if not base.is_dir():
        return specs
    for path in sorted(base.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                specs.append(_dict_to_spec(json.load(f)))
        except Exception:  # nosec
            continue
    return specs


def clear_persisted(*, tool_dir: Path | None = None) -> None:
    """Remove all persisted tool specs in a directory."""
    base = tool_dir or TOOL_DIR
    if not base.is_dir():
        return
    for path in base.glob("*.json"):
        path.unlink()
