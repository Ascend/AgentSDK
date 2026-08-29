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

"""Load handwritten MacroDefinition YAML (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import MacroConvertError
from .models import MacroDefinition, MacroRoute


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise MacroConvertError(
            "macro_yaml_unavailable",
            "PyYAML is required to load macro manifests",
        ) from exc
    return yaml


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value if str(item).strip()]


def parse_macro_route(data: dict[str, Any] | None, *, default_target: str = "") -> MacroRoute:
    raw = data if isinstance(data, dict) else {}
    match_mode = str(raw.get("match_mode") or "all")
    if match_mode not in ("exact", "all", "any"):
        match_mode = "all"
    selection = str(raw.get("selection") or "prefer")
    if selection not in ("exclusive", "prefer"):
        selection = "prefer"
    try:
        priority = int(raw.get("priority", 100))
    except (TypeError, ValueError):
        priority = 100
    scope = str(raw.get("scope") or "bundle")
    if scope not in ("session", "bundle", "builtin"):
        scope = "bundle"
    unavailable_policy = str(raw.get("unavailable_policy") or "restore-covered").strip()
    if unavailable_policy != "restore-covered":
        raise MacroConvertError(
            "macro_schema_invalid",
            f"invalid routing.unavailable_policy: {raw.get('unavailable_policy')!r}",
            field="routing.unavailable_policy",
        )
    target = str(raw.get("target_tool") or default_target or "").strip()
    return MacroRoute(
        phrases=_as_str_list(raw.get("phrases")),
        keywords=_as_str_list(raw.get("keywords")),
        negative_keywords=_as_str_list(raw.get("negative_keywords")),
        target_tool=target,
        match_mode=match_mode,  # type: ignore[arg-type]
        selection=selection,  # type: ignore[arg-type]
        priority=priority,
        verified=bool(raw.get("verified", False)),
        enabled=raw.get("enabled", True) is not False,
        intent_key=str(raw.get("intent_key") or "").strip(),
        covered_tools=_as_str_list(raw.get("covered_tools")),
        unavailable_policy=unavailable_policy,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
    )


def parse_macro_definition(data: dict[str, Any], *, source: str = "") -> MacroDefinition:
    if not isinstance(data, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro definition must be a mapping",
            manifest=source,
        )
    try:
        version = int(data.get("version", 1))
    except (TypeError, ValueError) as exc:
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro version must be an integer",
            manifest=source,
            field="version",
        ) from exc
    if version != 1:
        raise MacroConvertError(
            "macro_version_unsupported",
            f"unsupported macro schema version: {version}",
            manifest=source,
            field="version",
        )
    name = str(data.get("name") or "").strip()
    if not name:
        raise MacroConvertError(
            "macro_schema_invalid",
            "macro name is required",
            manifest=source,
            field="name",
        )
    scope = str(data.get("scope") or "bundle")
    if scope not in ("bundle", "session", "builtin"):
        raise MacroConvertError(
            "macro_schema_invalid",
            f"unsupported macro scope: {scope}",
            manifest=source,
            field="scope",
        )
    workflow = data.get("workflow")
    if not isinstance(workflow, dict):
        raise MacroConvertError(
            "macro_schema_invalid",
            "workflow block is required",
            manifest=source,
            field="workflow",
        )
    routing = parse_macro_route(data.get("routing"), default_target=name)
    if not routing.target_tool:
        routing.target_tool = name
    routing.scope = "bundle" if scope == "bundle" else routing.scope
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {"kind": "handwritten", "manifest": source}
    else:
        provenance = dict(provenance)
        provenance.setdefault("manifest", source)
    return MacroDefinition(
        version=version,
        name=name,
        description=str(data.get("description") or ""),
        scope=scope,  # type: ignore[arg-type]
        enabled=data.get("enabled", True) is not False,
        workflow=workflow,
        routing=routing,
        provenance=provenance,
    )


def load_macro_yaml(path: Path) -> MacroDefinition:
    yaml = _require_yaml()
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except OSError as exc:
        raise MacroConvertError(
            "macro_manifest_unreadable",
            f"cannot read macro manifest: {path}",
            manifest=str(path),
        ) from exc
    except Exception as exc:
        raise MacroConvertError(
            "macro_schema_invalid",
            f"invalid YAML in macro manifest: {exc}",
            manifest=str(path),
        ) from exc
    return parse_macro_definition(data, source=str(path))


def discover_macro_sources(
    *,
    source_dir: Path | None = None,
    manifest_paths: list[Path] | None = None,
) -> list[Path]:
    """Collect MacroDefinition paths from ``sop-macros/`` and explicit manifests."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        found.append(path)

    for path in manifest_paths or []:
        _add(Path(path))

    if source_dir is not None:
        macros_dir = Path(source_dir) / "sop-macros"
        if macros_dir.is_dir():
            for path in sorted(macros_dir.glob("*.yaml")) + sorted(macros_dir.glob("*.yml")):
                _add(path)

    return found


def load_macro_definitions(
    *,
    source_dir: Path | None = None,
    manifest_paths: list[Path] | None = None,
) -> list[MacroDefinition]:
    return [
        load_macro_yaml(path)
        for path in discover_macro_sources(
            source_dir=source_dir,
            manifest_paths=manifest_paths,
        )
    ]
