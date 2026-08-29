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

"""Atomic persist of MacroDefinition into bundle ``.clawcodex/macros/``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import MacroConvertError
from .models import MacroDefinition


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise MacroConvertError(
            "macro_yaml_unavailable",
            "PyYAML is required to persist macro manifests",
        ) from exc
    return yaml


def macros_dir(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / ".clawcodex" / "macros"


def macro_relative_manifest(name: str) -> str:
    return f".clawcodex/macros/{name}.yaml"


def macro_definition_to_dict(macro: MacroDefinition) -> dict[str, Any]:
    route = macro.routing
    return {
        "version": macro.version,
        "name": macro.name,
        "description": macro.description,
        "scope": macro.scope,
        "enabled": macro.enabled,
        "workflow": macro.workflow,
        "routing": {
            "phrases": list(route.phrases),
            "keywords": list(route.keywords),
            "negative_keywords": list(route.negative_keywords),
            "target_tool": route.target_tool or macro.name,
            "match_mode": route.match_mode,
            "selection": route.selection,
            "priority": route.priority,
            "verified": route.verified,
            "enabled": route.enabled,
            "intent_key": route.intent_key,
            "covered_tools": list(route.covered_tools),
            "unavailable_policy": route.unavailable_policy,
            "scope": route.scope,
        },
        "provenance": dict(macro.provenance),
    }


def write_macro_yaml(path: Path, macro: MacroDefinition) -> None:
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = macro_definition_to_dict(macro)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".yaml.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
        os.replace(tmp_name, path)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def persist_macros_atomic(
    macros: list[MacroDefinition],
    bundle_dir: Path,
) -> list[Path]:
    """Write all macros or leave no partial files from this batch."""
    target_dir = macros_dir(bundle_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        root = target_dir.resolve()
        for macro in macros:
            name = str(macro.name or "")
            if not name or "/" in name or "\\" in name or ".." in name:
                raise MacroConvertError(
                    "macro_persist_failed",
                    f"unsafe macro name for persist path: {name!r}",
                )
            macro.provenance["manifest"] = macro_relative_manifest(name)
            path = (target_dir / f"{name}.yaml").resolve()
            try:
                escaped = not path.is_relative_to(root)
            except ValueError:
                escaped = True
            if escaped:
                raise MacroConvertError(
                    "macro_persist_failed",
                    f"macro persist path escapes macros dir: {name!r}",
                )
            write_macro_yaml(path, macro)
            written.append(path)
        return written
    except MacroConvertError:
        for path in written:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    except Exception as exc:
        for path in written:
            try:
                path.unlink()
            except OSError:
                pass
        raise MacroConvertError(
            "macro_persist_failed",
            f"atomic macro persist failed: {exc}",
            manifest=str(bundle_dir),
        ) from exc
