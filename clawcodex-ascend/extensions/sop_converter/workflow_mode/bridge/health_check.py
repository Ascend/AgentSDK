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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Health check for generated bridge scripts."""

# pylint: disable=too-many-nested-blocks
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from ...path_resolver import infer_extra_sys_path_entries
from .cli_discovery import split_cli_prefix

logger = logging.getLogger(__name__)


def _cli_prefix_resolvable(cli_prefix: str) -> tuple[bool, str]:
    """Return (ok, issue message). *cli_prefix* may be a multi-token shell prefix."""
    tokens = split_cli_prefix(cli_prefix)
    if not tokens:
        return False, "empty CLI prefix"
    executable = tokens[0]
    path = Path(executable)
    if path.is_file():
        return True, ""
    if shutil.which(executable):
        return True, ""
    return False, f"CLI executable not found: {executable!r}"


def run_bridge_health_check(
    bridge_script: Path,
    source_dir: Path,
    stage_dispatch: dict[int, dict[str, Any]],
    *,
    mode: str = "python",
    cli_prefix: str | None = None,
) -> dict[str, Any]:
    """Verify bridge prerequisites; non-fatal diagnostics."""
    result: dict[str, Any] = {
        "bridge_script": str(bridge_script),
        "source_dir": str(source_dir),
        "mode": mode,
        "stages": [],
        "import_ok": False,
        "ok": True,
    }
    # Import probing below mutates sys.path; restore it on every exit path so
    # this diagnostics function has no lasting side effects on callers.
    saved_path = list(sys.path)
    try:
        if not bridge_script.is_file():
            result["ok"] = False
            result["error"] = "bridge script missing"
            return result

        if not source_dir.is_dir():
            result["ok"] = False
            result["error"] = "source_dir missing"
            return result

        if not stage_dispatch:
            result["ok"] = False
            result["error"] = "no stages registered in dispatch table"
            return result

        if mode == "python":
            for meta in stage_dispatch.values():
                module_path = meta.get("module_path")
                if module_path:
                    module_name = module_path.replace(".py", "").replace("/", ".")
                    extra_paths = infer_extra_sys_path_entries(str(source_dir), module_name)
                    for path in extra_paths:
                        if path not in sys.path:
                            sys.path.insert(0, path)

        if mode == "cli":
            if not cli_prefix:
                result["ok"] = False
                result["error"] = "CLI prefix missing"
                return result
            ok, issue = _cli_prefix_resolvable(cli_prefix)
            result["cli_prefix"] = cli_prefix
            if not ok:
                result["ok"] = False
                result["cli_issue"] = issue
            for stage_id, meta in sorted(stage_dispatch.items()):
                entry: dict[str, Any] = {
                    "stage_id": stage_id,
                    "ok": ok,
                    "stage_name": meta.get("stage_name"),
                    "issues": [] if ok else [issue],
                }
                if not ok:
                    result["ok"] = False
                result["stages"].append(entry)
            return result

        for stage_id, meta in sorted(stage_dispatch.items()):
            entry: dict[str, Any] = {"stage_id": stage_id, "ok": True, "issues": []}
            rel = meta.get("module_path")
            if not rel:
                entry["ok"] = False
                entry["issues"].append("no module_path")
            else:
                mod_path = source_dir / rel
                if not mod_path.is_file():
                    entry["ok"] = False
                    entry["issues"].append(f"missing module {rel}")
                else:
                    fn = meta.get("entry_function")
                    if fn:
                        try:
                            spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
                            if spec and spec.loader:
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                if not hasattr(mod, fn):
                                    entry["ok"] = False
                                    entry["issues"].append(f"missing function {fn}")
                        except Exception as exc:
                            entry["ok"] = False
                            entry["issues"].append(f"import failed: {exc}")
            if not entry["ok"]:
                result["ok"] = False
            result["stages"].append(entry)

        result["import_ok"] = all(entry["ok"] for entry in result["stages"])
        return result
    finally:
        sys.path[:] = saved_path


def write_health_json(bridge_dir: Path, report: dict[str, Any]) -> Path:
    bridge_dir.mkdir(parents=True, exist_ok=True)
    path = bridge_dir / "health.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
