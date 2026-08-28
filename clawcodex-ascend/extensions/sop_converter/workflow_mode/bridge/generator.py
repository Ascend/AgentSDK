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


"""Generate Python / CLI stage bridges for wrapper/hybrid profiles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...path_resolver import format_extra_sys_path_inserts, infer_extra_sys_path_entries
from ..capability.models import StageAgentMap
from ..extractors.models import WorkflowGraph
from .cli_discovery import discover_cli_prefix, split_cli_prefix
from .dispatch import build_bridge_tables
from .health_check import run_bridge_health_check, write_health_json

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class BridgeGenerator:
    """Generate stage-level bridge script for wrapper/hybrid profiles."""

    def __init__(self) -> None:
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
        )

    def generate(
        self,
        graph: WorkflowGraph,
        agent_map: StageAgentMap,
        source_dir: Path,
        output_dir: Path,
        *,
        mode: str = "python",
        project_name: str = "project",
        cli_entry: str | None = None,
        run_health_check: bool = True,
    ) -> Path | None:
        if mode not in ("python", "cli"):
            raise ValueError(f"unsupported bridge mode {mode!r}; use python or cli")

        source_dir = Path(source_dir).resolve()
        output_dir = Path(output_dir)
        tables = build_bridge_tables(graph, agent_map, source_dir)
        if tables is None:
            logger.info("No wrapper/hybrid stages — skipping bridge generation")
            return None

        stage_dispatch, stage_outputs = tables
        bridge_dir = output_dir / "bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        script_name = f"{project_name.replace('-', '_')}_bridge.py"
        script_path = bridge_dir / script_name
        resolved_cli_prefix: str | None = None

        if mode == "python":
            body = self._render_python_bridge(source_dir, stage_dispatch, stage_outputs)
        else:
            resolved_cli_prefix = discover_cli_prefix(source_dir, project_name, override=cli_entry)
            if not resolved_cli_prefix:
                logger.error("CLI bridge requires --bridge-cli or [project.scripts] in pyproject.toml")
                return None
            body = self._render_cli_bridge(resolved_cli_prefix, stage_dispatch, stage_outputs)

        script_path.write_text(body, encoding="utf-8")

        if run_health_check:
            report = run_bridge_health_check(
                script_path,
                source_dir,
                stage_dispatch,
                mode=mode,
                cli_prefix=resolved_cli_prefix,
            )
            write_health_json(bridge_dir, report)
            if not report.get("ok"):
                logger.warning("Bridge health check reported issues (see bridge/health.json)")

        return script_path

    def _render_python_bridge(
        self,
        source_dir: Path,
        stage_dispatch: dict[int, dict[str, Any]],
        stage_outputs: dict[int, list[str]],
    ) -> str:
        extra_paths: list[str] = []
        for meta in stage_dispatch.values():
            module_path = meta.get("module_path")
            if module_path:
                module_name = module_path.replace(".py", "").replace("/", ".")
                extra_paths.extend(infer_extra_sys_path_entries(str(source_dir), module_name))
        extra_paths = list(dict.fromkeys(extra_paths))
        extra_sys_path_inserts = format_extra_sys_path_inserts(extra_paths)

        template = self._jinja.get_template("python_bridge.py.j2")
        return template.render(
            source_dir=str(source_dir),
            stage_dispatch_json=repr(stage_dispatch),
            stage_outputs_json=repr(stage_outputs),
            extra_sys_path_inserts=extra_sys_path_inserts,
        )

    def _render_cli_bridge(
        self,
        cli_prefix: str,
        stage_dispatch: dict[int, dict[str, Any]],
        stage_outputs: dict[int, list[str]],
    ) -> str:
        template = self._jinja.get_template("cli_bridge.py.j2")
        argv_prefix = split_cli_prefix(cli_prefix)
        return template.render(
            cli_prefix_json=repr(argv_prefix),
            stage_dispatch_json=repr(stage_dispatch),
            stage_outputs_json=repr(stage_outputs),
        )
