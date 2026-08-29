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

# pylint: disable=cell-var-from-loop
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .build_tool import Tool, build_tool
from clawcodex_ext.tool_system.protocol import ToolResult
from .registry import ToolRegistry


def load_tools_from_dir(
    directory: str | Path,
    registry: ToolRegistry | None = None,
) -> list[Tool]:
    directory = Path(directory)
    if not directory.is_dir():
        return []

    tools: list[Tool] = []
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"_clawcodex_plugin_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:  # nosec
            continue

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, Tool):
                tools.append(attr)
                if registry is not None:
                    try:
                        registry.register(attr)
                    except ValueError:
                        pass  # Invalid candidate; continue with the surrounding fallback.

        if not any(isinstance(getattr(mod, a), Tool) for a in dir(mod)):
            spec_dict = getattr(mod, "tool_spec", None)
            run_fn = getattr(mod, "run", None)
            if isinstance(spec_dict, dict) and callable(run_fn):

                def _make_call(fn: Any) -> Any:
                    def _call(tool_input: dict[str, Any], context: Any) -> ToolResult:
                        result = fn(tool_input, context)
                        if isinstance(result, ToolResult):
                            return result
                        return ToolResult(name=spec_dict.get("name", ""), output=result)

                    return _call

                t = build_tool(
                    name=spec_dict.get("name", py_file.stem),
                    input_schema=spec_dict.get("input_schema", {"type": "object", "properties": {}}),
                    call=_make_call(run_fn),
                    description=spec_dict.get("description", ""),
                )
                tools.append(t)
                if registry is not None:
                    try:
                        registry.register(t)
                    except ValueError:
                        pass  # Invalid candidate; continue with the surrounding fallback.

    return tools
