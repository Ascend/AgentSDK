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


"""Register composite tools and optional workflow.yaml sidecars.

This module historically forced ``call_type=\"bash\"`` plus a copied wrapper
script. Registration now shares converters with
:mod:`extensions.sop_converter.runtime.composite_tools`, which respects
``CompositeToolSpec.call_type`` / ``call_impl`` (including ``workflow`` +
builtin catalog ids) and registers workflow specs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...adapters import DEFAULTS

from . import (
    _SKIP_PLACEHOLDER_COMPOSITE_TOOLS,
    _composite_to_agent_tool_spec,
    emit_composite_workflow_yaml,
    persist_builtin_retrieval_index,
    to_kebab_case,
)
from .builtin import builtin_composite_tools
from .models import CompositeToolSpec

logger = logging.getLogger(__name__)


def register_composite_tool(
    spec: CompositeToolSpec,
    *,
    persist: bool = True,
    bundle_dir: Path | None = None,
    sdk_source_dir: str | Path | None = None,
) -> str | None:
    """Persist one composite tool using the shared agent-spec converter.

    ``sdk_source_dir`` is accepted for API compatibility and ignored: workflow
    tools no longer require a per-macro Bash wrapper script copy.
    """
    del sdk_source_dir  # compatibility only
    if _SKIP_PLACEHOLDER_COMPOSITE_TOOLS and spec.call_impl is None:
        logger.info("Skipping placeholder composite tool: %s", spec.name)
        return None

    if spec.workflow_spec is not None:
        from extensions.sop_converter.runtime.macros import register_macro

        register_macro(
            f"builtin:{to_kebab_case(spec.name)}",
            spec.workflow_spec,
            replace=True,
        )

    bundle_path = bundle_dir.resolve() if bundle_dir is not None else None
    ta = DEFAULTS.tool_authoring
    tool_dir = ta.bundle_tool_dir(bundle_path) if bundle_path is not None else None
    agent_spec = _composite_to_agent_tool_spec(spec, bundle_dir=bundle_path)
    try:
        ta.validate_spec(agent_spec)
    except Exception as exc:
        logger.warning("Composite tool validation failed for %s: %s", spec.name, exc)
        return None

    if persist:
        ta.save_spec(agent_spec, tool_dir=tool_dir if tool_dir is not None else ta.TOOL_DIR)
    return agent_spec.name


def register_composite_tools(
    *,
    persist: bool = True,
    bundle_dir: Path | str | None = None,
    sdk_source_dir: str | Path | None = None,
    specs: list[CompositeToolSpec] | None = None,
) -> dict[str, str]:
    """Register built-in (or supplied) composite tools."""
    bundle_path = Path(bundle_dir).resolve() if bundle_dir is not None else None
    out: dict[str, str] = {}
    for spec in specs or builtin_composite_tools(bundle_dir=bundle_path):
        name = register_composite_tool(
            spec,
            persist=persist,
            bundle_dir=bundle_path,
            sdk_source_dir=sdk_source_dir,
        )
        if name:
            out[spec.name] = name
    if persist and bundle_path is not None and out:
        persist_builtin_retrieval_index(bundle_path)
    return out


__all__ = [
    "emit_composite_workflow_yaml",
    "register_composite_tool",
    "register_composite_tools",
]
