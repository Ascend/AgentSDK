# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""ToolAuthoring Protocol — interface for SOP-convertible tool authoring.

Aggregates the persistence / spec / validation / factory / registration
surface the SOP converter borrows from ``clawcodex_ext.agent.tool_authoring``
(``persistence``, ``spec``, ``validators``, ``factory``, ``registry_ext``).
Only the methods the SOP converter calls today are surfaced. See
``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    "AgentToolSpecProtocol",
    "ToolAuthoringProtocol",
    "ValidationError",
]


@runtime_checkable
class AgentToolSpecProtocol(Protocol):
    """Minimal contract for an ``AgentToolSpec``-shaped value.

    Mirrors ``clawcodex_ext.agent.tool_authoring.spec.AgentToolSpec``;
    implementations may add fields beyond those listed here.
    """

    name: str
    description: str
    input_schema: dict
    call_type: str
    call_impl: Any
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    source: str
    bundle_id: Optional[str]
    stateful_wrapper: bool
    output_schema: Optional[dict]


class ValidationError(Exception):
    """Raised by :meth:`ToolAuthoringProtocol.validate_spec` on bad specs.

    Mirrors ``clawcodex_ext.agent.tool_authoring.validators.ValidationError``
    so existing callers keep working through an adapter.
    """


@runtime_checkable
class ToolAuthoringProtocol(Protocol):
    """Aggregate boundary for the SOP converter's tool-authoring needs.

    The default implementation lives in
    ``extensions/sop_converter/adapters/tool_authoring_adapter.py``
    (Phase 3+); for now the Protocol is ``@runtime_checkable`` documentation.
    """

    # --- persistence (clawcodex_ext.agent.tool_authoring.persistence) ---
    TOOL_DIR: Path

    def bundle_tool_dir(self, bundle_path: Path) -> Path: ...

    def scripts_dir_for(self, tool_dir: Path) -> Path: ...

    def save_spec(self, spec: AgentToolSpecProtocol, *, tool_dir: Optional[Path] = None) -> None: ...

    def list_persisted_specs(self, tool_dir: Optional[Path] = None) -> list[Any]: ...

    def iter_bundle_tool_dirs(self, bundle_path: Path) -> list[Path]: ...

    def create_spec(self, **kwargs: Any) -> Any:
        """Create a concrete tool spec, duck-compatible with ``AgentToolSpecProtocol``."""

    # --- validation (clawcodex_ext.agent.tool_authoring.validators) ---
    def validate_spec(self, spec: AgentToolSpecProtocol) -> None: ...

    # --- factory / registration ---
    # ``create_and_validate`` returns the runtime ``Tool`` instance — typed as
    # ``Any`` because the SOP converter only registers it via ``add_tool``.
    def create_and_validate(self, spec: AgentToolSpecProtocol) -> Any: ...

    def add_tool(self, tool: Any) -> None: ...
