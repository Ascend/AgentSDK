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

"""Default adapter for :class:`ToolAuthoringProtocol`.

Aggregates the five ``clawcodex_ext.agent.tool_authoring.*`` sub-modules
(persistence, spec, validators, factory, registry_ext) into a single
class that implements the Protocol.

The adapter converts ``AgentToolSpecProtocol`` values to the upstream
``AgentToolSpec`` frozen dataclass before delegating to upstream
functions that expect the concrete type.

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3 and §3.4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from extensions.capabilities.tool_authoring_protocol import (
    AgentToolSpecProtocol,
    ToolAuthoringProtocol,
    ValidationError,
)

__all__ = [
    "DefaultToolAuthoring",
    "spec_from_protocol",
]


def spec_from_protocol(spec: AgentToolSpecProtocol) -> Any:
    """Convert an ``AgentToolSpecProtocol``-compatible value to an upstream ``AgentToolSpec``.

    The upstream ``AgentToolSpec`` is a frozen dataclass that expects
    ``call_type`` as a ``Literal["bash", "http", "python", "workflow"]``
    and ``call_impl`` as ``str | dict`` — both narrower than the Protocol
    signatures.  This function reads the fields and constructs the
    concrete dataclass, raising ``ValidationError`` if the values are
    outside the expected range.
    """
    from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec

    call_type = spec.call_type
    if call_type not in ("bash", "http", "python", "workflow"):
        raise ValidationError(f"Invalid call_type {call_type!r}: expected one of 'bash', 'http', 'python', 'workflow'")

    return AgentToolSpec(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        call_type=call_type,  # type: ignore[arg-type]
        call_impl=spec.call_impl,
        tags=spec.tags,
        aliases=spec.aliases,
        source=spec.source,
        bundle_id=spec.bundle_id,
        stateful_wrapper=spec.stateful_wrapper,
        output_schema=spec.output_schema,
    )


class DefaultToolAuthoring(ToolAuthoringProtocol):
    """Default implementation of :class:`ToolAuthoringProtocol`.

    Delegates to the upstream ``clawcodex_ext.agent.tool_authoring.*``
    sub-modules after converting Protocol values to concrete types where
    necessary.
    """

    # --- persistence ---

    @property
    def TOOL_DIR(self) -> Path:  # type: ignore[override]
        """Default tool storage directory."""
        from clawcodex_ext.agent.tool_authoring.persistence import TOOL_DIR as _TOOL_DIR

        return _TOOL_DIR

    def bundle_tool_dir(self, bundle_path: Path) -> Path:
        """Primary L3 storage: ``<bundle>/agent-tools``."""
        from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir

        return bundle_tool_dir(bundle_path)

    def scripts_dir_for(self, tool_dir: Path) -> Path:
        """Return the ``scripts/`` subdirectory under *tool_dir*."""
        from clawcodex_ext.agent.tool_authoring.persistence import scripts_dir_for

        return scripts_dir_for(tool_dir)

    def save_spec(self, spec: AgentToolSpecProtocol, *, tool_dir: Optional[Path] = None) -> None:
        """Persist a tool spec to disk."""
        from clawcodex_ext.agent.tool_authoring.persistence import save_spec as _save_spec

        _save_spec(spec_from_protocol(spec), tool_dir=tool_dir)

    def list_persisted_specs(self, tool_dir: Optional[Path] = None) -> list[Any]:
        """List all persisted tool specs in *tool_dir* (or the default TOOL_DIR)."""
        from clawcodex_ext.agent.tool_authoring.persistence import (
            list_persisted_specs as _list_persisted_specs,
        )

        return _list_persisted_specs(tool_dir=tool_dir)

    def iter_bundle_tool_dirs(self, bundle_path: Path) -> list[Path]:
        """Iterate over bundle-local tool directories."""
        from clawcodex_ext.agent.tool_authoring.persistence import (
            iter_bundle_tool_dirs as _iter_bundle_tool_dirs,
        )

        return list(_iter_bundle_tool_dirs(bundle_path))

    def create_spec(self, **kwargs: Any) -> Any:
        """Create a concrete ``AgentToolSpec`` from keyword arguments."""
        from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec

        return AgentToolSpec(**kwargs)

    # --- validation ---

    def validate_spec(self, spec: AgentToolSpecProtocol) -> None:
        """Validate an ``AgentToolSpec``-compatible spec.

        Raises :class:`ValidationError` on invalid specs.
        """
        from clawcodex_ext.agent.tool_authoring.validators import (
            validate_spec as _validate_spec,
        )

        _validate_spec(spec_from_protocol(spec))

    # --- factory / registration ---

    def create_and_validate(self, spec: AgentToolSpecProtocol) -> Any:
        """Build a ``Tool`` from a spec and register it.

        Returns the runtime ``Tool`` instance (type ``Any`` per Protocol).
        """
        from clawcodex_ext.agent.tool_authoring.factory import (
            create_and_validate as _create_and_validate,
        )

        return _create_and_validate(spec_from_protocol(spec))

    def add_tool(self, tool: Any) -> None:
        """Register an agent-created tool in the runtime registry."""
        from clawcodex_ext.agent.tool_authoring.registry_ext import add_tool as _add_tool

        _add_tool(tool)
