#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.

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

"""In-process macro catalog used by ``call_type=workflow``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ..composite_runtime import CompositeWorkflowSpec


def _contains_private_binding(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("$private.")
    if isinstance(value, dict):
        return any(_contains_private_binding(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_private_binding(child) for child in value)
    return False


def _validate_catalog_spec(catalog_id: str, spec: CompositeWorkflowSpec) -> None:
    scope = catalog_id.split(":", 1)[0]
    if scope not in {"builtin", "bundle", "session"}:
        raise ValueError(f"unsupported macro catalog scope: {scope}")
    if scope == "builtin":
        return
    if spec.trusted:
        raise ValueError("only builtin macros may be trusted")
    for step in spec.steps:
        if step.kind != "tool" or step.visibility != "public":
            raise ValueError("bundle/session macros may contain only public tool steps")
        if _contains_private_binding(step.args):
            raise ValueError("bundle/session macros cannot reference private bindings")
    if _contains_private_binding(spec.outputs):
        raise ValueError("bundle/session macros cannot expose private bindings")


@dataclass
class MacroCatalog:
    """Resolve trusted builtin and session workflow definitions by ID."""

    _specs: dict[str, CompositeWorkflowSpec] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def register(self, catalog_id: str, spec: CompositeWorkflowSpec, *, replace: bool = False) -> None:
        if not catalog_id or ":" not in catalog_id:
            raise ValueError("macro catalog_id must include a scope prefix")
        _validate_catalog_spec(catalog_id, spec)
        with self._lock:
            if not replace and catalog_id in self._specs:
                raise ValueError(f"macro already registered: {catalog_id}")
            self._specs[catalog_id] = spec

    def get(self, catalog_id: str) -> CompositeWorkflowSpec | None:
        with self._lock:
            return self._specs.get(catalog_id)

    def require(self, catalog_id: str) -> CompositeWorkflowSpec:
        spec = self.get(catalog_id)
        if spec is None:
            raise KeyError(catalog_id)
        return spec

    def clear(self) -> None:
        with self._lock:
            self._specs.clear()

    def unregister(self, catalog_id: str) -> None:
        with self._lock:
            self._specs.pop(catalog_id, None)


DEFAULT_MACRO_CATALOG = MacroCatalog()


def ensure_builtin_macros(catalog: MacroCatalog | None = None) -> MacroCatalog:
    target = catalog or DEFAULT_MACRO_CATALOG
    builtins = {
        "builtin:invoke-existing-agent": "invoke_existing_agent_workflow",
        "builtin:resume-resource": "resume_resource_workflow",
    }
    missing = [catalog_id for catalog_id in builtins if target.get(catalog_id) is None]
    if missing:
        from .. import composite_workflows

        for catalog_id in missing:
            factory = getattr(composite_workflows, builtins[catalog_id], None)
            if factory is None:
                raise KeyError(f"builtin macro factory {builtins[catalog_id]!r} missing from composite_workflows")
            target.register(catalog_id, factory())
    return target


def register_macro(
    catalog_id: str,
    spec: CompositeWorkflowSpec,
    *,
    catalog: MacroCatalog | None = None,
    replace: bool = False,
) -> None:
    (catalog or DEFAULT_MACRO_CATALOG).register(catalog_id, spec, replace=replace)


def resolve_macro(
    call_impl: dict[str, Any],
    *,
    catalog: MacroCatalog | None = None,
    bundle_path: Any | None = None,
    session_overlay: Any | None = None,
    owner_session_id: str | None = None,
) -> CompositeWorkflowSpec:
    catalog_id = str(call_impl.get("catalog_id") or "")
    manifest = str(call_impl.get("manifest") or "").strip()
    target = catalog or DEFAULT_MACRO_CATALOG

    if catalog_id.startswith("session:"):
        if session_overlay is None or not hasattr(session_overlay, "read"):
            raise KeyError(f"session macro requires overlay: {catalog_id}")
        snapshot = session_overlay.read()
        if snapshot is None:
            raise KeyError(f"session macro overlay empty: {catalog_id}")
        spec = snapshot.specs.get(catalog_id.lower())
        if spec is None:
            raise KeyError(f"session macro not in snapshot: {catalog_id}")
        session_id = str(owner_session_id or "").strip()
        if not session_id or session_id != snapshot.owner_session_id:
            raise KeyError(f"session macro owner mismatch: {catalog_id}")
        return spec

    if catalog_id:
        if catalog_id in {
            "builtin:invoke-existing-agent",
            "builtin:resume-resource",
        }:
            ensure_builtin_macros(target)
        try:
            return target.require(catalog_id)
        except KeyError:
            if not catalog_id.startswith("bundle:"):
                raise
            # Fall through to disk load for bundle macros registered only as files.
            name = catalog_id.split(":", 1)[1]
            manifest = manifest or f".clawcodex/macros/{name}.yaml"

    if not manifest:
        raise KeyError("workflow call_impl requires catalog_id or manifest")

    from pathlib import Path

    from .loader import load_macro_yaml
    from .validation import validate_macro_definition

    root: Path | None = None
    if bundle_path is not None:
        root = Path(bundle_path)
    else:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            active = get_active_bundle()
            if active is not None:
                root = Path(active.bundle_path)
        except Exception:
            root = None
        if root is None:
            env = os.environ.get("CLAWCODEX_BUNDLE_PATH", "").strip()
            if env:
                root = Path(env)
    if root is None:
        raise KeyError(f"cannot resolve workflow manifest without bundle path: {manifest}")

    root_resolved = root.resolve()
    path = (root_resolved / manifest).resolve()
    if not (str(path) == str(root_resolved) or str(path).startswith(str(root_resolved) + os.sep)):
        raise KeyError(f"workflow manifest escapes bundle: {manifest}")
    macro = load_macro_yaml(path)
    spec = validate_macro_definition(macro, tool_index=None)
    # Cache for subsequent calls in-process
    catalog_key = f"bundle:{macro.routing.target_tool or macro.name}"
    try:
        target.register(catalog_key, spec, replace=True)
    except ValueError:
        pass
    return spec


__all__ = [
    "DEFAULT_MACRO_CATALOG",
    "MacroCatalog",
    "ensure_builtin_macros",
    "register_macro",
    "resolve_macro",
]
