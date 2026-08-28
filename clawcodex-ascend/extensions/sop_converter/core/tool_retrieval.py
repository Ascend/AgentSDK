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

"""Structural metadata for layered macro / atomic ToolSearch.

The persisted index is a compiled runtime view.  Macro manifests remain the
authoritative source for ``intent_key`` / ``covered_tools``; the index makes
those relationships cheap and deterministic to consume during ToolSearch.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

RETRIEVAL_INDEX_VERSION = 1
RETRIEVAL_INDEX_RELATIVE_PATH = Path(".clawcodex") / "tool-retrieval.yaml"


def normalize_tool_ref(value: str) -> str:
    """Normalize dotted, underscored and kebab tool references."""

    return re.sub(r"[._\-]+", "-", str(value or "").strip()).strip("-").lower()


def _matches_reference(tool_name: str, reference: str) -> bool:
    name_lower = str(tool_name or "").strip().lower()
    ref_lower = str(reference or "").strip().lower()
    if not name_lower or not ref_lower:
        return False
    if name_lower == ref_lower:
        return True
    normalized_name = normalize_tool_ref(name_lower)
    normalized_ref = normalize_tool_ref(ref_lower)
    if normalized_name == normalized_ref:
        return True
    return bool(normalized_ref) and normalized_name.endswith(f"-{normalized_ref}")


def resolve_tool_references(
    references: Iterable[str],
    tool_names: Iterable[str],
    *,
    require_unique: bool = False,
) -> list[str]:
    """Resolve route references to canonical tool names.

    Resolution order is exact name, normalized exact name, then unique/all
    normalized suffix matches.  Convert uses ``require_unique=True``; runtime
    uses all matches so a builtin safety route can shadow namespaced variants.
    """

    available = [str(name) for name in tool_names if str(name).strip()]
    resolved: list[str] = []
    for raw_ref in references:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        exact = [name for name in available if name.lower() == ref.lower()]
        if exact:
            matches = exact
        else:
            ref_norm = normalize_tool_ref(ref)
            normalized_exact = [name for name in available if normalize_tool_ref(name) == ref_norm]
            matches = normalized_exact or [name for name in available if _matches_reference(name, ref)]
        matches = list(dict.fromkeys(matches))
        if require_unique and len(matches) != 1:
            if not matches:
                raise ValueError(f"covered tool reference not found: {ref}")
            raise ValueError(f"covered tool reference is ambiguous: {ref} -> {', '.join(matches)}")
        for name in matches:
            if name not in resolved:
                resolved.append(name)
    return resolved


@dataclass
class ToolRetrievalProfile:
    name: str
    layer: Literal["macro", "atomic", "neutral"] = "neutral"
    source: str = ""
    call_type: str = ""
    intent_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "source": self.source,
            "call_type": self.call_type,
            "intent_keys": list(self.intent_keys),
        }

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "ToolRetrievalProfile":
        layer = str(payload.get("layer") or "neutral")
        if layer not in ("macro", "atomic", "neutral"):
            layer = "neutral"
        return cls(
            name=name,
            layer=layer,  # type: ignore[arg-type]
            source=str(payload.get("source") or ""),
            call_type=str(payload.get("call_type") or ""),
            intent_keys=[str(key) for key in (payload.get("intent_keys") or []) if str(key).strip()],
        )


@dataclass
class MacroCoverage:
    intent_key: str
    macro_tool: str
    covered_tools: list[str] = field(default_factory=list)
    selection: Literal["exclusive", "prefer"] = "prefer"
    verified: bool = False
    unavailable_policy: Literal["restore-covered"] = "restore-covered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_key": self.intent_key,
            "macro_tool": self.macro_tool,
            "covered_tools": list(self.covered_tools),
            "selection": self.selection,
            "verified": self.verified,
            "unavailable_policy": self.unavailable_policy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MacroCoverage":
        selection = str(payload.get("selection") or "prefer")
        if selection not in ("exclusive", "prefer"):
            selection = "prefer"
        return cls(
            intent_key=str(payload.get("intent_key") or ""),
            macro_tool=str(payload.get("macro_tool") or ""),
            covered_tools=[str(name) for name in (payload.get("covered_tools") or []) if str(name).strip()],
            selection=selection,  # type: ignore[arg-type]
            verified=bool(payload.get("verified", False)),
            unavailable_policy="restore-covered",
        )


@dataclass
class ToolRetrievalIndex:
    version: int = RETRIEVAL_INDEX_VERSION
    tools: dict[str, ToolRetrievalProfile] = field(default_factory=dict)
    coverage: list[MacroCoverage] = field(default_factory=list)

    def profile_for(self, tool_name: str) -> ToolRetrievalProfile | None:
        direct = self.tools.get(tool_name)
        if direct is not None:
            return direct
        normalized = normalize_tool_ref(tool_name)
        for name, profile in self.tools.items():
            if normalize_tool_ref(name) == normalized:
                return profile
        return None

    def coverage_for_macro(self, macro_tool: str) -> MacroCoverage | None:
        normalized = normalize_tool_ref(macro_tool)
        for item in self.coverage:
            if normalize_tool_ref(item.macro_tool) == normalized:
                return item
        return None

    def covered_names(self, macro_tool: str, tool_names: Iterable[str]) -> list[str]:
        item = self.coverage_for_macro(macro_tool)
        if item is None:
            return []
        return resolve_tool_references(item.covered_tools, tool_names)

    def merge(self, overlay: "ToolRetrievalIndex") -> "ToolRetrievalIndex":
        if overlay.version != RETRIEVAL_INDEX_VERSION:
            return self
        merged_tools = dict(self.tools)
        for name, profile in overlay.tools.items():
            existing = merged_tools.get(name)
            if existing is None:
                merged_tools[name] = profile
                continue
            merged_tools[name] = ToolRetrievalProfile(
                name=name,
                layer=profile.layer if profile.layer != "neutral" else existing.layer,
                source=profile.source or existing.source,
                call_type=profile.call_type or existing.call_type,
                intent_keys=list(dict.fromkeys([*existing.intent_keys, *profile.intent_keys])),
            )
        overlay_macros = {normalize_tool_ref(item.macro_tool) for item in overlay.coverage}
        merged_coverage = [item for item in self.coverage if normalize_tool_ref(item.macro_tool) not in overlay_macros]
        merged_coverage.extend(overlay.coverage)
        return ToolRetrievalIndex(tools=merged_tools, coverage=merged_coverage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tools": {name: profile.to_dict() for name, profile in self.tools.items()},
            "coverage": [item.to_dict() for item in self.coverage],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolRetrievalIndex":
        try:
            version = int(payload.get("version", RETRIEVAL_INDEX_VERSION))
        except (TypeError, ValueError):
            version = 0
        if version != RETRIEVAL_INDEX_VERSION:
            raise ValueError(f"unsupported tool retrieval index version: {version}")
        tools_raw = payload.get("tools") or {}
        tools: dict[str, ToolRetrievalProfile] = {}
        if isinstance(tools_raw, dict):
            for name, body in tools_raw.items():
                if isinstance(body, dict):
                    tools[str(name)] = ToolRetrievalProfile.from_dict(str(name), body)
        coverage_raw = payload.get("coverage") or []
        coverage = [MacroCoverage.from_dict(body) for body in coverage_raw if isinstance(body, dict)]
        return cls(version=version, tools=tools, coverage=coverage)


def retrieval_index_path(bundle_path: Path | str) -> Path:
    return Path(bundle_path) / RETRIEVAL_INDEX_RELATIVE_PATH


def load_tool_retrieval_index(bundle_path: Path | str) -> ToolRetrievalIndex:
    path = retrieval_index_path(bundle_path)
    if not path.is_file():
        return ToolRetrievalIndex()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - same dependency as macros
        raise ValueError("PyYAML is required to load tool retrieval metadata") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tool retrieval index must be a mapping")
    return ToolRetrievalIndex.from_dict(payload)


def write_tool_retrieval_index(index: ToolRetrievalIndex, bundle_path: Path | str) -> Path:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - same dependency as macros
        raise ValueError("PyYAML is required to write tool retrieval metadata") from exc
    path = retrieval_index_path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tool-retrieval.",
        suffix=".yaml.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(index.to_dict(), handle, allow_unicode=True, sort_keys=False)
        os.replace(tmp_name, path)
    except Exception:
        # Close the raw fd in case os.fdopen raised before taking ownership;
        # if the with-block already closed it, os.close raises EBADF which is
        # safely ignored.
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def index_from_routes(
    routes: Iterable[Any],
    tool_names: Iterable[str],
    *,
    tool_specs: Iterable[Any] = (),
    require_unique: bool = False,
) -> ToolRetrievalIndex:
    """Compile runtime profiles and coverage from MacroRoute objects."""

    names = [str(name) for name in tool_names if str(name).strip()]
    specs_by_name = {
        str(getattr(spec, "name", "")): spec for spec in tool_specs if str(getattr(spec, "name", "")).strip()
    }
    index = ToolRetrievalIndex()
    macro_names = {
        str(getattr(route, "target_tool", "")) for route in routes if str(getattr(route, "target_tool", "")).strip()
    }
    for route in routes:
        macro_tool = str(getattr(route, "target_tool", "") or "").strip()
        if not macro_tool:
            continue
        intent_key = str(getattr(route, "intent_key", "") or "").strip()
        macro_spec = specs_by_name.get(macro_tool)
        source = str(getattr(macro_spec, "source", "") or "")
        call_type = str(getattr(macro_spec, "call_type", "") or "")
        if not source and str(getattr(route, "scope", "")) == "builtin":
            source = "composite-tool"
        if not call_type and str(getattr(route, "scope", "")) == "builtin":
            call_type = "workflow"
        index.tools[macro_tool] = ToolRetrievalProfile(
            name=macro_tool,
            layer="macro",
            source=source,
            call_type=call_type,
            intent_keys=[intent_key] if intent_key else [],
        )

        covered_refs = [str(name) for name in (getattr(route, "covered_tools", None) or []) if str(name).strip()]
        if not intent_key or not covered_refs:
            continue
        resolved = resolve_tool_references(
            covered_refs,
            names,
            require_unique=require_unique,
        )
        for atomic_name in resolved:
            if atomic_name in macro_names:
                raise ValueError(f"macro may not cover another macro: {atomic_name}")
            atomic_spec = specs_by_name.get(atomic_name)
            profile = index.tools.get(atomic_name) or ToolRetrievalProfile(
                name=atomic_name,
                layer="atomic",
                source=str(getattr(atomic_spec, "source", "") or ""),
                call_type=str(getattr(atomic_spec, "call_type", "") or ""),
            )
            profile.layer = "atomic"
            if intent_key not in profile.intent_keys:
                profile.intent_keys.append(intent_key)
            index.tools[atomic_name] = profile
        index.coverage.append(
            MacroCoverage(
                intent_key=intent_key,
                macro_tool=macro_tool,
                covered_tools=resolved or covered_refs,
                selection=str(getattr(route, "selection", "prefer")),  # type: ignore[arg-type]
                verified=bool(getattr(route, "verified", False)),
                unavailable_policy="restore-covered",
            )
        )
    return index


__all__ = [
    "MacroCoverage",
    "RETRIEVAL_INDEX_RELATIVE_PATH",
    "RETRIEVAL_INDEX_VERSION",
    "ToolRetrievalIndex",
    "ToolRetrievalProfile",
    "index_from_routes",
    "load_tool_retrieval_index",
    "normalize_tool_ref",
    "resolve_tool_references",
    "retrieval_index_path",
    "write_tool_retrieval_index",
]
