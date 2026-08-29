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

"""Runtime bundle isolation — L2 skill/tool registry + L3 per-bundle storage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from extensions.capabilities.agent_definition_protocol import AgentToolConstants
from ..core.bundle_manifest import resolve_sdk_source_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_active_bundle: BundleContext | None = None

# Base tools always visible in bundle mode (matches startup overview allowlist).
_BUNDLE_BASE_TOOL_NAMES: frozenset[str] = frozenset(AgentToolConstants.registered_proxy_base_tools())

# ``pos convert`` wrote ``pos-converter``; ``sop convert`` writes ``sop-converter``.
SOP_CONVERTER_SPEC_SOURCES = frozenset({"pos-converter", "sop-converter"})
SOP_BUNDLE_SPEC_SOURCES = SOP_CONVERTER_SPEC_SOURCES | frozenset(
    {"composite-tool", "sop-converter-composite", "sop-converter-macro"}
)


def is_sop_converter_spec_source(source: str | None) -> bool:
    return source in SOP_BUNDLE_SPEC_SOURCES


@dataclass(frozen=True)
class BundleContext:
    """Session-scoped isolation boundary for a POS agent bundle."""

    bundle_path: Path
    bundle_name: str
    skill_names: frozenset[str]
    skill_dirs: tuple[Path, ...]
    tool_names: frozenset[str]
    sdk_source_dir: Path | None = None


def set_active_bundle(context: BundleContext | None) -> None:
    global _active_bundle
    _active_bundle = context


def get_active_bundle() -> BundleContext | None:
    return _active_bundle


def build_bundle_context(
    *,
    bundle_path: Path,
    skill_names: list[str],
    skill_dirs: list[Path],
    tool_names: list[str],
    sdk_source_dir: Path | None = None,
    workspace_root: Path | None = None,
) -> BundleContext:
    resolved_sdk = sdk_source_dir
    if resolved_sdk is None:
        resolved_sdk = resolve_sdk_source_dir(bundle_path, workspace_root=workspace_root)
    return BundleContext(
        bundle_path=bundle_path.resolve(),
        bundle_name=bundle_path.name,
        skill_names=frozenset(skill_names),
        skill_dirs=tuple(p.resolve() for p in skill_dirs),
        tool_names=frozenset(tool_names),
        sdk_source_dir=resolved_sdk.resolve() if resolved_sdk is not None else None,
    )


def _same_resolved_path(a: Path, b: Path) -> bool:
    try:
        return a.expanduser().resolve() == b.expanduser().resolve()
    except OSError:
        return False


def apply_sdk_source_working_directory(tool_context: Any, bundle: BundleContext) -> Path | None:
    """Add ``bundle.sdk_source_dir`` to tool working-directory allowlist.

    Enables Bash ``cd`` and Read/Glob/Grep under the SDK tree recorded in
    ``bundle.json``, matching ``sop_exploration_guard`` expectations.
    """
    sdk = bundle.sdk_source_dir
    if sdk is None:
        return None
    try:
        resolved = sdk.expanduser().resolve()
    except OSError:
        logger.warning("bundle sdk_source_dir cannot be resolved: %s", sdk)
        return None
    if not resolved.is_dir():
        logger.debug("bundle sdk_source_dir is not a directory: %s", resolved)
        return None

    existing: tuple[Path, ...] = getattr(tool_context, "additional_working_directories", ()) or ()
    if any(_same_resolved_path(root, resolved) for root in existing):
        return resolved

    tool_context.additional_working_directories = (*existing, resolved)
    logger.info("Added SDK source to working directories: %s", resolved)
    return resolved


def collect_tool_names_from_skills(skills: list[Any]) -> list[str]:
    names: dict[str, bool] = {}
    for skill in skills:
        for tool in getattr(skill, "allowed_tools", None) or []:
            if isinstance(tool, str) and tool:
                names[tool] = True
    return sorted(names)


def collect_tool_names_from_bundle_specs(bundle_path: Path) -> list[str]:
    """Fallback allowlist from persisted ``pos convert --register-tools`` specs."""
    from ..adapters import DEFAULTS

    names: dict[str, bool] = {}
    for tool_dir in (DEFAULTS.tool_authoring.bundle_tool_dir(bundle_path),):
        if not tool_dir.is_dir():
            continue
        for spec in DEFAULTS.tool_authoring.list_persisted_specs(tool_dir=tool_dir):
            if spec.source in SOP_BUNDLE_SPEC_SOURCES and spec.name:
                names[spec.name] = True
    return sorted(names)


def is_pos_converter_tool(tool: Any) -> bool:
    return bool(getattr(tool, "should_defer", False))


def _spec_allowed_for_bundle(spec: Any, allowed_names: frozenset[str]) -> bool:
    """Whether a persisted spec is on the bundle allowlist (name or alias)."""
    if spec.name in allowed_names:
        return True
    for alias in getattr(spec, "aliases", ()) or ():
        if alias in allowed_names:
            return True
    return False


def _tool_in_bundle_allowlist(tool: Any, bundle: BundleContext) -> bool:
    from clawcodex_ext.tool_system.build_tool import tool_matches_name

    return any(tool_matches_name(tool, name) for name in bundle.tool_names)


def filter_tools_for_bundle(tools: list[Any], bundle: BundleContext | None = None) -> list[Any]:
    """Keep base tools + deferred SDK tools that belong to the active bundle.

    Session-macro provenance tools are re-appended after the allowlist filter
    so overlay tools remain visible even when not listed on the bundle.
    """
    from clawcodex_ext.tool_system.build_tool import tool_matches_name

    bundle = bundle or get_active_bundle()
    if bundle is None:
        return tools

    filtered: list[Any] = []
    for tool in tools:
        if tool.is_mcp or tool.name.startswith("mcp__"):
            filtered.append(tool)
            continue
        if any(tool_matches_name(tool, name) for name in _BUNDLE_BASE_TOOL_NAMES):
            filtered.append(tool)
            continue
        if is_pos_converter_tool(tool):
            if any(tool_matches_name(tool, name) for name in bundle.tool_names):
                filtered.append(tool)
            continue
        # Non-deferred agent-created tools: keep only if listed on the bundle.
        if any(tool_matches_name(tool, name) for name in bundle.tool_names):
            filtered.append(tool)

    # Re-append session macros dropped by the allowlist filter.
    try:
        from .macros.session import is_session_macro_tool
    except ImportError:
        return filtered

    present_ids = {id(tool) for tool in filtered}
    for tool in tools:
        if not is_session_macro_tool(tool):
            continue
        if id(tool) in present_ids:
            continue
        filtered.append(tool)
        present_ids.add(id(tool))
    return filtered


def _bundle_deferred_tools_loaded(registry: Any, bundle: BundleContext) -> bool:
    """Return True when every bundle tool is already registered."""

    sample = [name for name in bundle.tool_names if name]
    if not sample:
        return any(getattr(tool, "should_defer", False) for tool in registry.list_tools())
    return all(registry.get(name) is not None for name in sample)


def load_bundle_persisted_tools(registry: Any, bundle_path: Path) -> int:
    """Load SDK tool specs from bundle-local storage and register them."""
    from ..adapters import DEFAULTS

    bundle_name = bundle_path.name
    bundle = get_active_bundle()
    if bundle is not None and _bundle_deferred_tools_loaded(registry, bundle):
        return 0
    allowed_names = bundle.tool_names if bundle is not None else frozenset()

    loaded = 0
    seen: set[str] = set()

    ta = DEFAULTS.tool_authoring
    search_dirs = list(ta.iter_bundle_tool_dirs(bundle_path))
    search_dirs.append(ta.TOOL_DIR)

    for tool_dir in search_dirs:
        for spec in ta.list_persisted_specs(tool_dir=tool_dir):
            if spec.name in seen:
                continue
            if tool_dir == ta.TOOL_DIR:
                if spec.bundle_id and spec.bundle_id != bundle_name:
                    continue
                if not spec.bundle_id and allowed_names and not _spec_allowed_for_bundle(spec, allowed_names):
                    continue
                if not is_sop_converter_spec_source(spec.source):
                    continue
            elif spec.bundle_id and spec.bundle_id != bundle_name:
                continue
            if _register_persisted_spec(registry, spec, seen=seen):
                loaded += 1

    return loaded


def _register_persisted_spec(
    registry: Any,
    spec: Any,
    *,
    seen: set[str] | None = None,
) -> bool:
    """Register one persisted spec; return True when newly registered."""
    from ..adapters import DEFAULTS

    if seen is not None and spec.name in seen:
        return False
    try:
        tool = DEFAULTS.tool_authoring.create_and_validate(spec)
    except Exception as exc:
        logger.warning("Failed to load bundle tool %s: %s", spec.name, exc)
        return False
    existing = registry.get(spec.name)
    if existing is not None:
        registry.unregister(spec.name)
    try:
        registry.register(tool)
    except ValueError as exc:
        if "duplicate tool" in str(exc):
            logger.warning(
                "Skipping bundle tool %s due to registry conflict: %s",
                spec.name,
                exc,
            )
            return False
        raise
    DEFAULTS.tool_authoring.add_tool(tool)
    if seen is not None:
        seen.add(spec.name)
    return True


def ensure_bundle_tools_registered(
    registry: Any,
    tool_names: list[str],
    *,
    bundle_path: Path | None = None,
) -> int:
    """Load persisted specs for *tool_names* that are missing from *registry*."""

    if not tool_names:
        return 0

    bundle = get_active_bundle()
    path = bundle_path or (bundle.bundle_path if bundle is not None else None)
    if path is None:
        return 0

    needed = {name for name in tool_names if name and registry.get(name) is None}
    if not needed:
        return 0

    from ..adapters import DEFAULTS

    bundle_name = path.name
    allowed_names = bundle.tool_names if bundle is not None else frozenset()
    loaded = 0

    ta = DEFAULTS.tool_authoring
    search_dirs = list(ta.iter_bundle_tool_dirs(path))
    search_dirs.append(ta.TOOL_DIR)

    for tool_dir in search_dirs:
        for spec in ta.list_persisted_specs(tool_dir=tool_dir):
            if spec.name not in needed:
                continue
            if tool_dir == ta.TOOL_DIR:
                if spec.bundle_id and spec.bundle_id != bundle_name:
                    continue
                if not spec.bundle_id and allowed_names and not _spec_allowed_for_bundle(spec, allowed_names):
                    continue
                if not is_sop_converter_spec_source(spec.source):
                    continue
            elif spec.bundle_id and spec.bundle_id != bundle_name:
                continue
            if _register_persisted_spec(registry, spec):
                loaded += 1
                needed.discard(spec.name)
        if not needed:
            break

    if needed:
        logger.debug(
            "Bundle tool specs not found for ToolSearch matches: %s",
            sorted(needed)[:8],
        )
    return loaded


def prune_registry_to_bundle(registry: Any, bundle: BundleContext) -> int:
    """Remove deferred tools from the registry that are outside the bundle."""

    removed = 0
    for tool in list(registry.list_tools()):
        if not is_pos_converter_tool(tool):
            continue
        if _tool_in_bundle_allowlist(tool, bundle):
            continue
        if registry.unregister(tool.name):
            removed += 1
    return removed


def activate_bundle_isolation(
    registry: Any,
    bundle: BundleContext,
) -> None:
    """Apply L2/L3 isolation: load bundle tools, prune foreign deferred tools."""

    set_active_bundle(bundle)
    loaded = load_bundle_persisted_tools(registry, bundle.bundle_path)
    removed = prune_registry_to_bundle(registry, bundle)
    if loaded or removed:
        logger.info(
            "Bundle isolation %s: loaded %d tools, pruned %d foreign deferred tools",
            bundle.bundle_name,
            loaded,
            removed,
        )


def load_bundle_macro_routes(bundle_path: Path | str) -> list[Any]:
    """Load MacroRoute entries from ``<bundle>/.clawcodex/macros/*.yaml``.

    Each macro definition may embed a ``routing:`` block.
    Routes are tagged ``scope="bundle"``. Missing directory yields ``[]``.

    Parses routing via :func:`macros.loader.parse_macro_route` (route-only;
    does not require a ``workflow:`` block like :func:`load_macro_yaml`).
    """
    from .macros.loader import parse_macro_route
    from .macros.models import MacroRoute

    root = Path(bundle_path)
    macros_dir = root / ".clawcodex" / "macros"
    if not macros_dir.is_dir():
        return []

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available; cannot load bundle macro routes from %s", macros_dir)
        return []

    routes: list[MacroRoute] = []
    paths = sorted(macros_dir.glob("*.yaml")) + sorted(macros_dir.glob("*.yml"))
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to load macro route file %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        if data.get("enabled", True) is False:
            continue
        routing = data.get("routing") if isinstance(data.get("routing"), dict) else {}
        route = parse_macro_route(
            routing,
            default_target=str(data.get("name") or ""),
        )
        if not route.target_tool:
            continue
        route.scope = "bundle"
        routes.append(route)
    return routes
