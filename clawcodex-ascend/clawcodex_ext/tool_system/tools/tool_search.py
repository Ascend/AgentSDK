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

from __future__ import annotations

import json
from typing import Any
from ..context import RetrievalPlan, ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolRegistry
from ..build_tool import Tool, build_tool, tool_matches_name
from .tool_search_matching import (
    rank_tool_matches,
    resolve_select_tool_names,
    summarize_tool_match,
)


def _resolve_bundle_context(context: ToolContext) -> Any | None:
    bundle = getattr(context, "bundle_context", None)
    if bundle is not None:
        return bundle
    try:
        from extensions.sop_converter.bundle_context import get_active_bundle

        return get_active_bundle()
    except ImportError:
        return None


def _allowlist_search_stub(name: str) -> Tool:
    """Minimal deferred tool used only for ToolSearch indexing."""
    hint = name.replace("-", " ").replace("_", " ")

    def _unloaded_call(_input: dict[str, Any], _context: ToolContext) -> ToolResult:
        return ToolResult(
            name=name,
            output={"error": f"tool not loaded: {name}"},
            is_error=True,
        )

    return build_tool(
        name=name,
        input_schema={"type": "object", "properties": {}},
        call=_unloaded_call,
        prompt=hint,
        search_hint=hint,
        should_defer=True,
    )


def _bundle_scoped_tools(registry: ToolRegistry, context: ToolContext) -> list[Tool]:
    bundle = _resolve_bundle_context(context)

    try:
        from extensions.sop_converter.bundle_context import filter_tools_for_bundle

        filtered = filter_tools_for_bundle(registry.list_tools(), bundle)
    except ImportError:
        return registry.list_tools()

    if bundle is None:
        return filtered

    # Allowlisted bundle tools may not be registered yet (lazy persistence).
    # Include loaded tools and lightweight name stubs so exact ToolSearch
    # queries resolve to the intended deferred tool instead of base-tool noise.
    extras: list[Tool] = []
    for name in bundle.tool_names:
        if any(tool_matches_name(tool, name) for tool in filtered):
            continue
        loaded = registry.get(name)
        if loaded is not None:
            extras.append(loaded)
            continue
        extras.append(_allowlist_search_stub(name))
    return filtered + extras


def _load_lifecycle_graph(context: ToolContext) -> Any | None:
    bundle = _resolve_bundle_context(context)
    if bundle is None:
        return None
    try:
        from extensions.sop_converter.dependency import load_tool_dependencies

        return load_tool_dependencies(bundle.bundle_path)
    except ImportError:
        return None
    except Exception:  # nosec
        return None


def _load_macro_route_catalog(context: ToolContext) -> Any | None:
    """Load macro route catalog from session, bundle, and builtin scopes."""
    try:
        from extensions.sop_converter.runtime.macros.routing import (
            DEFAULT_MACRO_ROUTE_CATALOG,
            ensure_builtin_routes,
            MacroRouteCatalog,
        )

        ensure_builtin_routes(DEFAULT_MACRO_ROUTE_CATALOG)

        catalog = MacroRouteCatalog()

        # Load order then match rank: session > bundle > builtin (§8.3 / §8.4).
        # Register builtin first; later scopes replace by target_tool when needed.
        for route in DEFAULT_MACRO_ROUTE_CATALOG.get_routes():
            if getattr(route, "scope", None) != "builtin":
                route.scope = "builtin"
            catalog.register_route(route, replace=True)
        protected_builtin_targets = {
            route.target_tool
            for route in catalog.get_routes()
            if route.scope == "builtin" and route.selection == "exclusive" and route.verified
        }

        bundle = _resolve_bundle_context(context)
        if bundle is not None:
            from extensions.sop_converter.bundle_context import load_bundle_macro_routes

            for route in load_bundle_macro_routes(bundle.bundle_path):
                if route.target_tool in protected_builtin_targets:
                    continue
                route.scope = "bundle"
                catalog.register_route(route, replace=True)

        session_routes = None
        overlay = getattr(context, "session_macro_overlay", None)
        if overlay is not None:
            snapshot = overlay.read()
            if (
                snapshot is not None
                and getattr(context, "session_id", None)
                and context.session_id == snapshot.owner_session_id
            ):
                session_routes = snapshot.routes
        if session_routes is not None:
            for route in session_routes:
                if route.target_tool in protected_builtin_targets:
                    continue
                route.scope = "session"
                catalog.register_route(route, replace=True)

        return catalog
    except ImportError:
        return None


def _load_retrieval_index(
    context: ToolContext,
    searchable_tools: list[Tool],
    macro_route_catalog: Any | None,
) -> Any | None:
    """Merge persisted bundle metadata with the active route overlay."""

    try:
        from extensions.sop_converter.tool_retrieval import (
            ToolRetrievalIndex,
            index_from_routes,
            load_tool_retrieval_index,
        )

        index = ToolRetrievalIndex()
        bundle = _resolve_bundle_context(context)
        if bundle is not None:
            try:
                index = load_tool_retrieval_index(bundle.bundle_path)
            except (OSError, ValueError):
                # Invalid optional metadata must degrade to route + normal
                # search rather than make ToolSearch unavailable.
                index = ToolRetrievalIndex()
        routes = macro_route_catalog.get_routes() if macro_route_catalog is not None else []
        route_index = index_from_routes(
            routes,
            [tool.name for tool in searchable_tools],
            require_unique=False,
        )
        return index.merge(route_index)
    except (ImportError, ValueError):
        return None


def _find_persisted_macro_spec(bundle_path: Any, name: str) -> Any | None:
    if bundle_path is None:
        return None
    try:
        from clawcodex_ext.agent.tool_authoring.persistence import (
            TOOL_DIR,
            iter_bundle_tool_dirs,
            list_persisted_specs,
        )

        search_dirs = list(iter_bundle_tool_dirs(bundle_path))
        search_dirs.append(TOOL_DIR)
        for tool_dir in search_dirs:
            for spec in list_persisted_specs(tool_dir=tool_dir):
                if spec.name != name:
                    continue
                if spec.bundle_id and spec.bundle_id != bundle_path.name:
                    continue
                return spec
    except (ImportError, OSError):
        return None
    return None


def _preflight_macro(
    registry: ToolRegistry,
    context: ToolContext,
    target_tool: str,
) -> tuple[bool, str]:
    """Verify that a macro can be activated without executing its workflow."""
    from extensions.sop_converter.runtime.macros.resolve_tool import (
        resolve_tool_for_context,
    )

    _activate_toolsearch_matches(registry, context, [target_tool])
    tool = resolve_tool_for_context(context, target_tool, base_registry=registry)
    if tool is None:
        return False, "macro_tool_missing"
    try:
        if not tool.is_enabled():
            return False, "macro_tool_disabled"
    except Exception:  # nosec
        return False, "macro_tool_disabled"

    bundle = _resolve_bundle_context(context)
    bundle_path = getattr(bundle, "bundle_path", None)
    spec = _find_persisted_macro_spec(bundle_path, target_tool)
    if spec is None or getattr(spec, "call_type", None) != "workflow":
        # Builtins and directly registered test/runtime macros do not require a
        # persisted spec.  Presence + enabled state is enough for preflight.
        return True, "macro_ready"
    try:
        from extensions.sop_converter.runtime.macros import resolve_macro

        workflow = resolve_macro(
            dict(spec.call_impl),
            bundle_path=bundle_path,
            session_overlay=getattr(context, "session_macro_overlay", None),
            owner_session_id=getattr(context, "session_id", None),
        )
    except Exception:  # nosec
        return False, "macro_workflow_unresolved"

    allowlisted = set(getattr(bundle, "tool_names", None) or [])
    for step in getattr(workflow, "steps", ()):
        if getattr(step, "kind", "") != "tool":
            continue
        ref = str(getattr(step, "callable_ref", "") or "")
        if not ref:
            return False, "macro_step_unresolved"
        if resolve_tool_for_context(context, ref, base_registry=registry) is None and ref not in allowlisted:
            return False, "macro_step_unresolved"
    return True, "macro_ready"


def _commit_retrieval_plan(
    context: ToolContext,
    plan: RetrievalPlan,
) -> None:
    """Apply a reversible active-tool exposure mask."""

    suppressed = set(plan.suppressed_tools)
    current = list(context.options.tools or [])
    kept: list[Tool] = []
    hidden: list[Tool] = []
    for tool in current:
        if tool.name in suppressed:
            hidden.append(tool)
        else:
            kept.append(tool)
    context.options.tools = kept
    context.retrieval_hidden_tools = hidden
    context.retrieval_suppressed_tools = suppressed
    context.retrieval_plan = plan


def _bump_retrieval_metric(context: ToolContext, name: str, amount: int = 1) -> None:
    context.retrieval_metrics[name] = int(context.retrieval_metrics.get(name, 0)) + amount


def _activate_toolsearch_matches(
    registry: ToolRegistry,
    context: ToolContext,
    matches: list[str],
) -> None:
    """Expose matched tools on the next API turn.

    Session overlay macros are synced into ``options.tools`` without
    ``registry.register``. Bundle/persisted tools still use the registry.
    """
    if not matches:
        return
    from extensions.sop_converter.runtime.macros.resolve_tool import (
        resolve_tool_for_context,
    )
    from extensions.sop_converter.runtime.macros.session import (
        is_session_macro_tool,
        sync_effective_tools,
    )

    try:
        from extensions.sop_converter.bundle_context import (
            ensure_bundle_tools_registered,
            get_active_bundle,
        )
    except ImportError:
        ensure_bundle_tools_registered = None  # type: ignore[assignment]
        get_active_bundle = None  # type: ignore[assignment]

    session_names: list[str] = []
    pending: list[str] = []
    for name in matches:
        resolved = resolve_tool_for_context(context, name, base_registry=registry)
        if resolved is not None and is_session_macro_tool(resolved):
            session_names.append(name)
        else:
            pending.append(name)

    if session_names:
        sync_effective_tools(context)

    if pending and ensure_bundle_tools_registered is not None:
        bundle = _resolve_bundle_context(context) or (get_active_bundle() if get_active_bundle is not None else None)
        bundle_path = bundle.bundle_path if bundle is not None else None
        ensure_bundle_tools_registered(registry, pending, bundle_path=bundle_path)

    current = list(context.options.tools or [])
    present = {tool.name for tool in current}
    for name in matches:
        if name in present:
            continue
        tool = resolve_tool_for_context(context, name, base_registry=registry)
        if tool is not None:
            current.append(tool)
            present.add(name)
    if len(current) != len(context.options.tools or []):
        context.options.tools = current


def _find_tool_by_name(tools: list[Tool], name: str) -> Tool | None:
    for tool in tools:
        if tool_matches_name(tool, name):
            return tool
    return None


def _tool_search_output_payload(
    *,
    matches: list[str],
    query: str,
    registry: ToolRegistry,
    searchable_tools: list[Tool] | None = None,
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deferred_count = sum(1 for t in registry.list_tools() if getattr(t, "should_defer", False))
    pool = searchable_tools or []
    match_details: list[dict[str, Any]] = []
    for name in matches:
        tool = _find_tool_by_name(pool, name) or registry.get(name)
        if tool is not None:
            match_details.append(summarize_tool_match(tool))
        else:
            match_details.append({"name": name})
    payload = {
        "matches": matches,
        "match_details": match_details,
        "query": query,
        "total_deferred_tools": deferred_count,
    }
    if retrieval is not None:
        payload["retrieval"] = retrieval
    return payload


def _tool_search_map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    """Map ToolSearch output to API blocks, including tool_reference entries."""
    matches: list[str] = []
    query = ""
    total_deferred = 0
    retrieval: dict[str, Any] | None = None
    if isinstance(output, dict):
        raw_matches = output.get("matches")
        if isinstance(raw_matches, list):
            matches = [str(name) for name in raw_matches if name]
        query = str(output.get("query", "") or "")
        raw_total = output.get("total_deferred_tools")
        if isinstance(raw_total, int):
            total_deferred = raw_total
        raw_retrieval = output.get("retrieval")
        if isinstance(raw_retrieval, dict):
            retrieval = raw_retrieval

    summary = json.dumps(
        {
            "matches": matches,
            "query": query,
            "total_deferred_tools": total_deferred,
            "retrieval": retrieval,
        },
        ensure_ascii=False,
    )
    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": summary}]
    for name in matches:
        content_blocks.append({"type": "tool_reference", "tool_name": name})

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content_blocks,
    }


def make_tool_search_tool(registry: ToolRegistry) -> Tool:
    def _tool_search_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.tool_registry is None:
            context.tool_registry = registry
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        max_results = tool_input.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
            raise ToolInputError("max_results must be an integer between 1 and 50")

        q = query.strip()
        lowered = q.lower()
        # A new search decision replaces the previous turn-local exposure
        # overlay. Exact select is an explicit technical override and also
        # restores any prior hidden tools before resolving.
        context.restore_retrieval_tools()
        searchable_tools = _bundle_scoped_tools(registry, context)
        lifecycle_graph = _load_lifecycle_graph(context)
        if lowered.startswith("select:"):
            name = q.split(":", 1)[1].strip()
            matches = resolve_select_tool_names(name, searchable_tools)
            _activate_toolsearch_matches(registry, context, matches)
            return ToolResult(
                name="ToolSearch",
                output=_tool_search_output_payload(
                    matches=matches,
                    query=query,
                    registry=registry,
                    searchable_tools=searchable_tools,
                ),
            )

        macro_route_catalog = _load_macro_route_catalog(context)
        retrieval_index = _load_retrieval_index(
            context,
            searchable_tools,
            macro_route_catalog,
        )
        route_resolution = None
        try:
            from extensions.sop_converter.runtime.macros.routing import (
                resolve_macro_route_details,
            )

            route_resolution = resolve_macro_route_details(
                q,
                searchable_tools,
                catalog=macro_route_catalog,
            )
        except ImportError:
            route_resolution = None

        if route_resolution is not None and route_resolution.tool_names:
            _bump_retrieval_metric(context, "macro_route_hit_count")

        retrieval_payload: dict[str, Any] | None = None
        if route_resolution is not None and route_resolution.exclusive and len(route_resolution.tool_names) == 1:
            target = route_resolution.tool_names[0]
            route = route_resolution.routes[0] if route_resolution.routes else None
            covered: list[str] = []
            if retrieval_index is not None:
                covered = retrieval_index.covered_names(
                    target,
                    [tool.name for tool in searchable_tools],
                )
            if not covered and route is not None:
                try:
                    from extensions.sop_converter.tool_retrieval import (
                        resolve_tool_references,
                    )

                    covered = resolve_tool_references(
                        getattr(route, "covered_tools", None) or [],
                        [tool.name for tool in searchable_tools],
                    )
                except ImportError:
                    covered = []

            ready, preflight_reason = _preflight_macro(registry, context, target)
            intent_key = str(getattr(route, "intent_key", "") or "") if route else ""
            if ready:
                plan = RetrievalPlan(
                    query=q,
                    intent_key=intent_key or None,
                    selected_macros=[target],
                    suppressed_tools=covered,
                    selection="exclusive",
                    route_scope=str(getattr(route, "scope", "") or "") or None,
                    preflight_status="ready",
                    reason_codes=["verified_route", "macro_coverage", preflight_reason],
                )
                _activate_toolsearch_matches(registry, context, [target])
                _commit_retrieval_plan(context, plan)
                _bump_retrieval_metric(context, "macro_exclusive_commit_count")
                _bump_retrieval_metric(context, "atomic_suppressed_count", len(covered))
                context.retrieval_metrics.setdefault("first_selected_tool_layer", "macro")
                matches = [target][:max_results]
                return ToolResult(
                    name="ToolSearch",
                    output=_tool_search_output_payload(
                        matches=matches,
                        query=query,
                        registry=registry,
                        searchable_tools=searchable_tools,
                        retrieval=plan.to_dict(),
                    ),
                )

            # Preflight failed before any workflow step: restore covered
            # atomics in the same ToolSearch response and remove the broken
            # target from normal scoring so its route cannot win again.
            context.options.tools = [tool for tool in (context.options.tools or []) if tool.name != target]
            _bump_retrieval_metric(context, "macro_preflight_failure_count")
            _bump_retrieval_metric(context, "atomic_restore_count", len(covered))
            fallback_tools = [tool for tool in searchable_tools if tool.name != target]
            ranked_fallback = rank_tool_matches(
                q,
                fallback_tools,
                max_results=max_results,
                lifecycle_graph=lifecycle_graph,
                macro_route_catalog=macro_route_catalog,
                retrieval_index=retrieval_index,
            )
            matches = []
            for name in [*covered, *ranked_fallback]:
                if name not in matches:
                    matches.append(name)
            matches = matches[:max_results]
            _activate_toolsearch_matches(registry, context, matches)
            retrieval_payload = RetrievalPlan(
                query=q,
                intent_key=intent_key or None,
                selected_macros=[target],
                suppressed_tools=[],
                selection="normal",
                route_scope=str(getattr(route, "scope", "") or "") or None,
                preflight_status="unavailable",
                reason_codes=[
                    "macro_preflight_unavailable",
                    preflight_reason,
                    "atomic_restore",
                ],
            ).to_dict()
            return ToolResult(
                name="ToolSearch",
                output=_tool_search_output_payload(
                    matches=matches,
                    query=query,
                    registry=registry,
                    searchable_tools=searchable_tools,
                    retrieval=retrieval_payload,
                ),
            )

        matches = rank_tool_matches(
            q,
            searchable_tools,
            max_results=max_results,
            lifecycle_graph=lifecycle_graph,
            macro_route_catalog=macro_route_catalog,
            retrieval_index=retrieval_index,
        )
        if route_resolution is not None and route_resolution.tool_names:
            route = route_resolution.routes[0] if route_resolution.routes else None
            retrieval_payload = RetrievalPlan(
                query=q,
                intent_key=(str(getattr(route, "intent_key", "") or "") or None if route is not None else None),
                selected_macros=list(route_resolution.tool_names),
                suppressed_tools=[],
                selection="prefer" if not route_resolution.conflict else "normal",
                route_scope=(str(getattr(route, "scope", "") or "") or None if route is not None else None),
                preflight_status="pending",
                reason_codes=(["macro_route_conflict"] if route_resolution.conflict else ["macro_prefer"]),
            ).to_dict()
        _activate_toolsearch_matches(registry, context, matches)
        return ToolResult(
            name="ToolSearch",
            output=_tool_search_output_payload(
                matches=matches,
                query=query,
                registry=registry,
                searchable_tools=searchable_tools,
                retrieval=retrieval_payload,
            ),
        )

    return build_tool(
        name="ToolSearch",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        call=_tool_search_call,
        prompt="Search for available tools by name or keywords.",
        description="Search for available tools by name or keywords.",
        map_result_to_api=_tool_search_map_result_to_api,
        strict=True,
        max_result_size_chars=100_000,
        is_read_only=lambda _input: True,
        is_concurrency_safe=lambda _input: True,
        # ToolSearch's classifier-input is the query the model wants
        # to search for -- already compact enough for the classifier.
        to_auto_classifier_input=lambda input_data: ((input_data or {}).get("query", "") or ""),
    )
