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

"""Macro route matching and conflict resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .models import MacroRoute

_SCOPE_RANK = {"session": 3, "bundle": 2, "builtin": 1}


def _scope_rank(scope: str) -> int:
    return _SCOPE_RANK.get(scope, 0)


def _tokenize_query(query: str) -> list[str]:
    tokens = re.split(r"[\s_\-./]+", query.lower())
    return [token for token in tokens if len(token) >= 2]


def _phrase_matches(query: str, phrase: str) -> bool:
    query_lower = query.lower().strip()
    phrase_lower = phrase.lower().strip()
    if not phrase_lower:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(phrase_lower)}(?![a-z0-9])"
    return bool(re.search(pattern, query_lower))


def _keyword_matches(query: str, keyword: str) -> bool:
    query_lower = query.lower().strip()
    keyword_lower = keyword.lower().strip()
    if not keyword_lower:
        return False
    if len(keyword_lower) <= 1:
        return keyword_lower in query_lower
    words = re.split(r"[\s_\-./]+", query_lower)
    for word in words:
        if keyword_lower == word:
            return True
        if keyword_lower in word:
            idx = word.find(keyword_lower)
            if idx == 0:
                return True
            if not (word[idx - 1].isascii() and word[idx - 1].isalnum()):
                nxt = idx + len(keyword_lower)
                if nxt == len(word) or not (word[nxt].isascii() and word[nxt].isalnum()):
                    return True
    if len(keyword_lower) > 2:
        compact_query = re.sub(r"[\s_\-./]+", "", query_lower)
        compact_keyword = re.sub(r"[\s_\-./]+", "", keyword_lower)
        return compact_keyword in compact_query
    return False


def _matches_route(query: str, route: MacroRoute) -> bool:
    if not route.target_tool:
        return False
    if not route.enabled:
        return False

    query_lower = query.lower().strip()

    for neg_keyword in route.negative_keywords:
        if _keyword_matches(query_lower, neg_keyword):
            return False

    has_phrases = bool(route.phrases)
    has_keywords = bool(route.keywords)

    if route.match_mode == "exact":
        if has_phrases:
            return any(_phrase_matches(query_lower, phrase) for phrase in route.phrases)
        if has_keywords:
            return all(_keyword_matches(query_lower, keyword) for keyword in route.keywords)
        return False

    # Phrase hit is high-confidence and sufficient on its own.
    # match_mode (all/any) applies only to the keyword-only path below.
    if has_phrases and any(_phrase_matches(query_lower, phrase) for phrase in route.phrases):
        return True

    if has_keywords:
        if route.match_mode == "all":
            return all(_keyword_matches(query_lower, keyword) for keyword in route.keywords)
        else:
            return any(_keyword_matches(query_lower, keyword) for keyword in route.keywords)

    return False


@dataclass
class MacroRouteMatch:
    route: MacroRoute
    matched_tokens: int
    is_exact_phrase: bool = False


@dataclass
class MacroRouteResolution:
    tool_names: list[str] = field(default_factory=list)
    routes: list[MacroRoute] = field(default_factory=list)
    exclusive: bool = False
    conflict: bool = False


@dataclass
class MacroRouteCatalog:
    """Store and match macro routes across scopes."""

    _routes: list[MacroRoute] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def register_route(self, route: MacroRoute, *, replace: bool = False) -> None:
        with self._lock:
            if not replace:
                for existing in self._routes:
                    if existing.target_tool == route.target_tool:
                        raise ValueError(f"route for {route.target_tool} already exists")
            else:
                self._routes = [r for r in self._routes if r.target_tool != route.target_tool]
            self._routes.append(route)

    def unregister_route(self, target_tool: str) -> None:
        with self._lock:
            self._routes = [r for r in self._routes if r.target_tool != target_tool]

    def get_routes(self) -> list[MacroRoute]:
        with self._lock:
            return list(self._routes)

    def match_routes(self, query: str) -> list[MacroRouteMatch]:
        matches: list[MacroRouteMatch] = []
        query_lower = query.lower().strip()

        for route in self.get_routes():
            if _matches_route(query_lower, route):
                matched_tokens = sum(1 for keyword in route.keywords if _keyword_matches(query_lower, keyword))
                is_exact_phrase = any(_phrase_matches(query_lower, phrase) for phrase in route.phrases)
                matches.append(
                    MacroRouteMatch(route=route, matched_tokens=matched_tokens, is_exact_phrase=is_exact_phrase)
                )

        # §8.4: exact phrase > scope (session > bundle > builtin) > priority > tokens
        matches.sort(
            key=lambda m: (
                m.is_exact_phrase,
                _scope_rank(m.route.scope),
                m.route.priority,
                m.matched_tokens,
            ),
            reverse=True,
        )
        return matches

    def clear(self) -> None:
        with self._lock:
            self._routes.clear()


DEFAULT_MACRO_ROUTE_CATALOG = MacroRouteCatalog()


def ensure_builtin_routes(catalog: MacroRouteCatalog | None = None) -> MacroRouteCatalog:
    target = catalog or DEFAULT_MACRO_ROUTE_CATALOG
    existing_tools = {r.target_tool for r in target.get_routes()}
    if "invoke-existing-agent" not in existing_tools:
        target.register_route(
            MacroRoute(
                phrases=[
                    # Canonical + named-bot route acceptance.
                    "用已创建的 agent 回复",
                    "用 verify-bot 回复",
                    "调用已有 agent",
                    "通过 agent id 调用",
                    "按 id 调用 agent",
                    # Regression corpus (was covered by hardcoded ToolSearch shortcut)
                    "invoke existing agent by agent_id",
                    "invoke llm agent send message to agent",
                    "llmagent invoke",
                ],
                keywords=["agent", "回复"],
                negative_keywords=["创建", "配置", "删除", "列出"],
                target_tool="invoke-existing-agent",
                match_mode="all",
                selection="exclusive",
                priority=100,
                verified=True,
                intent_key="agent.invoke_existing",
                covered_tools=["llmagent-invoke", "send-to-agent"],
                unavailable_policy="restore-covered",
                scope="builtin",
            ),
            replace=True,
        )
    # Always replace so phrase/keyword updates apply after reload.
    target.register_route(
        MacroRoute(
            phrases=[
                "resume-resource",
                "resume resource",
                "恢复资源",
                "资源恢复",
                "恢复一下资源",
                "把资源恢复",
                "按 resource_type 调用",
                "按 resource_ref 恢复",
                "resume catalog resource",
                "invoke resource by resource_ref",
            ],
            # Chinese NL often inserts particles between the two keywords, so
            # phrase-only matching is too brittle. Require both keywords.
            keywords=["恢复", "资源"],
            negative_keywords=["创建", "删除", "列出"],
            target_tool="resume-resource",
            match_mode="all",
            selection="prefer",
            priority=90,
            verified=False,
            intent_key="resource.resume",
            covered_tools=[],
            unavailable_policy="restore-covered",
            scope="builtin",
        ),
        replace=True,
    )
    return target


def register_macro_route(
    route: MacroRoute,
    *,
    catalog: MacroRouteCatalog | None = None,
    replace: bool = False,
) -> None:
    (catalog or DEFAULT_MACRO_ROUTE_CATALOG).register_route(route, replace=replace)


def match_macro_routes(
    query: str,
    *,
    catalog: MacroRouteCatalog | None = None,
) -> list[MacroRouteMatch]:
    return (catalog or DEFAULT_MACRO_ROUTE_CATALOG).match_routes(query)


def get_negative_keyword_exclusions(
    query: str,
    tools: list[Any],
    *,
    catalog: MacroRouteCatalog | None = None,
) -> list[str]:
    target_catalog = catalog or DEFAULT_MACRO_ROUTE_CATALOG
    available_tools = {tool.name for tool in tools}
    excluded: list[str] = []

    for route in target_catalog.get_routes():
        if not route.enabled:
            continue
        if route.target_tool not in available_tools:
            continue
        query_lower = query.lower().strip()
        for neg_keyword in route.negative_keywords:
            if _keyword_matches(query_lower, neg_keyword):
                if route.target_tool not in excluded:
                    excluded.append(route.target_tool)
                break

    return excluded


def resolve_macro_route_details(
    query: str,
    tools: list[Any],
    *,
    catalog: MacroRouteCatalog | None = None,
) -> MacroRouteResolution:
    """Resolve matched macro tools and retain the winning route metadata.

    Returns a :class:`MacroRouteResolution`. ``exclusive=True`` means
    ToolSearch may build an exclusive RetrievalPlan; ``exclusive=False`` means
    prefer/conflict and callers keep normal search candidates.

        Tied verified exclusive routes (§8.4) return all tied candidates with
        ``exclusive=False`` (do not silently pick one; fall back toward normal
        ToolSearch). Builtin verified exclusives are safety routes and cannot be
        overridden by session/bundle exclusives.
    """
    matches = match_macro_routes(query, catalog=catalog)
    if not matches:
        return MacroRouteResolution()

    available_tools = {tool.name for tool in tools}
    exclusive_matches: list[MacroRouteMatch] = []
    prefer_names: list[str] = []
    prefer_routes: list[MacroRoute] = []

    for match in matches:
        route = match.route
        if route.target_tool not in available_tools:
            continue
        if route.selection == "exclusive":
            if not route.verified:
                continue
            exclusive_matches.append(match)
        elif route.selection == "prefer" and route.target_tool not in prefer_names:
            prefer_names.append(route.target_tool)
            prefer_routes.append(route)

    if exclusive_matches:
        # §8.4: builtin safety exclusives cannot be silently overridden.
        safety = [m for m in exclusive_matches if m.route.scope == "builtin"]
        pool = safety if safety else exclusive_matches
        # Re-rank pool with the same full ordering as match_macro_routes:
        # exact phrase > scope (session > bundle > builtin) > priority > tokens.
        pool = sorted(
            pool,
            key=lambda m: (
                m.is_exact_phrase,
                _scope_rank(m.route.scope),
                m.route.priority,
                m.matched_tokens,
            ),
            reverse=True,
        )
        top_key = (
            pool[0].is_exact_phrase,
            _scope_rank(pool[0].route.scope),
            pool[0].route.priority,
            pool[0].matched_tokens,
        )
        tied = [
            m
            for m in pool
            if (
                m.is_exact_phrase,
                _scope_rank(m.route.scope),
                m.route.priority,
                m.matched_tokens,
            )
            == top_key
        ]
        names: list[str] = []
        for match in tied:
            if match.route.target_tool not in names:
                names.append(match.route.target_tool)
        if len(names) > 1:
            # Tied exclusives: return both candidates, do not truncate as exclusive
            return MacroRouteResolution(
                tool_names=names[:5],
                routes=[match.route for match in tied[:5]],
                exclusive=False,
                conflict=True,
            )
        return MacroRouteResolution(
            tool_names=names[:5],
            routes=[tied[0].route],
            exclusive=True,
        )

    return MacroRouteResolution(
        tool_names=prefer_names[:5],
        routes=prefer_routes[:5],
        exclusive=False,
    )


def resolve_macro_route(
    query: str,
    tools: list[Any],
    *,
    catalog: MacroRouteCatalog | None = None,
) -> tuple[list[str], bool]:
    """Compatibility tuple wrapper around :func:`resolve_macro_route_details`."""

    resolution = resolve_macro_route_details(query, tools, catalog=catalog)
    return resolution.tool_names, resolution.exclusive
