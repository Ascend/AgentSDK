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

"""ToolSearch scoring helpers — token-aware matching over tool metadata."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ..build_tool import Tool

_TOKEN_SPLIT_RE = re.compile(r"[\s_\-./]+")
_NAME_SEGMENT_RE = re.compile(r"[-_./]+")
# Alphanumeric word boundary for tag / description text (not kebab segments).
_TEXT_TOKEN_BOUNDARY = r"(?<![a-z0-9]){token}(?![a-z0-9])"  # nosec


def tokenize_query(query: str) -> list[str]:
    """Split a ToolSearch query into lowercase tokens (min length 2)."""
    return [token for token in _TOKEN_SPLIT_RE.split(query.lower()) if len(token) >= 2]


def _name_segments(tool_name: str) -> list[str]:
    return [segment for segment in _NAME_SEGMENT_RE.split(tool_name.lower()) if segment]


def _query_as_kebab(query: str) -> str:
    parts = [part for part in _TOKEN_SPLIT_RE.split(query.lower().strip()) if part]
    return "-".join(parts)


def _compact(text: str) -> str:
    """Remove separators so ``run teamcli`` can match ``run team cli`` tags / kebab names."""
    return re.sub(r"[\s_\-./]+", "", text.lower())


def _compact_query_in_compact_text(query: str, text: str, *, min_compact_len: int = 5) -> bool:
    compact_query = _compact(query)
    if len(compact_query) < min_compact_len:
        return False
    compact_text = _compact(text)
    return bool(compact_text) and compact_query in compact_text


def _consecutive_segment_compacts(tool_name: str) -> set[str]:
    """Compact forms of every consecutive kebab-name segment run."""
    segments = _name_segments(tool_name)
    compacts: set[str] = set()
    for start in range(len(segments)):
        for end in range(start + 1, len(segments) + 1):
            joined = "-".join(segments[start:end])
            compacts.add(_compact(joined))
    return compacts


def _compact_hint_phrases(hint_lower: str, *, max_words: int = 4) -> set[str]:
    """Compact forms of individual tags and short consecutive tag runs in ``search_hint``."""
    words = [word for word in hint_lower.split() if word]
    phrases: set[str] = set()
    for start in range(len(words)):
        for end in range(start + 1, min(start + max_words, len(words)) + 1):
            phrases.add(_compact(" ".join(words[start:end])))
    return phrases


def _token_matches_glued_segments(token: str, tool_name: str) -> bool:
    """``teamcli`` matches when name segments contain ``team`` + ``cli`` adjacently."""
    compact_token = _compact(token)
    if len(compact_token) < 4:
        return False
    return compact_token in _consecutive_segment_compacts(tool_name)


def _token_in_compact_hint(token: str, hint_lower: str) -> bool:
    compact_token = _compact(token)
    if len(compact_token) < 4:
        return False
    return compact_token in _compact_hint_phrases(hint_lower)


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Match a multi-word or single-word phrase with text word boundaries."""
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    pattern = _TEXT_TOKEN_BOUNDARY.format(token=re.escape(phrase))
    return bool(re.search(pattern, text.lower()))


def _token_in_text(token: str, text: str) -> bool:
    return _phrase_in_text(token, text)


def _token_in_tool_name(token: str, tool_name: str) -> bool:
    """Match a token against kebab-case name segments (exact segment equality)."""
    lowered = token.lower()
    return lowered in _name_segments(tool_name)


def _query_subphrases(tokens: list[str], *, min_tokens: int = 2) -> list[str]:
    """Longest-first consecutive token sub-phrases from the query."""
    if len(tokens) < min_tokens:
        return []
    phrases: list[str] = []
    for start in range(len(tokens)):
        for end in range(start + min_tokens, len(tokens) + 1):
            phrases.append(" ".join(tokens[start:end]))
    return sorted(set(phrases), key=lambda phrase: len(phrase.split()), reverse=True)


def _tool_description_text(tool: Tool) -> str | None:
    """Return the tool's top-level description (when available)."""
    desc_fn = getattr(tool, "description", None)
    if not callable(desc_fn):
        return None
    try:
        text = desc_fn({})
    except Exception:  # nosec
        return None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _collect_schema_descriptions(
    schema: Mapping[str, Any] | None,
    *,
    depth: int = 0,
    max_depth: int = 2,
) -> list[str]:
    """Extract human-readable description strings from a JSON Schema."""
    if not schema or not isinstance(schema, dict) or depth > max_depth:
        return []

    texts: list[str] = []
    root_desc = schema.get("description")
    if isinstance(root_desc, str) and root_desc.strip():
        texts.append(root_desc.strip())

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return texts

    for prop_schema in properties.values():
        if not isinstance(prop_schema, dict):
            continue
        prop_desc = prop_schema.get("description")
        if isinstance(prop_desc, str) and prop_desc.strip():
            texts.append(prop_desc.strip())
        if depth < max_depth and isinstance(prop_schema.get("properties"), dict):
            texts.extend(
                _collect_schema_descriptions(
                    prop_schema,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
    return texts


def tool_search_document(tool: Tool) -> str:
    """Build the searchable text blob for a tool."""
    parts: list[str] = [tool.name, tool.prompt()]

    desc = _tool_description_text(tool)
    if desc:
        parts.append(desc)

    search_hint = getattr(tool, "search_hint", None)
    if search_hint:
        parts.append(search_hint)

    input_schema = getattr(tool, "input_schema", None)
    if input_schema is not None:
        parts.extend(_collect_schema_descriptions(input_schema))

    aliases = getattr(tool, "aliases", ()) or ()
    if aliases:
        parts.extend(str(alias) for alias in aliases)
    return "\n".join(parts).lower()


def _kebab_suffix_in_tool_name(query: str, tool_name: str) -> bool:
    """Match ``team-memory-dir`` or ``team memory dir`` against a tool name suffix."""
    lowered = query.lower().strip()
    name_lower = tool_name.lower()
    kebab = _query_as_kebab(lowered)
    if kebab and len(kebab) >= 8 and name_lower.endswith(kebab):
        return True
    if "-" in lowered and len(lowered) >= 8 and lowered in name_lower:
        return True
    tokens = tokenize_query(lowered)
    for n in range(len(tokens), 1, -1):
        suffix = "-".join(tokens[-n:])
        if len(suffix) >= 8 and name_lower.endswith(suffix):
            return True
        if tokens[-n:] and tokens[-1] == "directory":
            alt = "-".join([*tokens[-n:-1], "dir"])
            if len(alt) >= 8 and name_lower.endswith(alt):
                return True
    return False


def _path_vs_ensure_intent(tokens: list[str], tool_name: str) -> tuple[int, int, int, str] | None:
    """Boost path-lookup tools vs directory-creation tools for ambiguous memory queries."""
    token_set = set(tokens)
    if not token_set & {"memory", "team"}:
        return None
    name_lower = tool_name.lower()
    wants_path = bool(token_set & {"path", "return", "get", "directory"})
    wants_ensure = bool(token_set & {"ensure", "create", "mkdir"})
    if wants_path and not wants_ensure:
        if name_lower.endswith("team-memory-dir") or ("memory-dir" in name_lower and "ensure" not in name_lower):
            return (0, 2, 0, tool_name)
    if wants_ensure:
        if "ensure-dir" in name_lower:
            return (0, 2, 1, tool_name)
    return None


def _tool_search_parts(tool: Tool) -> tuple[str, str | None, str]:
    search_hint = getattr(tool, "search_hint", None)
    return tool.name, search_hint, tool_search_document(tool)


def _normalize_tool_ref(value: str) -> str:
    return re.sub(r"[._\-]+", "-", str(value or "").strip()).strip("-").lower()


def _tool_ref_matches_name(tool_name: str, ref: str) -> bool:
    if not ref:
        return False
    lowered_name = tool_name.lower()
    lowered_ref = str(ref).lower()
    if lowered_name == lowered_ref:
        return True
    return _normalize_tool_ref(tool_name) == _normalize_tool_ref(ref)


def _route_keyword_matches(query: str, keyword: str) -> bool:
    keyword = str(keyword or "").strip().lower()
    if not keyword:
        return False
    query_lower = query.lower().strip()
    if keyword in query_lower:
        return True
    return _compact_query_in_compact_text(keyword, query_lower, min_compact_len=4)


def _lifecycle_group_for_tool(
    tool_name: str,
    lifecycle_graph: Any,
) -> tuple[Any, int] | None:
    groups = getattr(lifecycle_graph, "intent_groups", None) or []
    for group in groups:
        tools = getattr(group, "tools", None) or []
        for idx, ref in enumerate(tools):
            if _tool_ref_matches_name(tool_name, str(ref)):
                return group, idx
    return None


def _lifecycle_route_for_query(query: str, lifecycle_graph: Any) -> Any | None:
    routes = getattr(lifecycle_graph, "priority_routes", None) or []
    for route in routes:
        keywords = getattr(route, "keywords", None) or []
        if any(_route_keyword_matches(query, str(keyword)) for keyword in keywords):
            return route
    return None


def _dependency_order_for_tool(tool_name: str, lifecycle_graph: Any) -> int:
    deps = getattr(lifecycle_graph, "dependencies", None) or []
    for dep in deps:
        from_tool = str(getattr(dep, "from_tool", "") or "")
        to_tool = str(getattr(dep, "to_tool", "") or "")
        if _tool_ref_matches_name(tool_name, from_tool):
            return 0
        if _tool_ref_matches_name(tool_name, to_tool):
            return 1
    return 2


def _lifecycle_sort_key(
    query: str,
    tool_name: str,
    lifecycle_graph: Any | None,
) -> tuple[int, int, int]:
    """Return a bias key from ``tool-dependencies.yaml`` metadata.

    Lower is better.  The key is intentionally coarse; normal ToolSearch scoring
    still decides relevance, while lifecycle metadata only breaks collisions and
    lifts the routed intent group when route keywords match.
    """
    if lifecycle_graph is None:
        return (1, 2, 999)
    route = _lifecycle_route_for_query(query, lifecycle_graph)
    if route is None:
        return (1, 2, 999)
    group_name = str(getattr(route, "intent_group", "") or "")
    group_hit = _lifecycle_group_for_tool(tool_name, lifecycle_graph)
    if group_hit is None:
        return (1, 2, 999)
    group, group_idx = group_hit
    if str(getattr(group, "name", "") or "") != group_name:
        return (1, 2, 999)
    primary_entry = str(getattr(group, "primary_entry", "") or "")
    entry_first = bool(getattr(route, "entry_first", True))
    is_primary = bool(primary_entry and _tool_ref_matches_name(tool_name, primary_entry))
    dep_order = _dependency_order_for_tool(tool_name, lifecycle_graph)
    if entry_first:
        order = 0 if is_primary else (dep_order if dep_order < 2 else 1)
    else:
        order = 0 if dep_order == 1 else (1 if is_primary or dep_order == 0 else 2)
    return (0, order, group_idx)


def _lifecycle_chain_query(query: str) -> str | None:
    lowered = query.lower().strip()
    for prefix in (
        "lifecycle-chain:",
        "lifecycle-chain",
        "lifecycle chain:",
        "lifecycle chain",
    ):
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return None


def _lifecycle_chain_tool_names(
    query: str,
    tools: list[Tool],
    lifecycle_graph: Any | None,
) -> list[str]:
    if lifecycle_graph is None:
        return []
    chain_query = _lifecycle_chain_query(query)
    if chain_query is None:
        return []
    route = _lifecycle_route_for_query(chain_query, lifecycle_graph)
    group_name = str(getattr(route, "intent_group", "") or "") if route is not None else ""
    if not group_name:
        tokens = tokenize_query(chain_query)
        for group in getattr(lifecycle_graph, "intent_groups", None) or []:
            haystack = " ".join(
                [
                    str(getattr(group, "name", "") or ""),
                    str(getattr(group, "description", "") or ""),
                    " ".join(str(t) for t in getattr(group, "tools", None) or []),
                ]
            ).lower()
            if all(token in haystack for token in tokens) if tokens else True:
                group_name = str(getattr(group, "name", "") or "")
                break
    if not group_name:
        return []
    group = None
    for candidate in getattr(lifecycle_graph, "intent_groups", None) or []:
        if str(getattr(candidate, "name", "") or "") == group_name:
            group = candidate
            break
    if group is None:
        return []
    available = {tool.name: tool for tool in tools}
    ordered_refs: list[str] = []
    primary = str(getattr(group, "primary_entry", "") or "")
    if primary:
        ordered_refs.append(primary)
    group_refs = {_normalize_tool_ref(str(ref)) for ref in (getattr(group, "tools", None) or []) if ref}
    for dep in getattr(lifecycle_graph, "dependencies", None) or []:
        dep_from = str(getattr(dep, "from_tool", "") or "")
        dep_to = str(getattr(dep, "to_tool", "") or "")
        if group_refs and not (
            _normalize_tool_ref(dep_from) in group_refs and _normalize_tool_ref(dep_to) in group_refs
        ):
            continue
        for ref in (
            str(getattr(dep, "from_tool", "") or ""),
            str(getattr(dep, "to_tool", "") or ""),
        ):
            if ref and ref not in ordered_refs:
                ordered_refs.append(ref)
    for ref in getattr(group, "tools", None) or []:
        ref = str(ref)
        if ref and ref not in ordered_refs:
            ordered_refs.append(ref)
    names: list[str] = []
    for ref in ordered_refs:
        for name in available:
            if _tool_ref_matches_name(name, ref) and name not in names:
                names.append(name)
                break
    return names


def _direct_macro_route_names(
    query: str,
    tools: list[Tool],
    macro_route_catalog: Any | None = None,
) -> tuple[list[str], list[str], bool]:
    """Return macro tools matched by direct route and excluded tools.

    Macro routes are data-driven and independent of PriorityRoute.
    Routes can actively recall a target tool that would not otherwise
    appear in normal ToolSearch results.

    Only verified routes or routes with selection=prefer are considered.
    Exclusive routes require verified=true to prevent accidental truncation
    of candidate lists.

    Returns:
        Tuple of (matched_tool_names, excluded_tool_names, exclusive).
        When ``exclusive`` is True, callers should return only matched tools.
        When False (prefer), callers should prepend matched tools to normal
        search results.
    """
    matched: list[str] = []
    excluded: list[str] = []
    exclusive = False

    if macro_route_catalog is None:
        try:
            from extensions.sop_converter.runtime.macros.routing import (
                DEFAULT_MACRO_ROUTE_CATALOG,
                ensure_builtin_routes,
                resolve_macro_route,
                get_negative_keyword_exclusions,
            )

            ensure_builtin_routes(DEFAULT_MACRO_ROUTE_CATALOG)
            matched, exclusive = resolve_macro_route(query, tools, catalog=DEFAULT_MACRO_ROUTE_CATALOG)
            excluded = get_negative_keyword_exclusions(query, tools, catalog=DEFAULT_MACRO_ROUTE_CATALOG)
        except ImportError:
            pass  # Optional integration is unavailable; keep the fallback.
    else:
        try:
            from extensions.sop_converter.runtime.macros.routing import (
                resolve_macro_route,
                get_negative_keyword_exclusions,
            )

            matched, exclusive = resolve_macro_route(query, tools, catalog=macro_route_catalog)
            excluded = get_negative_keyword_exclusions(query, tools, catalog=macro_route_catalog)
        except ImportError:
            pass  # Optional integration is unavailable; keep the fallback.

    return matched, excluded, exclusive


def _apply_macro_structural_bias(
    ranked: list[str],
    score_tiers: Mapping[str, int],
    retrieval_index: Any | None,
) -> list[str]:
    """Place a macro before covered atomics only within the same score tier."""

    if not ranked or retrieval_index is None:
        return ranked
    result = list(ranked)
    for coverage in getattr(retrieval_index, "coverage", None) or []:
        macro = str(getattr(coverage, "macro_tool", "") or "")
        if macro not in result:
            continue
        try:
            covered = retrieval_index.covered_names(macro, result)
        except Exception:  # nosec
            continue
        same_tier = [name for name in covered if name in result and score_tiers.get(name) == score_tiers.get(macro)]
        if not same_tier:
            continue
        macro_index = result.index(macro)
        first_atomic = min(result.index(name) for name in same_tier)
        if macro_index > first_atomic:
            result.pop(macro_index)
            first_atomic = min(result.index(name) for name in same_tier)
            result.insert(first_atomic, macro)
    return result


def rank_tools_by_lifecycle(
    matches: list[str],
    query: str,
    lifecycle_graph: Any | None,
) -> list[str]:
    """Sort ToolSearch match names using lifecycle metadata.

    This implements the documented P2 hook: when ``priority_routes`` match the
    query, tools in the routed ``intent_group`` are lifted and ordered by the
    dependency chain with ``primary_entry`` first for create-style routes.
    """
    if not matches or lifecycle_graph is None:
        return matches
    indexed = list(enumerate(matches))
    indexed.sort(
        key=lambda item: (
            _lifecycle_sort_key(query, item[1], lifecycle_graph),
            item[0],
        )
    )
    return [name for _, name in indexed]


def score_tool_match(
    query: str,
    *,
    tool_name: str,
    search_hint: str | None = None,
    document: str | None = None,
) -> tuple[int, int, int, str] | None:
    """Return a sort key for a tool match, or ``None`` if no match.

    Sort key tiers (lower is better):
      0 — full query matches tool name (exact, kebab, compact, or segment)
      1 — full query matches search_hint / tags (word-boundary or compact phrase)
      2 — full query matches full document (word-boundary or compact phrase)
      3 — longest consecutive query sub-phrase (≥2 tokens) in search_hint
      4 — word-boundary / segment / glued-segment token overlap
            (secondary: more matched tokens; tertiary: higher coverage ratio)

    Within a tier, secondary ``0`` beats ``1`` (exact spacing beats compact).
    """
    lowered_query = query.lower().strip()
    if not lowered_query:
        return None

    doc_lower = (document or tool_name).lower()
    hint_lower = (search_hint or "").lower()
    name_lower = tool_name.lower()
    kebab_query = _query_as_kebab(lowered_query)
    tokens = tokenize_query(lowered_query)
    if not tokens:
        return None

    intent_key = _path_vs_ensure_intent(tokens, tool_name)
    if intent_key is not None:
        return intent_key

    wants_ensure = bool(set(tokens) & {"ensure", "create", "mkdir"})
    wants_path_only = bool(set(tokens) & {"path", "return", "get", "directory"}) and not wants_ensure

    # Tier 0 — strong tool-name match (segment-aware; no ``team`` ⊂ ``teams``)
    if lowered_query == name_lower:
        return (0, 0, 0, tool_name)
    if _kebab_suffix_in_tool_name(lowered_query, tool_name):
        if wants_ensure and "ensure" not in name_lower:
            pass
        elif wants_path_only and "ensure" in name_lower:
            pass
        else:
            return (0, 1, 0, tool_name)
    if len(tokens) >= 2 and kebab_query and kebab_query in name_lower:
        return (0, 0, 0, tool_name)
    if len(tokens) >= 2 and _compact_query_in_compact_text(lowered_query, name_lower):
        return (0, 1, 0, tool_name)
    if len(tokens) == 1 and _token_in_tool_name(tokens[0], tool_name):
        return (0, 0, 0, tool_name)
    if len(tokens) == 1 and _token_matches_glued_segments(tokens[0], tool_name):
        return (0, 1, 0, tool_name)

    # Tier 1 — full phrase in tags (exact spacing, then compact)
    if hint_lower and _phrase_in_text(lowered_query, hint_lower):
        return (1, 0, 0, tool_name)
    if hint_lower and _compact_query_in_compact_text(lowered_query, hint_lower):
        return (1, 1, 0, tool_name)

    # Tier 2 — full phrase anywhere in document
    if _phrase_in_text(lowered_query, doc_lower):
        return (2, 0, 0, tool_name)
    if _compact_query_in_compact_text(lowered_query, doc_lower):
        return (2, 1, 0, tool_name)

    # Tier 3 — consecutive sub-phrase in tags (prefer longer phrases)
    if hint_lower:
        for phrase in _query_subphrases(tokens):
            if _phrase_in_text(phrase, hint_lower):
                phrase_len = len(phrase.split())
                return (3, 0, -phrase_len, tool_name)
            if _compact_query_in_compact_text(phrase, hint_lower):
                phrase_len = len(phrase.split())
                return (3, 1, -phrase_len, tool_name)

    # Tier 4 — token overlap (tags + kebab name segments + glued segments)
    matched = 0
    for token in tokens:
        if hint_lower and _token_in_text(token, hint_lower):
            matched += 1
            continue
        if _token_in_tool_name(token, tool_name):
            matched += 1
            continue
        if _token_matches_glued_segments(token, tool_name):
            matched += 1
            continue
        if hint_lower and _token_in_compact_hint(token, hint_lower):
            matched += 1
            continue
        # Fall back to full document (description, schema param text, etc.).
        if document and _token_in_text(token, doc_lower):
            matched += 1

    if matched == 0:
        return None

    coverage_pct = int(round((matched / len(tokens)) * 1000))
    return (4, -matched, -coverage_pct, tool_name)


def _name_segment_overlap(tokens: list[str], tool_name: str) -> int:
    return sum(1 for token in tokens if _token_in_tool_name(token, tool_name))


def resolve_select_tool_names(select: str, tools: list[Tool]) -> list[str]:
    """Resolve ``select:<ref>`` to tool names (exact, alias, suffix, or rank)."""
    from ..build_tool import tool_matches_name

    ref = select.strip()
    if not ref:
        return []
    lowered = ref.lower()

    exact = [tool.name for tool in tools if tool.name.lower() == lowered]
    if exact:
        return exact[:1]

    alias_hits = [tool.name for tool in tools if tool_matches_name(tool, ref)]
    if alias_hits:
        return alias_hits[:1]

    suffix_hits = [tool.name for tool in tools if lowered in tool.name.lower() and tool.name.lower().endswith(lowered)]
    if len(suffix_hits) == 1:
        return suffix_hits

    if "-" in ref:
        partial = [tool.name for tool in tools if _kebab_suffix_in_tool_name(ref, tool.name)]
        if len(partial) == 1:
            return partial

    ranked = rank_tool_matches(ref, tools, max_results=1)
    return ranked[:1]


def summarize_tool_match(tool: Tool) -> dict[str, Any]:
    """Build a compact summary for ToolSearch results (name + docstring + params)."""
    summary: dict[str, Any] = {"name": tool.name}

    desc = _tool_description_text(tool)
    if not desc:
        hint = getattr(tool, "search_hint", None)
        if isinstance(hint, str) and hint.strip():
            desc = hint.strip()
    if desc:
        summary["description"] = desc

    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        required = schema.get("required")
        if isinstance(required, list):
            req_names = [str(name) for name in required if name]
            if req_names:
                summary["required"] = req_names

        properties = schema.get("properties")
        if isinstance(properties, dict):
            param_descs: dict[str, str] = {}
            for param_name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    continue
                param_desc = prop_schema.get("description")
                if isinstance(param_desc, str) and param_desc.strip():
                    param_descs[str(param_name)] = param_desc.strip()
            if param_descs:
                summary["parameter_descriptions"] = param_descs

    return summary


def rank_tool_matches(
    query: str,
    tools: list[Tool],
    *,
    max_results: int,
    lifecycle_graph: Any | None = None,
    macro_route_catalog: Any | None = None,
    retrieval_index: Any | None = None,
) -> list[str]:
    """Rank tools for a ToolSearch query and return tool names.

    Order (§8.3):
      direct macro route → lifecycle-chain → normal scoring → lifecycle reorder
    """
    macro_route_matches, excluded_tools, exclusive = _direct_macro_route_names(query, tools, macro_route_catalog)
    # exclusive: truncate to the macro only. prefer: fall through and prepend.
    if exclusive and macro_route_matches:
        return macro_route_matches[:max_results]

    chain_matches = _lifecycle_chain_tool_names(query, tools, lifecycle_graph)
    if chain_matches:
        if macro_route_matches:
            merged: list[str] = []
            for name in macro_route_matches + chain_matches:
                if name not in merged:
                    merged.append(name)
            return merged[:max_results]
        return chain_matches[:max_results]

    excluded_set = set(excluded_tools)
    scored: list[tuple[tuple[int, int, int, str], str]] = []
    for tool in tools:
        if tool.name in excluded_set:
            continue
        name, search_hint, document = _tool_search_parts(tool)
        key = score_tool_match(
            query,
            tool_name=name,
            search_hint=search_hint,
            document=document,
        )
        if key is not None:
            scored.append((key, tool.name))

    scored.sort(
        key=lambda item: (
            _lifecycle_sort_key(query, item[1], lifecycle_graph),
            item[0][0],
            item[0][1],
            item[0][2],
            -_name_segment_overlap(tokenize_query(query), item[1]),
            item[1],
        )
    )
    score_tiers = {name: key[0] for key, name in scored}
    if (
        scored
        and scored[0][0][0] == 0
        and _lifecycle_route_for_query(query, lifecycle_graph) is None
        and not macro_route_matches
    ):
        tier_zero = [name for key, name in scored if key[0] == 0]
        tier_zero = _apply_macro_structural_bias(
            tier_zero,
            score_tiers,
            retrieval_index,
        )
        return tier_zero[:max_results]
    ranked = [name for _, name in scored]
    ranked = rank_tools_by_lifecycle(ranked, query, lifecycle_graph)
    ranked = _apply_macro_structural_bias(ranked, score_tiers, retrieval_index)

    if macro_route_matches:
        merged: list[str] = []
        for name in macro_route_matches + ranked:
            if name not in merged:
                merged.append(name)
        return merged[:max_results]
    return ranked[:max_results]
