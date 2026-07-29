"""SkillSearch tool — Agent-facing skill search via TF-IDF.

Provides the ``SkillSearch`` tool that allows agents to search for
relevant skills, manage pinned skills, inspect skill tokenization, and
rebuild the search index.

Actions:
    - ``search``: Rank skills by TF-IDF relevance to a query.
    - ``pin`` / ``unpin``: Manage pinned (high-priority) skills.
    - ``inspect``: Show per-field token breakdown for a skill.
    - ``rebuild``: Force-rebuild the index from the registry.
    - ``stats``: Show index statistics (doc count, term count, etc.).
"""

from __future__ import annotations

from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult


def _result(output: Any, *, is_error: bool = False) -> ToolResult:
    return ToolResult(name="SkillSearch", output=output, is_error=is_error)


def _get_searcher():
    """Lazily create and cache a SkillSearcher singleton."""
    from extensions.skills_ext.registry_ext import get_default_registry

    from clawcodex_ext.services.skill_search.config import SkillSearchConfig
    from clawcodex_ext.services.skill_search.searcher import SkillSearcher
    from clawcodex_ext.services.skill_search.tokenizer import create_default_tokenizer

    searcher: SkillSearcher | None = getattr(_get_searcher, "_instance", None)
    if searcher is None:
        config = SkillSearchConfig.from_feature_gate()
        registry = get_default_registry()
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)
        _get_searcher._instance = searcher  # type: ignore[attr-defined]
        # Start watcher for incremental index updates (P92-E).
        if config.enabled:
            searcher.create_watcher().start()
    return searcher


def _skill_search_call(input_data: dict[str, Any], context: ToolContext) -> ToolResult:
    """Dispatch to the appropriate action handler."""
    action = input_data.get("action", "search")

    if action == "search":
        return _handle_search(input_data)
    if action == "pin":
        return _handle_pin(input_data)
    if action == "unpin":
        return _handle_unpin(input_data)
    if action == "inspect":
        return _handle_inspect(input_data)
    if action == "rebuild":
        return _handle_rebuild()
    if action == "stats":
        return _handle_stats()
    return _result(
        f"Unknown action: {action}. Valid actions: search, pin, unpin, inspect, rebuild, stats",
        is_error=True,
    )


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_search(input_data: dict[str, Any]) -> ToolResult:
    import asyncio

    query = input_data.get("query", "")
    if not query:
        return _result(
            "query is required for search action — provide a natural language description of the skill you need",
            is_error=True,
        )

    top_k = input_data.get("top_k")
    tags = input_data.get("tags")
    source = input_data.get("source")

    searcher = _get_searcher()

    try:
        results = asyncio.run(searcher.search(query, top_k=top_k, tags=tags, source=source))
    except Exception as e:
        return _result(f"Search for '{query}' failed: {e}", is_error=True)

    if not results:
        return _result("No matching skills found.")

    lines = [f'Search results for "{query}":', ""]
    for i, r in enumerate(results, 1):
        doc = r.document
        lines.append(f"{i}. {doc.name}  (score: {r.score:.3f}, source: {doc.source})")
        if doc.description:
            lines.append(f"   {doc.description}")
        if r.reason:
            lines.append(f"   {r.reason}")
        lines.append("")

    return _result("\n".join(lines))


def _handle_pin(input_data: dict[str, Any]) -> ToolResult:
    name = input_data.get("name", "")
    if not name:
        return _result(
            "name is required for pin action — use a skill name from search results",
            is_error=True,
        )

    searcher = _get_searcher()
    searcher.pin(name)
    return _result(f"Pinned skill: {name}")


def _handle_unpin(input_data: dict[str, Any]) -> ToolResult:
    name = input_data.get("name", "")
    if not name:
        return _result(
            "name is required for unpin action — specify the skill name to unpin",
            is_error=True,
        )

    searcher = _get_searcher()
    searcher.unpin(name)
    return _result(f"Unpinned skill: {name}")


def _handle_inspect(input_data: dict[str, Any]) -> ToolResult:
    import asyncio

    name = input_data.get("name", "")
    if not name:
        return _result(
            "name is required for inspect action — use a skill name from search results",
            is_error=True,
        )

    searcher = _get_searcher()

    try:
        asyncio.run(searcher.ensure_index())
    except Exception as e:
        return _result(f"Cannot inspect '{name}': {e}", is_error=True)

    result = searcher.inspect(name)
    if result is None:
        return _result(f"Skill not found in index: {name}")

    lines = [
        f"Inspect: {result.name}",
        f"  Source: {result.source}",
        f"  Total tokens: {result.token_count}",
        "",
        "  Per-field breakdown:",
    ]
    for field_name, field_info in result.fields.items():
        lines.append(f"    {field_name}: {field_info.token_count} tokens")
        if field_info.token_sample:
            sample = ", ".join(field_info.token_sample[:10])
            lines.append(f"      sample: {sample}")

    return _result("\n".join(lines))


def _handle_rebuild() -> ToolResult:
    import asyncio

    searcher = _get_searcher()
    try:
        asyncio.run(searcher.refresh())
        stats = searcher.stats()
        if stats:
            return _result(f"Index rebuilt: {stats.total_docs} docs, {stats.total_terms} terms")
        return _result("Index rebuilt successfully.")
    except Exception as e:
        return _result(
            f"Index rebuild failed: {e}. The feature gate or index may be unavailable.",
            is_error=True,
        )


def _handle_stats() -> ToolResult:
    import asyncio

    searcher = _get_searcher()

    try:
        asyncio.run(searcher.ensure_index())
    except Exception:  # nosec
        return _result("Index not loaded (feature flag may be off).")

    stats = searcher.stats()
    if stats is None:
        return _result("Index not loaded.")

    pinned = searcher.get_pinned()
    lines = [
        "Skill Search Index Stats",
        "=======================",
        f"  Documents:     {stats.total_docs}",
        f"  Unique terms:  {stats.total_terms}",
        f"  Inverted size: {stats.total_inverted_entries} entries",
        f"  Approx memory: {stats.approximate_bytes} bytes",
        f"  Pinned skills: {len(pinned)}",
    ]
    if pinned:
        lines.append(f"    {', '.join(pinned)}")

    return _result("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

SKILL_SEARCH_TOOL_PROMPT = """Use SkillSearch to find relevant skills for a task.
When you're unsure which skill to use, search with a natural language description
of what you want to do. The TF-IDF index will rank skills by relevance.

You can also pin frequently-used skills, inspect a skill's token representation,
or check index statistics with the stats action."""


SkillSearchTool: Tool = build_tool(
    name="SkillSearch",
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "pin", "unpin", "inspect", "rebuild", "stats"],
                "description": "Action to perform: search (find relevant skills), pin/unpin (manage pinned skills), inspect (show token breakdown), rebuild (force index rebuild), stats (index statistics)",
            },
            "query": {
                "type": "string",
                "description": 'Natural language description of what you want to do. E.g., "browser automation", "git commit helper". Required for search action.',
            },
            "name": {
                "type": "string",
                "description": "Skill name. Required for pin, unpin, and inspect actions.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 8). Only for search action.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter results to skills that have any of these tags. Only for search action.",
            },
            "source": {
                "type": "string",
                "enum": ["local", "project", "mcp", "template"],
                "description": "Filter results to skills from this source. Only for search action.",
            },
        },
        "required": ["action"],
    },
    call=_skill_search_call,
    prompt=SKILL_SEARCH_TOOL_PROMPT,
    description="Search for relevant skills, manage pinned skills, and inspect skill index",
    is_read_only=lambda _input: _input.get("action") not in ("pin", "unpin", "rebuild"),
    is_concurrency_safe=lambda _input: False,
    search_hint="skill search find relevant skills tfidf discover",
    to_auto_classifier_input=lambda _input: _input.get("query", ""),
)
