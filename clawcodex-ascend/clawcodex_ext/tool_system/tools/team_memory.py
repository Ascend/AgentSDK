"""F-93 TeamMem — TeamMemoryTool (P93-F).

Agent-facing tool that exposes :class:`TeamMemoryService` to the LLM.
One tool, five actions (F-93 §1.9): ``remember`` / ``recall`` / ``list``
/ ``delete`` / ``compact``.

The tool is gated by ``is_team_memory_enabled()`` (env +
auto-memory). When disabled, ``is_enabled`` returns ``False`` and the
tool registry skips registration — the agent never sees it.

This is a Layer 1 module (``clawcodex_ext/``): it imports the Layer 2
service from ``extensions/agents/`` and the upstream tool primitives
from ``clawcodex_ext.tool_system``. No ``src/`` file is modified.
"""

from __future__ import annotations

import logging
from typing import Any

from clawcodex_ext.memdir.team_mem_paths import is_team_memory_enabled

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult

logger = logging.getLogger(__name__)

TEAM_MEMORY_TOOL_NAME = "TeamMemory"

_VALID_ACTIONS = frozenset({"remember", "recall", "list", "delete", "compact"})
_VALID_SOURCES = frozenset({"manual", "send_message", "task_result", "review", "system"})
_VALID_SCOPES = frozenset({"team", "lead_only", "agent_pair"})


def _err(message: str, **extra: Any) -> ToolResult:
    payload: dict[str, Any] = {"success": False, "message": message}
    payload.update(extra)
    return ToolResult(name=TEAM_MEMORY_TOOL_NAME, output=payload, is_error=True)


def _ok(message: str, **extra: Any) -> ToolResult:
    payload: dict[str, Any] = {"success": True, "message": message}
    payload.update(extra)
    return ToolResult(name=TEAM_MEMORY_TOOL_NAME, output=payload)


def _resolve_agent_id(context: ToolContext) -> str:
    """Best-effort agent id for permission checks.

    The team-memory policy keys off ``agent_id`` (the team lead id or a
    teammate's spawn id). On ``ToolContext`` this lives at
    ``context.agent_id``; when unset (e.g. ad-hoc SDK call) we fall back
    to the team lead id from the team dict, then to ``"unknown"`` —
    the policy will refuse ``"unknown"`` as a non-member, which is the
    correct fail-closed behavior.
    """
    agent_id = getattr(context, "agent_id", None)
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    team = getattr(context, "team", None)
    if isinstance(team, dict):
        lead = team.get("lead_agent_id")
        if isinstance(lead, str) and lead:
            return lead
    return "unknown"


def _team_memory_service(context: ToolContext):
    """Build a :class:`TeamMemoryService` from the tool context.

    Imports lazily so the tool module import is cheap and the
    extensions/agents layer is only loaded when the tool is actually
    invoked. Returns ``None`` and sets a disabled message when team
    memory is off or no team file exists.
    """
    from extensions.agents.team_memory import (
        TeamMemoryConfig,
        TeamMemoryDisabledError,
        TeamMemoryService,
        TeamNotFoundError,
    )

    if not is_team_memory_enabled():
        return None, TeamMemoryDisabledError("team memory disabled (env/auto-memory off)")
    try:
        service = TeamMemoryService(
            workspace_root=context.workspace_root,
            config=TeamMemoryConfig(enabled=True),
        )
    except TeamNotFoundError as exc:
        return None, exc
    return service, None


def _team_memory_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    action = tool_input.get("action")
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        raise ToolInputError(f"'action' must be one of {sorted(_VALID_ACTIONS)}, got {action!r}")

    service, err = _team_memory_service(context)
    if service is None:
        # Disabled / no team → fail soft with a structured message rather
        # than raising; the agent can proceed without team memory.
        return _err(f"Team memory unavailable: {err}")

    agent_id = _resolve_agent_id(context)

    if action == "remember":
        return _action_remember(service, tool_input, agent_id)
    if action == "recall":
        return _action_recall(service, tool_input, agent_id)
    if action == "list":
        return _action_list(service, tool_input, agent_id)
    if action == "delete":
        return _action_delete(service, tool_input, agent_id)
    if action == "compact":
        return _action_compact(service, agent_id)
    # Unreachable — action validated above.
    raise ToolInputError(f"unsupported action {action!r}")  # pragma: no cover


def _action_remember(service, tool_input: dict[str, Any], agent_id: str) -> ToolResult:
    from extensions.agents.team_memory import (
        TeamMemoryDisabledError,
        TeamMemoryPermissionError,
        TeamMemoryTooLargeError,
    )

    content = tool_input.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ToolInputError("'content' must be a non-empty string for action=remember")
    summary = tool_input.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise ToolInputError("'summary' must be a string when provided")
    source = tool_input.get("source", "manual")
    if not isinstance(source, str) or source not in _VALID_SOURCES:
        raise ToolInputError(f"'source' must be one of {sorted(_VALID_SOURCES)}, got {source!r}")
    scope = tool_input.get("scope", "team")
    if not isinstance(scope, str) or scope not in _VALID_SCOPES:
        raise ToolInputError(f"'scope' must be one of {sorted(_VALID_SCOPES)}, got {scope!r}")
    tags = tool_input.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ToolInputError("'tags' must be a list of strings")
    related = tool_input.get("related_agents") or []
    if not isinstance(related, list) or not all(isinstance(r, str) for r in related):
        raise ToolInputError("'related_agents' must be a list of strings")
    confidence = tool_input.get("confidence", 1.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("'confidence' must be a number") from exc
    try:
        entry = service.remember(
            content,
            author_agent_id=agent_id,
            tags=tags,
            source=source,
            scope=scope,
            related_agents=related,
            summary=summary,
            confidence=confidence,
        )
    except TeamMemoryPermissionError as exc:
        return _err(str(exc), action="remember")
    except TeamMemoryTooLargeError as exc:
        return _err(str(exc), action="remember")
    except TeamMemoryDisabledError as exc:
        return _err(str(exc), action="remember")
    return _ok(
        f"Remembered team memory entry {entry.id}.",
        action="remember",
        entry_id=entry.id,
        summary=entry.summary,
    )


def _action_recall(service, tool_input: dict[str, Any], agent_id: str) -> ToolResult:
    from extensions.agents.team_memory import TeamMemoryQuery

    query = tool_input.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolInputError("'query' must be a non-empty string for action=recall")
    top_k = tool_input.get("top_k", 8)
    try:
        top_k_int = int(top_k)
        if top_k_int <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ToolInputError("'top_k' must be a positive integer") from exc
    tags = tool_input.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ToolInputError("'tags' must be a list of strings")
    results = service.recall(
        TeamMemoryQuery(
            team_id=service.team_id,
            query=query,
            requester_agent_id=agent_id,
            top_k=top_k_int,
            tags=tuple(tags),
        )
    )
    return _ok(
        f"Recalled {len(results)} team memory entries.",
        action="recall",
        results=[
            {
                "entry_id": r.entry.id,
                "summary": r.entry.summary,
                "score": round(r.score, 4),
                "matched_terms": list(r.matched_terms),
                "tags": list(r.entry.tags),
                "source": r.entry.source,
                "scope": r.entry.scope,
                "created_at": r.entry.created_at,
            }
            for r in results
        ],
    )


def _action_list(service, tool_input: dict[str, Any], agent_id: str) -> ToolResult:
    limit = tool_input.get("limit", 50)
    try:
        limit_int = int(limit)
        if limit_int <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ToolInputError("'limit' must be a positive integer") from exc
    tags = tool_input.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ToolInputError("'tags' must be a list of strings")
    sources = tool_input.get("sources") or []
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ToolInputError("'sources' must be a list of strings")
    entries = service.list_entries(
        requester_agent_id=agent_id,
        limit=limit_int,
        tags=tags,
        sources=sources,
    )
    return _ok(
        f"Listed {len(entries)} team memory entries.",
        action="list",
        entries=[
            {
                "entry_id": e.id,
                "summary": e.summary,
                "tags": list(e.tags),
                "source": e.source,
                "scope": e.scope,
                "author": e.author_agent_id,
                "created_at": e.created_at,
            }
            for e in entries
        ],
    )


def _action_delete(service, tool_input: dict[str, Any], agent_id: str) -> ToolResult:
    from extensions.agents.team_memory import TeamMemoryPermissionError

    entry_id = tool_input.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        raise ToolInputError("'entry_id' must be a non-empty string for action=delete")
    reason = tool_input.get("reason") or "tool delete"
    if not isinstance(reason, str):
        raise ToolInputError("'reason' must be a string when provided")
    try:
        removed = service.delete(entry_id, actor=agent_id, reason=reason)
    except TeamMemoryPermissionError as exc:
        return _err(str(exc), action="delete")
    if not removed:
        return _err(f"Entry {entry_id!r} not found or already deleted.", action="delete")
    return _ok(f"Deleted team memory entry {entry_id}.", action="delete", entry_id=entry_id)


def _action_compact(service, agent_id: str) -> ToolResult:
    from extensions.agents.team_memory import TeamMemoryPermissionError

    try:
        entry = service.compact(actor=agent_id)
    except TeamMemoryPermissionError as exc:
        return _err(str(exc), action="compact")
    return _ok(
        f"Compacted team memory into summary entry {entry.id}.",
        action="compact",
        entry_id=entry.id,
        summary=entry.summary,
    )


TeamMemoryTool: Tool = build_tool(
    name=TEAM_MEMORY_TOOL_NAME,
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_VALID_ACTIONS),
                "description": "Team memory operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Content to remember (action=remember).",
            },
            "summary": {
                "type": "string",
                "description": "Optional one-line summary (action=remember).",
            },
            "source": {
                "type": "string",
                "enum": sorted(_VALID_SOURCES),
                "description": "Entry source (action=remember). Defaults to manual.",
            },
            "scope": {
                "type": "string",
                "enum": sorted(_VALID_SCOPES),
                "description": "Visibility scope (action=remember). Defaults to team.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to attach or filter by (action=remember/recall/list).",
            },
            "related_agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agents paired with this entry (action=remember, scope=agent_pair).",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence weight 0..1 (action=remember). Defaults to 1.0.",
            },
            "query": {
                "type": "string",
                "description": "Search query (action=recall).",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (action=recall). Defaults to 8.",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to list (action=list). Defaults to 50.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by source (action=list).",
            },
            "entry_id": {
                "type": "string",
                "description": "Entry to delete (action=delete).",
            },
            "reason": {
                "type": "string",
                "description": "Reason for deletion (action=delete).",
            },
        },
    },
    call=_team_memory_call,
    prompt=(
        "Read, write, search, and compact the team's shared long-term memory. "
        "Use action=remember to persist cross-agent knowledge, action=recall to "
        "find relevant entries, action=list to browse, action=delete to tombstone, "
        "and action=compact to collapse history into a summary."
    ),
    description="Team shared memory: remember / recall / list / delete / compact.",
    strict=True,
    max_result_size_chars=100_000,
    is_read_only=lambda inp: inp.get("action") in {"recall", "list"},
    is_concurrency_safe=lambda _inp: False,
    is_enabled=is_team_memory_enabled,
    to_auto_classifier_input=lambda inp: (inp or {}).get("action", "") or "",
)
