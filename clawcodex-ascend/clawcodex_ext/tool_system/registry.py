# pylint: disable=redefined-builtin
from __future__ import annotations

import inspect
import logging
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .build_tool import Tool, Tools
from .context import ToolContext
from .protocol import ToolCall, ToolResult
from .schema_validation import validate_tool_input
from src.permissions.check import has_permissions_to_use_tool
from src.permissions.handler import handle_permission_ask
from src.permissions.types import (
    PermissionAskDecision,
    PermissionAskHandler,
    PermissionAskReply,
    PermissionAskRequest,
    PermissionUpdate,
    PermissionUpdateAddDirectories,
    ToolPermissionContext,
)

log = logging.getLogger(__name__)


# Tools that modify files on disk — blocked in plan mode unless targeting
# a temporary / scratch path.
_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"Bash", "Edit", "Write", "NotebookEdit"})

# Resolved once: system temp directory (e.g. /tmp, %TEMP%).
_SYSTEM_TEMP_DIR: Path = Path(tempfile.gettempdir()).resolve()


def _adapt_permission_handler(raw_handler: Any) -> PermissionAskHandler:
    """Bridge legacy ``(tool_name, message, suggestion) -> (allowed, enable)`` handlers
    to the :class:`PermissionAskRequest` / :class:`PermissionAskReply` API expected by
    :func:`handle_permission_ask`.
    """

    sig = inspect.signature(raw_handler)
    param_count = len(sig.parameters)

    if param_count == 1:

        def _passthrough(request: PermissionAskRequest) -> PermissionAskReply:
            result = raw_handler(request)
            if isinstance(result, PermissionAskReply):
                return result
            if isinstance(result, tuple) and result:
                allowed = bool(result[0])
                return PermissionAskReply(
                    behavior="allow" if allowed else "deny",
                )
            raise TypeError(f"permission_handler returned unexpected type: {type(result).__name__}")

        return _passthrough

    def _legacy(request: PermissionAskRequest) -> PermissionAskReply:
        allowed, _enable = raw_handler(request.tool_name, request.message, None)
        return PermissionAskReply(behavior="allow" if allowed else "deny")

    return _legacy


def _apply_and_persist_updates(context: ToolContext, updates: tuple[PermissionUpdate, ...]) -> None:
    """Apply accepted "don't ask again" updates and persist them.

    In-memory application makes the rule effective for later dispatches
    through THIS ToolContext. Persistence (for userSettings/projectSettings/
    localSettings destinations) makes it survive restarts, read back at
    startup via ``setup_permissions``. Both halves are best-effort: a failed
    settings write must never fail the already-approved tool call — but it is
    logged, since the user was just promised "don't ask again".
    """

    from src.permissions.settings_paths import settings_path_for_destination
    from clawcodex_ext.permissions.updates import (
        apply_permission_updates,
        persist_permission_updates,
        supports_persistence,
    )

    try:
        updates_list = list(updates)
        base_ctx = context.permission_context
        prev_mode = getattr(base_ctx, "mode", None)
        # Plan-mode port: a dialog-accepted setMode (e.g. the plan-approval
        # "Yes, auto-accept edits") changes the LIVE mode mid-turn. Run the
        # transition seam (plan enter/exit attachment flags, pre_plan_mode
        # stash/clear) BEFORE applying — the entry-side stash reads the
        # PRE-switch mode, so running it after apply would no-op.
        final_mode = prev_mode
        for u in updates_list:
            if getattr(u, "type", "") == "setMode":
                final_mode = getattr(u, "mode", final_mode)
        mode_changed = final_mode != prev_mode and prev_mode is not None and final_mode is not None
        if mode_changed:
            try:
                from src.permissions.plan_transitions import (
                    transition_permission_mode,
                )

                base_ctx = transition_permission_mode(prev_mode, final_mode, base_ctx)
            except Exception:  # noqa: BLE001  # nosec — transition side effects are best-effort
                log.debug("permission-mode transition seam failed", exc_info=True)
        # apply_permission_updates returns a FRESH context (input unchanged)
        # — rebind it so every later dispatch sees the new rules.
        context.permission_context = apply_permission_updates(base_ctx, updates_list)
        if mode_changed:
            cb = getattr(context, "on_permission_mode_change", None)
            if cb is not None:
                try:
                    cb(str(final_mode))
                except Exception:  # noqa: BLE001  # nosec
                    log.debug("permission-mode change notify failed", exc_info=True)
    except Exception:  # nosec
        log.exception("failed to apply accepted permission updates in-memory")
    try:
        cwd = str(context.workspace_root) if context.workspace_root else None
        results = persist_permission_updates(
            list(updates),
            settings_path_for_destination=lambda destination: (settings_path_for_destination(destination, cwd)),
        )
        for update, ok in zip(updates, results):
            if not ok and supports_persistence(update.destination):
                log.warning(
                    "permission update not persisted (destination=%s); the rule applies this session only",
                    update.destination,
                )
    except Exception:  # nosec
        log.exception("failed to persist accepted permission updates")


def _one_shot_directory_updates(
    decision: PermissionAskDecision,
) -> tuple[PermissionUpdate, ...]:
    """Return directory grants needed only for the currently approved call."""

    return tuple(update for update in decision.suggestions or () if isinstance(update, PermissionUpdateAddDirectories))


def _is_temp_path(tool_name: str, tool_input: dict[str, Any], context: ToolContext) -> bool:
    """Return True if the tool targets a temporary / scratch path.

    Temporaries are defined as:
    - the system temp directory (``tempfile.gettempdir()``, e.g. ``/tmp``,
      ``/var/folders/…``, ``%TEMP%``)
    - any path whose components include ``.clawcodex``
      (e.g. ``/home/user/project/.clawcodex/plan.md``)
    - any path whose components include ``.reports``
      (e.g. ``/home/user/project/.reports/1.md``)
    - the worktree root, if set (``context.worktree_root``) — worktrees are
      short-lived scratch copies of a repository

    Bash is assigned ``False`` unconditionally because its file-modification
    intent cannot be determined statically — the system prompt already
    instructs the LLM not to execute commands in plan mode.
    """
    if tool_name == "Bash":
        return False

    file_path = tool_input.get("file_path") or tool_input.get("filePath")
    if not isinstance(file_path, str):
        return False

    p = Path(file_path).expanduser()
    if not p.is_absolute():
        base = context.cwd or context.workspace_root
        p = (base / p).resolve()
    else:
        p = p.resolve()

    # System temp directory (e.g. /tmp, /var/folders/..., %TEMP%).
    try:
        p.relative_to(_SYSTEM_TEMP_DIR)
        return True
    except ValueError:
        pass

    # Worktree root — a short-lived scratch copy of a repository.
    if context.worktree_root is not None:
        try:
            p.relative_to(context.worktree_root)
            return True
        except ValueError:
            pass

    # Path contains ".clawcodex" as a path component
    # (e.g. /home/user/project/.clawcodex/plan.md).
    if ".clawcodex" in p.parts:
        return True

    # Path contains ".reports" as a path component
    # (e.g. /home/user/project/.reports/1.md).
    if ".reports" in p.parts:
        return True

    return False


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: Tools = []
        self._by_name: dict[str, Tool] = {}
        # MCP servers whose tools are hidden (the original's MCPServerMultiselect-
        # Dialog). A tool named ``mcp__<server>__<tool>`` is hidden when <server>
        # is in this set — so list_tools() (the agent's view) excludes it.
        self.disabled_servers: set[str] = set()
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        key = tool.name.lower()
        if key in self._by_name:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools.append(tool)
        self._by_name[key] = tool
        added_alias_keys: list[str] = []
        try:
            for alias in tool.aliases:
                alias_key = alias.lower()
                if alias_key in self._by_name:
                    raise ValueError(f"duplicate tool alias: {alias}")
                self._by_name[alias_key] = tool
                added_alias_keys.append(alias_key)
        except ValueError:
            self._tools = [t for t in self._tools if t.name != tool.name]
            self._by_name.pop(key, None)
            for alias_key in added_alias_keys:
                self._by_name.pop(alias_key, None)
            raise

    def unregister(self, name: str) -> bool:
        """Remove a tool by name (and its aliases). Returns True if it existed."""
        tool = self._by_name.get(name.lower())
        if tool is None:
            return False
        self._tools = [t for t in self._tools if t.name != tool.name]
        self._by_name.pop(tool.name.lower(), None)
        for alias in tool.aliases:
            self._by_name.pop(alias.lower(), None)
        return True

    def remove_tool(self, name: str) -> bool:
        """ch15 round-4 — remove a tool (and its aliases) by name. Returns
        True if a tool was removed. Needed so MCP tools can be SWAPPED live
        when a server sends notifications/tools/list_changed — the registry
        was otherwise append-only, so a re-fetch couldn't reach the agent.

        Also backs ``_filter_registry`` (--allowed-tools/--disallowed-tools in
        agent_server.py + headless.py): those call sites now call this method
        directly. They historically called a non-existent
        ``registry.unregister`` inside a try/except, so the registry-level
        filtering silently no-op'd — the flags removed nothing from the pool
        the model saw. Both paths only ever REMOVE tools, so activating them
        can only narrow the toolset.

        Resolving by ``name`` accepts a primary name OR an alias, then drops
        the tool from ``_tools`` and EVERY ``_by_name`` key that still points
        at it (canonical + aliases). Removing by an alias therefore also
        clears the canonical key, so the tool can't stay dispatch-reachable
        through a sibling key.
        """
        tool = self._by_name.get(name.lower())
        if tool is None:
            return False
        self._tools = [t for t in self._tools if t is not tool]
        for key in (tool.name, *(getattr(tool, "aliases", ()) or ())):
            # Only drop a key that still points at THIS tool (another tool may
            # have claimed an alias — don't clobber that).
            if self._by_name.get(key.lower()) is tool:
                self._by_name.pop(key.lower(), None)
        return True

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name.lower())

    def canonicalize_names(self, names: Iterable[str]) -> set[str]:
        """Resolve each requested name to its tool's lowercased PRIMARY name.

        Backs ``--allowed-tools`` / ``--disallowed-tools`` so they match whether
        the caller passes a tool's primary name or one of its aliases
        (e.g. ``KillShell`` -> ``TaskStop``, ``Task`` -> ``Agent``).
        ``list_tools()`` only yields primary names and ``remove_tool`` cleans a
        tool's aliases with it, so canonicalizing the request set to primary
        names is what makes alias-form flags actually filter. Unknown names
        pass through lowercased — they simply match nothing.
        """
        out: set[str] = set()
        for name in names:
            if not name or not name.strip():
                # Skip blanks so a stray "" can't become a match-nothing
                # allowlist that removes every tool.
                continue
            tool = self.get(name)
            out.add((tool.name if tool else name).lower())
        return out

    def list_tools(self) -> Tools:
        """Tools visible to the agent — excludes disabled-MCP-server tools."""
        if not self.disabled_servers:
            return list(self._tools)
        return [t for t in self._tools if not self._server_disabled(t.name)]

    def all_tools(self) -> Tools:
        """Every registered tool, including hidden ones (for the multiselect UI)."""
        return list(self._tools)

    def _server_disabled(self, tool_name: str) -> bool:
        return any(tool_name.startswith(f"mcp__{s}__") for s in self.disabled_servers)

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        from extensions.sop_converter.runtime.macros.resolve_tool import (
            resolve_tool_for_context,
        )

        tool = resolve_tool_for_context(context, call.name, base_registry=self)
        if tool is None:
            return ToolResult(
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )

        plan = context.retrieval_plan
        selected_macros = set(plan.selected_macros) if plan is not None else set()
        selected_macro_call = tool.name in selected_macros
        if tool.name in context.retrieval_suppressed_tools and not context.workflow_stack:
            context.retrieval_metrics["shadowed_atomic_call_count"] = (
                int(context.retrieval_metrics.get("shadowed_atomic_call_count", 0)) + 1
            )
            recommended = plan.selected_macros[0] if plan and plan.selected_macros else ""
            return ToolResult(
                name=tool.name,
                output={
                    "status": "error",
                    "error": f"tool is shadowed by macro: {recommended}",
                    "error_code": "tool_shadowed_by_macro",
                    "recommended_tool": recommended,
                    "intent_key": plan.intent_key if plan else None,
                    "retryable": True,
                },
                is_error=True,
                tool_use_id=call.tool_use_id,
            )

        context.ensure_tool_allowed(tool.name)
        # Semantic-coerce + validate; the coerced input (string "true"/"30" →
        # bool/number) replaces the raw model input for permissions and call,
        # mirroring TS carrying ``parsedInput.data`` forward.
        coerced_input = validate_tool_input(tool.name, call.input, tool.input_schema)
        if coerced_input is not call.input:
            call = ToolCall(
                name=call.name,
                input=coerced_input,
                tool_use_id=call.tool_use_id,
            )

        # Downstream plan-mode defense-in-depth: the permission layer is the
        # primary gate, while direct registry callers still cannot mutate
        # project files (temporary/scratch paths remain available).
        if context.plan_mode and tool.name in _DESTRUCTIVE_TOOLS:
            if not _is_temp_path(tool.name, call.input, context):
                return ToolResult(
                    name=call.name,
                    output={
                        "error": (
                            f"Plan mode: {tool.name} is not allowed on project files. "
                            "Exit plan mode first via ExitPlanMode to modify files."
                        )
                    },
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )

        if tool.validate_input is not None:
            validation = tool.validate_input(call.input, context)
            if not validation.result:
                validation_output: dict[str, Any] = {"error": validation.message}
                if validation.error_code:
                    validation_output["error_code"] = validation.error_code
                return ToolResult(
                    name=tool.name,
                    output=validation_output,
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )

        decision = has_permissions_to_use_tool(
            tool,
            call.input,
            context.permission_context,
            tool_use_context=context,
        )

        if decision.behavior == "deny":
            return ToolResult(
                name=tool.name,
                output={"error": getattr(decision, "message", None) or "permission denied"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )

        restore_permission_context: ToolPermissionContext | None = None

        if decision.behavior == "ask":
            assert isinstance(decision, PermissionAskDecision)
            handler_cb: PermissionAskHandler | None = None
            if context.permission_handler is not None:
                handler_cb = _adapt_permission_handler(context.permission_handler)

            final, chosen_updates = handle_permission_ask(
                tool.name,
                decision,
                handler_cb,
                tool_input=call.input,
                context=context,
                tool_use_id=call.tool_use_id,
            )

            if final.behavior == "deny":
                return ToolResult(
                    name=tool.name,
                    output={"error": getattr(final, "message", None) or "permission denied by user"},
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )

            if chosen_updates:
                _apply_and_persist_updates(context, chosen_updates)
            else:
                one_shot_updates = _one_shot_directory_updates(decision)
                if one_shot_updates:
                    from clawcodex_ext.permissions.updates import (
                        apply_permission_updates,
                    )

                    restore_permission_context = context.permission_context
                    context.permission_context = apply_permission_updates(
                        context.permission_context,
                        list(one_shot_updates),
                    )

            if hasattr(final, "updated_input") and final.updated_input:
                call = ToolCall(
                    name=call.name,
                    input=final.updated_input,
                    tool_use_id=call.tool_use_id,
                )

        elif decision.behavior == "allow":
            if hasattr(decision, "updated_input") and decision.updated_input:
                call = ToolCall(
                    name=call.name,
                    input=decision.updated_input,
                    tool_use_id=call.tool_use_id,
                )

        try:
            try:
                result = _invoke_tool_call(tool, call.input, context)
            finally:
                # Once the selected macro has started, do not carry its
                # suppression overlay into later unrelated model turns.  The
                # workflow's internal atomic calls are allowed while
                # ``workflow_stack`` is non-empty.
                if selected_macro_call:
                    context.restore_retrieval_tools()
        finally:
            if restore_permission_context is not None:
                context.permission_context = restore_permission_context
        if result.tool_use_id is None and call.tool_use_id is not None:
            return ToolResult(
                name=result.name,
                output=result.output,
                is_error=result.is_error,
                tool_use_id=call.tool_use_id,
                content_type=result.content_type,
                new_messages=result.new_messages,
                context_modifier=result.context_modifier,
            )
        return result


def _invoke_tool_call(tool: Any, input: dict, context: ToolContext) -> ToolResult:
    """Dispatch a tool's ``call`` — sync or async — and return its
    ``ToolResult``.

    Chapter-10 / Chunk D / WI-4.0: the ``Tool.call`` signature was
    historically sync-only, but TaskOutputTool's polling loop (WI-4.1)
    needs async; broadening the dispatcher here unblocks that without
    rippling through every existing tool. Sync tools are dispatched
    unchanged; async tools (``inspect.iscoroutinefunction(tool.call)``
    is True) are awaited via ``asyncio.run`` if no loop is active or
    via a thread-bridge if we're already inside one (the same pattern
    Chunk B introduced for TaskStop's async-kill bridge).
    """
    fn = tool.call
    if not inspect.iscoroutinefunction(fn):
        return fn(input, context)

    # Async tool — drive the coroutine to completion via the shared bridge
    # (extracted to utils/async_bridge in HOOKS-1 so the permission-ask
    # seam reuses the exact same run-or-thread semantics). No timeout on
    # the wait — async tools are expected to self-bound (TaskOutput uses
    # its own ``timeout`` knob; TaskStop uses ``asyncio.wait_for`` inside
    # its body). A hang here is a tool bug, not something the dispatcher
    # papers over with a global cap.
    from src.utils.async_bridge import run_coroutine_blocking

    return run_coroutine_blocking(
        fn(input, context),
        thread_name=f"tool-async-bridge:{getattr(tool, 'name', '?')}",
    )


def get_all_base_tools(registry: ToolRegistry) -> Tools:
    return registry.list_tools()


def filter_tools_by_deny_rules(
    tools: Tools,
    permission_context: ToolPermissionContext,
) -> Tools:
    return [t for t in tools if not permission_context.blocks(t.name)]


def get_tools(
    registry: ToolRegistry,
    permission_context: ToolPermissionContext,
) -> Tools:
    all_tools = get_all_base_tools(registry)
    allowed = filter_tools_by_deny_rules(all_tools, permission_context)
    return [t for t in allowed if t.is_enabled()]


def assemble_tool_pool(
    registry: ToolRegistry,
    permission_context: ToolPermissionContext,
    mcp_tools: Tools | None = None,
) -> Tools:
    """Assemble the full tool pool sent to the API.

    Built-ins are sorted and concatenated FIRST, then MCP tools sorted
    and concatenated AFTER. Built-ins win on name collision (insertion
    order preserved).

    The split is not aesthetic: the API server places its prompt-cache
    breakpoint after the last built-in tool. A flat sort across all
    tools would interleave MCP tools mid-list, and adding or removing
    one MCP tool would shift the position of every built-in that sorts
    after it -- invalidating the cache key for the whole tool block.
    Keeping built-ins as a contiguous prefix means MCP churn doesn't
    cost a cache miss on the built-in schemas. See
    ``claude-code-from-source/book/ch06-tools.md`` section
    "assembleToolPool: Merging Built-in and MCP Tools".
    """
    builtin_tools = get_tools(registry, permission_context)
    if not mcp_tools:
        builtin_tools.sort(key=lambda t: t.name)
        return builtin_tools

    allowed_mcp = filter_tools_by_deny_rules(mcp_tools, permission_context)
    builtin_tools.sort(key=lambda t: t.name)
    allowed_mcp.sort(key=lambda t: t.name)

    seen: set[str] = set()
    merged: Tools = []
    for t in builtin_tools:
        if t.name not in seen:
            seen.add(t.name)
            merged.append(t)
    for t in allowed_mcp:
        if t.name not in seen:
            seen.add(t.name)
            merged.append(t)
    return merged


def get_merged_tools(
    registry: ToolRegistry,
    permission_context: ToolPermissionContext,
    mcp_tools: Tools | None = None,
) -> Tools:
    builtin_tools = get_tools(registry, permission_context)
    if not mcp_tools:
        return builtin_tools
    return builtin_tools + list(mcp_tools)
