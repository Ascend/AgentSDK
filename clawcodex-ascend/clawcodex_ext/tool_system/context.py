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

# AgentSDK migration Parts do not contain every host facade during incremental validation.
# pylint: disable=C0201,E0611

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import ToolPermissionError
from .task_manager import TaskManager
from clawcodex_ext.permissions.types import ToolPermissionContext
from clawcodex_ext.services.swarm.agent_name_registry import AgentNameRegistry
from clawcodex_ext.task_registry import RuntimeTaskRegistry
from clawcodex_ext.utils.abort_controller import AbortController

from clawcodex_ext.query.outbox_types import OutboxEvent


def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(slots=True)
class ToolUseOptions:
    commands: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    debug: bool = False
    main_loop_model: str = ""
    verbose: bool = False
    thinking_config: dict[str, Any] | None = None
    mcp_clients: list[Any] = field(default_factory=list)
    mcp_resources: dict[str, list[Any]] = field(default_factory=dict)
    is_non_interactive_session: bool = False
    agent_definitions: dict[str, Any] = field(default_factory=dict)
    max_budget_usd: float | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    query_source: str | None = None
    refresh_tools: Callable[[], list[Any]] | None = None
    provider_override: dict[str, str] | None = None
    hooks: dict[str, list[Any]] | None = None


@dataclass(slots=True)
class QueryChainTracking:
    chain_id: str = ""
    depth: int = 0


@dataclass(slots=True)
class FileReadingLimits:
    max_tokens: int | None = None
    max_size_bytes: int | None = None


@dataclass(slots=True)
class GlobLimits:
    max_results: int | None = None


@dataclass(slots=True)
class RetrievalPlan:
    """Turn-local decision for macro / atomic tool exposure."""

    query: str = ""
    intent_key: str | None = None
    selected_macros: list[str] = field(default_factory=list)
    suppressed_tools: list[str] = field(default_factory=list)
    selection: str = "normal"
    route_scope: str | None = None
    preflight_status: str = "pending"
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_key": self.intent_key,
            "selection": self.selection,
            "selected_layer": "macro" if self.selected_macros else "normal",
            "selected_macros": list(self.selected_macros),
            "suppressed_tools": list(self.suppressed_tools),
            "route_scope": self.route_scope,
            "preflight": self.preflight_status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(slots=True)
class ToolContext:
    workspace_root: Path
    permission_context: ToolPermissionContext = field(
        default_factory=lambda: ToolPermissionContext(mode="bypassPermissions")
    )
    cwd: Path | None = None
    read_file_fingerprints: dict[Path, tuple[int, int] | tuple[int, int, bool]] = field(default_factory=dict)
    task_manager: TaskManager = field(default_factory=TaskManager)
    mcp_clients: dict[str, Any] = field(default_factory=dict)
    # Event loop that owns the active MCP stdio transports. Populated by
    # RuntimeContext when MCP servers are bootstrapped; generic MCP tool calls
    # use it to run client.call_tool on the correct loop.
    mcp_manager_loop: Any | None = field(default=None, repr=False)
    lsp_client: Any | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Chapter-10 / Chunk B / WI-1.3 — typed runtime-task registry. Houses
    # ``LocalShellTaskState`` / ``LocalAgentTaskState`` / etc. as
    # ``TaskStateBase`` subclasses. Replaces the un-typed
    # ``background_bash_tasks`` and ``_internal=True`` agent entries that
    # used to live on ``tasks``. ``runtime_tasks`` is the source of truth
    # for the chapter-10 task state machine; ``tasks`` continues to host
    # ``tasks_v2``/todo entries for the unrelated TaskCreate system.
    runtime_tasks: RuntimeTaskRegistry = field(default_factory=RuntimeTaskRegistry)
    # WI-5.1: per-message tool-result aggregate counter. The execution
    # pipeline (Step 11) reads + increments this each time a tool result
    # is mapped to its API form; when the running total exceeds
    # ``MAX_TOOL_RESULTS_PER_MESSAGE_CHARS`` (default 200K) the next
    # result is persisted to disk regardless of its individual size.
    # Reset to 0 between messages by the turn-loop dispatcher.
    #
    # ``_aggregate_lock`` synchronizes the read-decide-write across
    # concurrent tool dispatches (critic B6). ``_run_tools_partitioned``
    # uses ``asyncio.to_thread`` to fan out concurrency-safe tools (Read,
    # Grep, Glob) — without this lock, N threads would all read 0, all
    # decide their block is under the cap, and the per-message budget
    # would be silently bypassed. The full read+decide+write runs
    # serialized so the persistence decision uses the LIVE counter and
    # the cap is strictly enforced. Cost: the rare persist-to-disk path
    # serializes against the lock, but persists are O(1) per turn in
    # typical workloads (the common case under-threshold returns the
    # block without I/O).
    tool_result_chars_so_far: int = 0
    # Environment variables merged into every Bash subprocess env.
    # Values override inherited daemon env. Populated by the headless
    # entrypoint from workflow AgentConfig.
    env: dict[str, str] = field(default_factory=dict)
    # Base environment captured before live configuration overlays are applied.
    _configuration_env_base: dict[str, str] | None = None
    _aggregate_lock: threading.Lock = field(default_factory=threading.Lock)
    # Custom agent directory override (set by ``--agent <dir>``).
    # When non-None, ``get_agent_definitions`` will also scan this
    # directory's ``.claude/agents/`` for agent definitions.
    _agent_dir_override: Path | None = None
    # Session-cumulative tokens spent on client-side advisor calls.
    # ``src/tool_system/tools/advisor.py`` accumulates here on every
    # consultation; the status surface reads them to display
    # ``advisor: <in>/<out>`` next to the worker's
    # token counts. Distinct from ``tool_result_chars_so_far`` (which
    # is a per-message budget tied to API-result persistence) — these
    # are per-session totals for UI display.
    advisor_input_tokens: int = 0
    advisor_output_tokens: int = 0
    # Chapter-10 / Chunk F / WI-6.1 — agent-name registry. Maps the
    # human-readable ``name`` (passed via Agent({name: "researcher"}))
    # to the random ``agent_id`` returned by the spawn. SendMessage
    # consults this registry first when resolving a ``to:`` field;
    # falling back to "treat ``to`` as a raw agent_id" when the name
    # isn't registered preserves the legacy code path.
    #
    # Per Chunk-F-Phase-6 critic concern C1 (Phase-7 fix): the registry
    # is a typed ``AgentNameRegistry`` (not a bare dict) so the
    # collision check + claim is atomic under its own RLock. Two
    # concurrent same-name spawns can't both succeed.
    #
    # Collision policy (gap analysis ambiguity #2 + critic C2):
    # * spawn-name-collision-with-running task → AgentNameAlreadyClaimedError
    #   (translated to ToolInputError at the agent-tool boundary).
    # * spawn-name-collision-with-terminal task → silent overwrite;
    #   old terminal holders remain reachable by raw task_id + auto-
    #   resume (WI-7.4).
    agent_name_registry: AgentNameRegistry = field(default_factory=AgentNameRegistry)
    # Tool registry reference — allows CreateAgentTool to register new tools
    # into the active registry. Set by build_default_registry and runtime contexts.
    tool_registry: Any = None
    # Active provider reference -- set by query() before each tool-dispatch
    # turn so the client-side advisor (advisor.py) can reuse the same
    # provider (and its config) when advising the model. Read via
    # getattr(context, "_active_provider", None) at advisor.py:66.
    _active_provider: Any = None

    # Cron scheduler instance -- set by attach_cron_runtime() on REPL init.
    # Read via getattr(ctx, "cron_scheduler", None) at runtime.py:97.
    cron_scheduler: Any = None
    # Mutable flag shared by the TUI agent bridge and cron scheduler.  The
    # entrypoint installs the concrete flag object; declaring the slot here is
    # required because ToolContext uses ``slots=True`` and therefore rejects
    # ad-hoc attributes at runtime.
    _in_agent_loop: Any | None = field(default=None, repr=False)
    # Cron jitter config loader -- set by attach_cron_runtime() on REPL init.
    # Read via getattr(ctx, "cron_jitter_config", None) at runtime.py:97.
    cron_jitter_config: Any = None
    # Background Bash commands spawned via ``run_in_background: true``.
    # Kept as a deprecated dict-of-dicts compatibility view during the
    # Chunk-B migration cycle; the bash spawn writer now populates
    # ``runtime_tasks`` as the source of truth and mirrors the legacy dict
    # shape here so any external test fixtures or readers that haven't
    # migrated yet continue to work. Removed in a follow-up phase.
    background_bash_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    worktree_root: Path | None = None
    outbox: list[OutboxEvent] = field(default_factory=list)
    ask_user: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None
    crons: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Live scheduled-task engine (src.scheduled_tasks.SessionCronScheduler).
    # Set for the MAIN agent-server session only; when present the Cron*
    # and ScheduleWakeup tools register real firing jobs on it, and the
    # worker's idle branch runs due prompts as internal turns. When None
    # (subagents, SDK, bare tests) CronCreate falls back to the inert
    # legacy ``crons`` dict above.
    cron_scheduler: Any | None = None
    team: dict[str, Any] | None = None
    output_style_name: str | None = None
    output_style_dir: Path | None = None
    additional_working_directories: tuple[Path, ...] = ()
    allow_docs: bool = False

    permission_handler: Callable[[str, str, Optional[str]], tuple[bool, bool]] | None = None
    # Snapshot of the non-bypass permission handler, captured at
    # construction. The runtime permission controller
    # (``clawcodex_ext/permissions/runtime.py``) reads this on every
    # cycle out of ``bypassPermissions`` mode so the swap can restore
    # the original handler without reaching into the REPL/TUI's
    # internals. ``__post_init__`` defaults it to ``permission_handler``
    # when omitted so existing call sites keep working.
    default_permission_handler: Callable[..., Any] | None = None

    # Plan-mode port — fired whenever the LIVE permission mode changes off
    # the control channel (a chosen_updates setMode from a permission dialog,
    # EnterPlanMode/ExitPlanMode transitions). The agent-server wires this to
    # its status push (`{type:'system', subtype:'status', permission_mode}`,
    # the print.ts:1054-1073 analog) so the TUI footer badge tracks mid-turn
    # flips; None (SDK/tests) skips the notification.
    on_permission_mode_change: Callable[[str], None] | None = None

    options: ToolUseOptions = field(default_factory=ToolUseOptions)
    # Always present; callers that own the per-run cancellation lifecycle
    # (TUI bridge, REPL engine) overwrite this with their own controller
    # in ``submit()`` / ``__init__`` so tools, hooks, and subagents see
    # the same signal the UI trips. The default factory keeps the field
    # non-``None`` for unit tests and SDK callers that never explicitly
    # set it — readers can drop the historical ``if ctrl and …`` /
    # ``or AbortController()`` defensive checks that masked the "field
    # is None" hazard class.
    abort_controller: AbortController = field(default_factory=AbortController)
    messages: list[Any] = field(default_factory=list)
    set_response_length: Callable[[Callable[[int], int]], None] | None = None
    set_in_progress_tool_use_ids: Callable[[Callable[[set[str]], set[str]]], None] | None = None
    # Optional hook to stream a spawned subagent's live progress to the UI
    # (the Agent tool sets run_params.on_message → this). Wired only by the
    # agent-server (forwards an ``agent_progress`` message to the client); SDK /
    # REPL / unit-test paths leave it ``None`` and emit nothing.
    agent_progress_emit: Callable[[dict[str, Any]], None] | None = None
    # Mirrors TS Tool.ts:231
    # ``setHasInterruptibleToolInProgress?: (v: boolean) => void``.
    # Optional callback wired only in interactive (REPL/TUI) contexts; SDK
    # and unit-test paths leave it ``None`` and
    # ``StreamingToolExecutor._update_interruptible_state`` will skip the
    # call. The flag drives the UI's "press ESC to interrupt" indicator:
    # ``True`` only when at least one tool is currently executing AND every
    # executing tool's ``interrupt_behavior()`` returns ``"cancel"``. Fired
    # from ``StreamingToolExecutor._execute_tool`` on every transition into
    # or out of the ``executing`` status, matching TS lines 270, 290, 386.
    set_has_interruptible_tool_in_progress: Callable[[bool], None] | None = None
    query_tracking: QueryChainTracking | None = None
    file_reading_limits: FileReadingLimits | None = None
    glob_limits: GlobLimits | None = None
    content_replacement_state: Any | None = None
    agent_id: str | None = None
    # QUERY-1 — teammate identity (TS teammate.ts:125: a teammate requires
    # BOTH). Threaded by the Agent tool's NAMED spawn (run_agent →
    # create_subagent_context) when the parent carries a TeamCreate'd team;
    # None for plain subagents, so the teammate stop-hook block never fires
    # for them.
    teammate_name: str | None = None
    team_name: str | None = None
    agent_type: str | None = None
    # Main-loop agent definition for ``--agent <bundle_dir>`` (overview allowlist).
    startup_agent: Any | None = None
    bundle_context: Any | None = None
    workflow_stack: list[str] = field(default_factory=list)
    # layered ToolSearch. Suppression is reversible and scoped to the
    # current search/dispatch decision; it never unregisters or deletes tools.
    retrieval_plan: RetrievalPlan | None = None
    retrieval_hidden_tools: list[Any] = field(default_factory=list)
    retrieval_suppressed_tools: set[str] = field(default_factory=set)
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str | None = None
    user_modified: bool = False
    # Identifier of the active query/session. Surfaced to skills (SKILL.md
    # bodies may reference ``${CLAUDE_SESSION_ID}``) and any other tool
    # that needs to correlate with persisted session state. ``None`` is
    # interpreted as "unknown" by callers; substitutions yield an empty
    # string in that case.
    session_id: str | None = None
    # LKB Plan binding.  The workspace selects a shared Board, while this
    # session-scoped field selects one Plan graph inside it.  Subagents
    # inherit the value explicitly; unrelated sessions do not.
    lkb_plan_id: str | None = None
    # One-way Task-v2/LKB cutover provenance.  Tasks present when LKB is
    # first enabled remain native for the lifetime of this context; new
    # TaskCreate calls are LKB-owned.  The explicit set prevents a merged
    # projection from erasing source authority.
    lkb_native_task_ids: set[str] = field(default_factory=set)
    _lkb_task_cutover_initialized: bool = False
    # Throttle/signature cache for the optional LKB REPL/TUI projection
    # refresher. ToolContext uses slots, so the integration cannot attach
    # this state dynamically at runtime.
    _lkb_repl_status_state: dict[str, Any] | None = None
    # goal-mode model tools. ``session_id`` is the normal persisted
    # thread id; ``goal_thread_id`` lets tests or protocol adapters pin the
    # recoverable thread explicitly. ``goal_service`` is injected when a
    # caller needs an isolated GoalStore, otherwise tools construct the
    # default service lazily.
    goal_thread_id: str | None = None
    goal_service: Any | None = None
    goal_runtime: Any | None = None

    # session macros — overlay snapshot + registration gates.
    # Declared early (slots=True) so later tasks can construct real contexts.
    session_macro_overlay: Any | None = None
    confirm_session_macro_plan: Callable | None = None
    allow_session_macro_registration: bool = False

    # Request-scoped overrides installed by the canonical Skill runtime.
    skill_model_override: str | None = None
    skill_effort_override: str | int | None = None
    _skill_permission_base: ToolPermissionContext | None = None
    _skill_options_base: ToolUseOptions | None = None
    skill_scope_pending: bool = False
    skill_resource_roots: tuple[str, ...] = ()
    _skill_resource_roots_base: tuple[str, ...] | None = None
    skill_scope_active: bool = False
    active_skill_names: tuple[str, ...] = ()

    # Skill-declared hooks are merged with the frozen settings snapshot.
    skill_hooks: dict[str, list[Any]] = field(default_factory=dict)
    skill_hook_keys: set[str] = field(default_factory=set)

    # Chapter-12 / Phase 0 / WI-0.1 — frozen snapshot of hook config.
    # The snapshot is built once at startup by ``HookConfigManager.load()``
    # and updated only via explicit channels (the ``/hooks`` command or
    # an explicit ``reload_if_changed()`` call). Hook execution reads from
    # ``hook_config_manager.snapshot`` instead of ``options.hooks`` so a
    # malicious post-trust mutation of ``settings.json`` cannot affect
    # in-flight tool calls.
    #
    # ``options.hooks`` survives as a deprecated fallback for one release
    # cycle (see ``_get_hooks_from_snapshot`` in ``src/hooks/hook_executor.py``):
    # callers that still pass hooks via options get a ``DeprecationWarning``
    # but their behavior is preserved.
    hook_config_manager: Any | None = None
    # Chapter-12 / Phase 0 / WI-0.2 — workspace-trust gate. Bootstrap flips
    # this to ``True`` after the user accepts the trust dialog. Hooks (other
    # than ``HookSource.POLICY_SETTINGS``) are skipped while the workspace is
    # untrusted, mirroring TS' ``shouldSkipHookDueToTrust`` gate.
    workspace_trusted: bool = False

    # Chapter-9 / Fork Agents — captured bytes of the system prompt used on
    # the parent's most recent API call. Threaded into fork children so the
    # API request prefix is byte-identical across all parallel children
    # (chapter 9 §"The Byte-Identical Prefix Trick", Layer 1). Mirrors
    # ``toolUseContext.renderedSystemPrompt`` from
    # ``typescript/src/tools/AgentTool/AgentTool.tsx:496``.
    #
    # When ``None``, the fork path falls back to recomputing the parent
    # system prompt from ``options.custom_system_prompt`` or the active
    # agent definition. That fallback can diverge under feature-flag
    # transitions (GrowthBook cold→warm) and bust the prompt cache; the
    # main loop should populate this field whenever it has the rendered
    # bytes on hand for the parent's last turn.
    #
    # ch09 round-4 WI-1 — now POPULATED by query() at turn entry
    # (query.py, after the options.tools sync). Accepts the parent's actual
    # ``system_prompt`` shape: a ``list[dict]`` on the live agent-server /
    # headless path (build_effective_system_prompt) or a ``str`` on
    # string-prompt callers. The fork threads it verbatim into the child's
    # QueryParams.system_prompt so both sides run the identical
    # _call_model_sync assembly → byte-identical wire prefix.
    rendered_system_prompt: "str | list[dict[str, Any]] | None" = None

    plan_mode: bool = False

    @property
    def lkb_task_cutover_initialized(self) -> bool:
        """Expose LKB cutover state without leaking its private storage field."""

        return self._lkb_task_cutover_initialized

    @lkb_task_cutover_initialized.setter
    def lkb_task_cutover_initialized(self, value: bool) -> None:
        self._lkb_task_cutover_initialized = value

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()
        if self.default_permission_handler is None:
            self.default_permission_handler = self.permission_handler

    def mark_file_read(self, path: Path, *, partial: bool = False) -> None:
        stat = path.stat()
        self.read_file_fingerprints[path.resolve()] = (
            int(stat.st_mtime),
            int(stat.st_size),
            partial,
        )

    def was_file_read_and_unchanged(self, path: Path) -> bool:
        resolved = path.resolve()
        fingerprint = self.read_file_fingerprints.get(resolved)
        if fingerprint is None:
            return False
        mtime, size = fingerprint[0], fingerprint[1]
        stat = resolved.stat()
        return (mtime, size) == (int(stat.st_mtime), int(stat.st_size))

    def file_read_status(self, path: Path) -> str:
        """Return the read status of a file for write/edit staleness checks.

        Returns one of:
        - ``"not_read"`` -- no prior read recorded
        - ``"partial"`` -- file was read with offset/limit (partial view)
        - ``"modified"`` -- file changed on disk since last read
        - ``"ok"`` -- file was fully read and unchanged
        """
        resolved = path.resolve()
        fingerprint = self.read_file_fingerprints.get(resolved)
        if fingerprint is None:
            return "not_read"
        mtime, size = fingerprint[0], fingerprint[1]
        is_partial = fingerprint[2] if len(fingerprint) > 2 else False
        if is_partial:
            return "partial"
        stat = resolved.stat()
        if (mtime, size) != (int(stat.st_mtime), int(stat.st_size)):
            return "modified"
        return "ok"

    def allowed_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = [self.workspace_root]
        roots.extend(self.additional_working_directories)
        roots.extend(_resolve_path(path) for path in self.permission_context.additional_working_directories.keys())
        return tuple(roots)

    def ensure_allowed_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        else:
            p = p.resolve()
        # Mirror TS ``shouldBypassPermissions`` at
        # ``typescript/src/utils/permissions/permissions.ts:1268-1281``:
        # bypassPermissions mode (set by --dangerously-skip-permissions),
        # or plan mode when the user started with bypass available,
        # short-circuits the working-directory allowlist so the tool can
        # operate outside ``workspace_root``.
        mode = self.permission_context.mode
        if mode == "bypassPermissions" or (
            mode == "plan" and self.permission_context.is_bypass_permissions_mode_available
        ):
            return p
        roots = self.allowed_roots()
        if any(_is_within(p, root) for root in roots):
            return p
        # The current session's plan files (~/.clawcodex/plans/{slug}*.md)
        # are writable even though they sit outside the working roots — TS's
        # checkEditableInternalPath exempts them BEFORE the working-dir gate
        # (filesystem.ts:1259-1268), and the plan-mode attachment explicitly
        # directs the model to write there. Session-scoped slug prefix +
        # ``.md`` suffix keeps the carve-out narrow; failures fall through to
        # the normal refusal (fail closed).
        try:
            from src.utils.plans import is_session_plan_file

            if is_session_plan_file(str(p)):
                return p
        except Exception:  # noqa: BLE001  # nosec
            pass  # Plan-file detection failed; retain the normal deny-by-default path.
        roots_str = ", ".join(str(r) for r in roots)
        raise ToolPermissionError(f"path is outside allowed working directories: {p} (allowed: {roots_str})")

    def ensure_readable_path(self, path: str | Path) -> Path:
        """Like :meth:`ensure_allowed_path` but also permits harness-internal
        readable paths (tool-results / budget spill / scratchpad / memdir).

        The Read tool uses this so reading back the runtime's own spilled tool
        results — which sit outside ``workspace_root`` — does not raise. Mirrors
        TS keeping ``checkReadableInternalPath`` separate from the write/cwd
        allowlist; writes still go through :meth:`ensure_allowed_path`, so this
        never widens write or ``cd`` scope.
        """
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        else:
            p = p.resolve()
        mode = self.permission_context.mode
        if mode == "bypassPermissions" or (
            mode == "plan" and self.permission_context.is_bypass_permissions_mode_available
        ):
            return p
        roots = self.allowed_roots()
        if any(_is_within(p, root) for root in roots):
            return p
        # Harness-internal readable paths (spilled tool results, scratchpad,
        # memory) are readable even though they sit outside the working roots.
        from clawcodex_ext.permissions.filesystem import check_readable_internal_path

        if check_readable_internal_path(str(p), self):
            return p
        roots_str = ", ".join(str(r) for r in roots)
        raise ToolPermissionError(f"path is outside allowed working directories: {p} (allowed: {roots_str})")

    def ensure_tool_allowed(self, tool_name: str) -> None:
        if self.permission_context.blocks(tool_name):
            raise ToolPermissionError(f"tool is blocked by permission context: {tool_name}")

    def restore_retrieval_tools(self) -> None:
        """Clear the active retrieval plan and restore hidden tool objects.

        Session-macro provenance tools are never revived here — they belong
        only to the overlay and are synced via ``sync_effective_tools``.
        """

        current = list(self.options.tools or [])
        present = {getattr(tool, "name", "") for tool in current}
        for tool in self.retrieval_hidden_tools:
            # Do not revive foreign/stale session macros into options.tools.
            if getattr(tool, "_session_macro", False):
                continue
            name = str(getattr(tool, "name", "") or "")
            if name and name not in present:
                current.append(tool)
                present.add(name)
        self.options.tools = current
        self.retrieval_hidden_tools.clear()
        self.retrieval_suppressed_tools.clear()
        self.retrieval_plan = None
