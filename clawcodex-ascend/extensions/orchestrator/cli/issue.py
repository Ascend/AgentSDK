#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""orchestrator issue — manage individual issues handled by the orchestrator.

Usage (noun-verb, all using self-describing ``--id`` parameters):

  # Query
  clawcodex orchestrator issue list [--status <filter>]
  clawcodex orchestrator issue show --id <id>
  clawcodex orchestrator issue tail --id <id>

  # Lifecycle
  clawcodex orchestrator issue stop --id <id>
  clawcodex orchestrator issue pause --id <id> [--reason <text>]
  clawcodex orchestrator issue resume --id <id>
  clawcodex orchestrator issue takeover --id <id>

  # Operator interaction
  clawcodex orchestrator issue clarify --id <id> --answer <text> [--forward-to-author]
  clawcodex orchestrator issue inject --id <id> <hint> [--list] [--remove N]

  # Workspace
  clawcodex orchestrator issue workspace --id <id> [--ls] [--cat FILE] [--edit FILE --with CONTENT]

Design principles:
  - Self-describing parameters: use ``--id <id>`` instead of positional ``issue_id``
  - All commands are idempotent where possible
  - Stable behaviour: same args produce same outcome (or equivalent no-op)
"""

from __future__ import annotations

import argparse
import sys

from extensions.orchestrator.cli.resume_session import (  # noqa: E402
    _run_resume_session,
)
from .issue_control import (
    _run_list,
    _run_show,
    _run_tail,
)
from .issue_inject import _run_inject
from .issue_ops import (
    _run_clarify,
    _run_diff,
    _run_feedback,
    _run_pause,
    _run_resume,
    _run_review,
    _run_workspace,
)
from .issue_rebase import _run_init, _run_rebase, _run_retry
from .issue_transcript import _run_stop, _run_transcript

from extensions.orchestrator.cli.takeover import (  # noqa: E402,F401
    _run_takeover,
)


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_issue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``issue`` sub-subcommands."""
    issue_parser = subparsers.add_parser(
        "issue",
        help="Manage individual issues handled by the orchestrator",
        description="List, show, tail, stop, pause, resume, takeover, clarify, "
        "inject, or view workspace of issues managed by the orchestrator. "
        "All issue-level commands use --id for self-describing parameters "
        "and are designed to be idempotent.",
    )
    issue_sub = issue_parser.add_subparsers(
        dest="issue_subcommand",
        required=True,
    )

    # --- issue list ---
    list_parser = issue_sub.add_parser(
        "list",
        help="List all issues with their status",
        description="Display all issues known to the orchestrator, optionally "
        "filtered by status. Idempotent (pure read).",
    )
    list_parser.add_argument(
        "--status",
        choices=["pending", "running", "synced", "completed", "failed", "abandoned"],
        help="Filter by issue status",
    )
    list_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    list_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (resolution hint when metadata is missing)",
    )

    # --- issue show ---
    show_parser = issue_sub.add_parser(
        "show",
        help="Show details for a specific issue",
        description="Display issue metadata: status, branch, PR, token usage, "
        "and workspace path. Idempotent (pure read).",
    )
    show_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier (e.g. 42 or owner/repo#42)",
    )
    show_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path",
    )
    show_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (resolution hint when metadata is missing)",
    )

    # --- issue tail ---
    tail_parser = issue_sub.add_parser(
        "tail",
        help="Tail tool call logs for a running issue in real-time",
        description="Stream tool call events from a running issue's event log. "
        "Idempotent (pure read, non-destructive).",
    )
    tail_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to tail",
    )
    tail_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path",
    )
    tail_parser.add_argument(
        "--turn",
        type=int,
        default=None,
        metavar="N",
        help="Filter to show only events from turn number N",
    )

    # --- issue transcript ---
    transcript_parser = issue_sub.add_parser(
        "transcript",
        help="Print a session transcript for an issue or run",
        description="Read the full session transcript from "
        "~/.clawcodex/sessions/{run_id}/transcript.jsonl and "
        "print it as text. Idempotent (pure read, suitable for "
        "piping).",
    )
    transcript_parser.add_argument(
        "--id",
        type=str,
        default=None,
        metavar="ISSUE_ID",
        help="Issue identifier (resolves to run_id via the registry)",
    )
    transcript_parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Run identifier (skips registry resolution)",
    )
    transcript_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path",
    )
    transcript_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (resolution hint when metadata is missing)",
    )
    transcript_parser.add_argument(
        "--role",
        choices=["user", "assistant"],
        default=None,
        help="Filter to show only messages with this role",
    )
    transcript_parser.add_argument(
        "--tool-use-id",
        dest="tool_use_id",
        type=str,
        default=None,
        metavar="TOOL_USE_ID",
        help="Filter to show only tool_use / tool_result blocks with this id",
    )
    transcript_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N messages",
    )

    # --- issue stop ---
    stop_parser = issue_sub.add_parser(
        "stop",
        help="Force-terminate a running agent for an issue",
        description="Write a stop control command for the orchestrator to pick up "
        "on its next poll cycle. The agent will be marked as failed. "
        "Idempotent: stopping an already-stopped issue succeeds silently.",
    )
    stop_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to stop",
    )
    stop_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )
    stop_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Send stop and return immediately without waiting for agent to terminate",
    )

    # --- issue pause ---
    pause_parser = issue_sub.add_parser(
        "pause",
        help="Pause a running agent at the next tool call boundary",
        description="Write a pause control command. The agent will complete its "
        "current tool call then pause (no new tool calls until resume). "
        "Idempotent: pausing an already-paused issue succeeds silently.",
    )
    pause_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to pause",
    )
    pause_parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Reason for pausing (visible to the agent)",
    )
    pause_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Send pause and return immediately without waiting for confirmation",
    )

    # --- issue resume ---
    resume_parser = issue_sub.add_parser(
        "resume",
        help="Resume a paused agent",
        description="Write a resume control command to allow the agent to continue. "
        "Idempotent: resuming a running (non-paused) issue succeeds silently.",
    )
    resume_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to resume",
    )
    resume_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Send resume and return immediately without waiting for confirmation",
    )

    # --- issue resume-session ---
    # Load the JSONL transcript written by the
    # headless agent and rehydrate the LLM context (the
    # orchestrator-side counterpart of `clawcodex --resume <run_id>`).
    # This does NOT touch the control socket; the agent is unaffected.
    resume_session_parser = issue_sub.add_parser(
        "resume-session",
        help="Rehydrate an orchestrator session's LLM context from disk",
        description=(
            "Look up the run_id for an issue in the IssueRegistry, "
            "call Session.resume(run_id) to update bootstrap state, "
            "and read the JSONL transcript written by the headless "
            "agent. Prints a short summary of the rehydrated "
            "Conversation. Use `issue takeover --id X` to take over a "
            "live run, or start a fresh REPL against the same "
            "workspace to continue the conversation."
        ),
    )
    resume_session_parser.add_argument(
        "--id",
        type=str,
        default=None,
        metavar="ISSUE_ID",
        help="Issue identifier or ID (preferred)",
    )
    resume_session_parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Specific run_id (overrides the registry)",
    )

    # --- issue takeover ---
    # Read-only snapshot viewer: spawns a --resume REPL against the
    # agent's run_id so the operator can inspect the current
    # conversation history. The agent is NOT paused — it keeps
    # running unaffected. ``--id`` is preferred; ``--run`` +
    # ``--workspace`` is a fallback when the registry is unavailable.
    takeover_parser = issue_sub.add_parser(
        "takeover",
        help="Take a read-only snapshot of an issue's conversation history",
        description=(
            "Start an interactive clawcodex REPL with "
            "--resume <run_id> in the issue's workspace to inspect "
            "the agent's current conversation history. The agent is "
            "NOT paused — it continues running unaffected. When the "
            "REPL exits, the orchestrator proceeds normally. "
            "Idempotent: if the agent has already ended, the REPL "
            "loads the on-disk transcript directly."
        ),
    )
    takeover_parser.add_argument(
        "--id",
        type=str,
        default=None,
        metavar="ISSUE_ID",
        help="Issue identifier or ID (preferred)",
    )
    takeover_parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Specific run_id (overrides the registry)",
    )
    takeover_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="WORKSPACE",
        help=("Workspace path (overrides the registry; required for --run)"),
    )

    # --- issue clarify ---
    clarify_parser = issue_sub.add_parser(
        "clarify",
        help="Answer a clarification request from the orchestrator",
        description="Record an operator answer for a pending clarification. "
        "The orchestrator picks up the answer on its next poll cycle. "
        "Idempotent: answering an already-answered clarification "
        "updates the answer in place.",
    )
    clarify_parser.add_argument(
        "--id",
        type=str,
        required=False,
        metavar="ISSUE_ID",
        help="Issue ID being clarified",
    )
    clarify_parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help="Operator's answer to the clarification question",
    )
    clarify_parser.add_argument(
        "--forward-to-author",
        action="store_true",
        help="Skip local answer, forward directly to author (@mention)",
    )
    clarify_action = clarify_parser.add_mutually_exclusive_group()
    clarify_action.add_argument(
        "--list",
        dest="list_clarifications",
        action="store_true",
        help="List current clarification records",
    )
    clarify_action.add_argument(
        "--recheck",
        action="store_true",
        help="Clear the cached clarity decision so the daemon analyzes the issue again",
    )
    clarify_action.add_argument(
        "--resolve",
        action="store_true",
        help="Manually mark the clarity gate resolved and allow dispatch",
    )
    clarify_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit orchestrator workspace root",
    )
    clarify_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (workspace discovery hint)",
    )

    # --- issue inject ---
    inject_parser = issue_sub.add_parser(
        "inject",
        help="Inject operator hints into a running agent",
        description=(
            "Send a hint to the agent. When the agent's control "
            "socket is alive, the hint is queued via pending_messages for "
            "delivery at the next tool result boundary (near-real-time). "
            "Otherwise, the hint is written to .operator_hints.md and the "
            "agent reads it at the next turn boundary. "
            "Idempotent: re-injecting the same hint is a no-op.\n\n"
            "Tips: Be concise and directive — the hint is added to the "
            "LLM's context as an operator instruction. Good examples: "
            "'Run pytest before committing', 'Check the error handling "
            "in src/api.py', 'The bug is in the date parsing logic'. "
            "The agent will see the hint in its next response but may "
            "choose how to act on it."
        ),
    )
    inject_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to inject hint for",
    )
    inject_parser.add_argument(
        "hint",
        nargs="?",
        default=None,
        help="Hint text to inject (omit to just list existing hints)",
    )
    inject_parser.add_argument(
        "--list",
        dest="list_hints",
        action="store_true",
        help="List existing hints for this issue",
    )
    inject_parser.add_argument(
        "--remove",
        dest="remove_hint",
        type=int,
        metavar="N",
        help="Remove hint number N",
    )
    inject_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Send inject and return immediately without waiting for delivery confirmation",
    )

    # --- issue workspace ---
    ws_parser = issue_sub.add_parser(
        "workspace",
        help="View and modify files in an issue's workspace",
        description="List, view, or edit files in an issue's workspace directory. "
        "Use with caution — concurrent edits may conflict with agent changes. "
        "Idempotent: listing and viewing are pure reads; editing overwrites.",
    )
    ws_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier whose workspace to operate on",
    )
    ws_parser.add_argument(
        "--ls",
        action="store_true",
        help="List files in the workspace",
    )
    ws_parser.add_argument(
        "--cat",
        metavar="FILE",
        help="Show contents of a file in the workspace",
    )
    ws_parser.add_argument(
        "--edit",
        metavar="FILE",
        help="Edit a file (requires --with)",
    )
    ws_parser.add_argument(
        "--with",
        dest="content",
        metavar="CONTENT",
        help="New file content (for use with --edit)",
    )

    # --- issue review ---
    review_parser = issue_sub.add_parser(
        "review",
        help="Approve or reject a completed issue's changes (LocalTracker)",
        description="Review a LocalTracker issue after agent completes git commit. "
        "Approve to mark as completed, or reject to inject feedback and retry.",
    )
    review_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to review",
    )
    review_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the changes — mark issue as completed",
    )
    review_parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject the changes — inject feedback and retry",
    )
    review_parser.add_argument(
        "--feedback",
        type=str,
        default=None,
        metavar="TEXT",
        help="Feedback for rejection (required with --reject)",
    )
    review_parser.add_argument(
        "--comment",
        type=str,
        default=None,
        metavar="TEXT",
        help="Optional comment for approval",
    )

    # --- issue feedback ---
    feedback_parser = issue_sub.add_parser(
        "feedback",
        help="List, approve, or dismiss pending PR review feedback",
        description="Manage pending PR review feedback items. Use --list to show pending items, "
        "--approve to trigger follow-up for pending feedback, or --dismiss to remove "
        "feedback without processing.",
    )
    feedback_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier with pending feedback",
    )
    feedback_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_feedback",
        help="List all pending feedback items for the issue",
    )
    feedback_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve pending feedback and trigger follow-up agent run",
    )
    feedback_parser.add_argument(
        "--dismiss",
        action="store_true",
        help="Dismiss pending feedback without triggering follow-up",
    )
    feedback_parser.add_argument(
        "--feedback-id",
        type=str,
        nargs="*",
        metavar="FEEDBACK_ID",
        help="Specific feedback item IDs to approve/dismiss (all pending if omitted)",
    )
    feedback_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path",
    )
    feedback_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md",
    )

    # --- issue diff ---
    diff_parser = issue_sub.add_parser(
        "diff",
        help="Show code changes for a completed or pending_review issue",
        description="Display a summary or full diff of changes made by the agent. "
        "Shows stats by default, use --full for complete diff output.",
    )
    diff_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to show diff for",
    )
    diff_parser.add_argument(
        "--full",
        action="store_true",
        help="Show complete diff output (not just summary stats)",
    )
    diff_parser.add_argument(
        "--stat",
        action="store_true",
        help="Show only file change statistics (default when no --full)",
    )
    diff_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )

    # --- issue retry (CLI fallback command) ---
    retry_parser = issue_sub.add_parser(
        "retry",
        help="Retry/follow-up/unblock an issue via the CLI fallback",
        description="Operator-driven fallback for retry / follow-up / unblock intents when label / "
        "comment paths are inconvenient. Records the action in "
        "~/.clawcodex/orchestrator/audit.jsonl and updates the "
        "local issue registry so the next daemon poll picks up "
        "the new intent.",
    )
    retry_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier to retry / follow-up / unblock",
    )
    retry_parser.add_argument(
        "--mode",
        type=str,
        choices=["reset", "followup", "unblock"],
        required=True,
        help="Intent mode: 'reset' clears state and re-runs (agent:retry), "
        "'followup' appends a commit to the existing branch "
        "(agent:follow-up), 'unblock' rolls an abandoned issue back "
        "to pending so the daemon reconsiders it.",
    )
    retry_parser.add_argument(
        "--reason",
        type=str,
        default="",
        metavar="TEXT",
        help="Free-form reason recorded in audit.jsonl",
    )
    retry_parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the max_retries_per_issue rate limit (CLI-only override; logged as a high-priority audit entry).",
    )
    retry_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Operator override for max_retries_per_issue (default: 3). "
        "Has no effect unless --force is also set; the audit "
        "log records both the configured limit and the actual "
        "retry_count when --force triggers a bypass.",
    )
    retry_parser.add_argument(
        "--operator",
        type=str,
        default=None,
        metavar="LOGIN",
        help="Operator login recorded in audit.jsonl (defaults to $USER / os.getlogin())",
    )
    retry_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    retry_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (resolution hint when metadata is missing)",
    )
    retry_parser.add_argument(
        "--stop-first",
        dest="stop_first",
        action="store_true",
        default=False,
        help="If the agent is still running, stop it first before retrying. "
        "Equivalent to 'issue stop' followed by 'issue retry'.",
    )

    # --- issue init ---
    init_parser = issue_sub.add_parser(
        "init",
        help="Scaffold an issue card from the issue-card.template.md",
        description="Copy the packaged issue-card.template.md to the specified "
        "output path and optionally replace <...> placeholders. "
        "Useful for local-tracker workflows where issues are *.md files.",
    )
    init_parser.add_argument(
        "--id",
        default="",
        metavar="ID",
        help="Issue ID (e.g. <ID>-pr-auto-fix)",
    )
    init_parser.add_argument(
        "--identifier",
        default="",
        metavar="IDENTIFIER",
        help="Short identifier (e.g. <id>)",
    )
    init_parser.add_argument(
        "--title",
        default="",
        metavar="TITLE",
        help="Issue title",
    )
    init_parser.add_argument(
        "--priority",
        default="",
        metavar="PRIORITY",
        help="Priority 0-3",
    )
    init_parser.add_argument(
        "--state",
        default="open",
        metavar="STATE",
        help="Initial state (default: open)",
    )
    init_parser.add_argument(
        "--category",
        default="",
        metavar="TAG",
        help="Category label (e.g. review-auto-fix, docs, refactor)",
    )
    init_parser.add_argument(
        "--branch-name",
        default="",
        metavar="NAME",
        help="Preferred branch name (leave blank for auto-generation)",
    )
    init_parser.add_argument(
        "--base-branch",
        default="",
        metavar="BRANCH",
        help="Base branch (e.g. dev-decoupling, main)",
    )
    init_parser.add_argument(
        "--assignee",
        default="",
        metavar="USER",
        help="Assignee / team for tracking",
    )
    init_parser.add_argument(
        "--url",
        default="",
        metavar="URL",
        help="Upstream issue / document URL",
    )
    init_parser.add_argument(
        "--output",
        "--out",
        default="./issue.md",
        metavar="FILE",
        help="Output file path (default: ./issue.md)",
    )
    init_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; use defaults for missing values",
    )

    # --- issue rebase (CLI fallback command) ---
    rebase_parser = issue_sub.add_parser(
        "rebase",
        help="Rebase the PR's feature branch onto the latest base (CLI fallback)",
        description="Operator-driven fallback for PR conflict resolution. "
        "Writes a control file that the daemon picks up on its next "
        "poll cycle. The orchestrator itself performs the rebase "
        "(no external agent for clean rebases); the agent is only "
        "invoked if the rebase leaves actual content conflicts.",
    )
    rebase_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="ISSUE_ID",
        help="Issue identifier whose PR should be rebased",
    )
    rebase_parser.add_argument(
        "--force",
        action="store_true",
        help="Use plain `git push --force` (default: --force-with-lease). "
        "Bypasses the max_rebase_attempts_per_issue rate limit. "
        "Logged as a high-priority audit entry.",
    )
    rebase_parser.add_argument(
        "--reason",
        type=str,
        default="",
        metavar="TEXT",
        help="Free-form reason recorded in audit.jsonl",
    )
    rebase_parser.add_argument(
        "--operator",
        type=str,
        default=None,
        metavar="LOGIN",
        help="Operator login recorded in audit.jsonl",
    )
    rebase_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (auto-detection override)",
    )
    rebase_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (resolution hint when metadata is missing)",
    )


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate issue subcommand."""
    cmd = args.issue_subcommand

    # Resolve workspace/registry helpers
    from extensions.orchestrator.workspace_locator import (
        get_registry_path,
        get_workspace_root,
    )

    ws = get_workspace_root(
        workspace_arg=getattr(args, "workspace", None),
        workflow_path=getattr(args, "workflow", None),
    )
    registry_path = get_registry_path(
        workspace_arg=getattr(args, "workspace", None),
        workflow_path=getattr(args, "workflow", None),
    )

    if cmd == "list":
        return _run_list(registry_path, args)
    elif cmd == "show":
        return _run_show(registry_path, args)
    elif cmd == "tail":
        return _run_tail(registry_path, args)
    elif cmd == "transcript":
        return _run_transcript(registry_path, args)
    elif cmd == "stop":
        return _run_stop(args, registry_path=registry_path, workspace_root=ws)
    elif cmd == "pause":
        return _run_pause(args, workspace_root=ws)
    elif cmd == "resume":
        return _run_resume(args, workspace_root=ws)
    elif cmd == "resume-session":
        return _run_resume_session(registry_path, args)
    elif cmd == "takeover":
        return _run_takeover(registry_path, ws, args)
    elif cmd == "clarify":
        return _run_clarify(args, registry_path=registry_path, workspace_root=ws)
    elif cmd == "inject":
        return _run_inject(args)
    elif cmd == "workspace":
        return _run_workspace(args)
    elif cmd == "review":
        return _run_review(registry_path, args, workspace_root=ws)
    elif cmd == "diff":
        return _run_diff(registry_path, args)
    elif cmd == "retry":
        return _run_retry(registry_path, args, workspace_root=ws)
    elif cmd == "rebase":
        return _run_rebase(registry_path, args, workspace_root=ws)
    elif cmd == "feedback":
        return _run_feedback(registry_path, args, workspace_root=ws)
    elif cmd == "init":
        return _run_init(args)

    print(f"error: unknown issue subcommand '{cmd}'", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
