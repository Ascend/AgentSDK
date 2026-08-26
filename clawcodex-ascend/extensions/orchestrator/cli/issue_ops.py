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

# pylint: disable=too-many-nested-blocks,inconsistent-return-statements

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .issue_control import _resolve_issue_workspace_path, _resolve_sock_path, _send_and_wait, _write_control


def _run_pause(args: argparse.Namespace, workspace_root: str | Path | None = None) -> int:
    """Pause a running issue agent. Idempotent — already-paused → success."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2
    reason = getattr(args, "reason", "") or "operator requested pause"
    no_wait = getattr(args, "no_wait", False)

    sock_path = _resolve_sock_path(issue_id, workspace_root)
    if sock_path is not None and not no_wait:

        async def _do_pause() -> int:
            t0 = asyncio.get_event_loop().time()
            data = await _send_and_wait(sock_path, "pause", reason, "Paused", timeout=30.0)
            elapsed = asyncio.get_event_loop().time() - t0
            if data is not None:
                turn = data.get("turn", "?")
                tool = data.get("tool_name", "?")
                print(f"Agent paused at turn {turn}, tool {tool!r} ({elapsed:.1f}s).")
                return 0
            else:
                print(
                    "Pause acknowledged but agent is in a long operation "
                    "(30s timeout). It will pause at the next tool boundary."
                )
                return 0

        return asyncio.run(_do_pause())
    elif sock_path is not None and no_wait:
        # Fire and forget via socket.
        return _write_control("pause", issue_id, reason, workspace_root=workspace_root)
    else:
        print(f"Issue pause: sending pause command for {issue_id}")
        return _write_control("pause", issue_id, reason, workspace_root=workspace_root)


# ---------------------------------------------------------------------------
# issue resume
# ---------------------------------------------------------------------------


def _run_resume(args: argparse.Namespace, workspace_root: str | Path | None = None) -> int:
    """Resume a paused issue agent. Idempotent — running → success."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2
    no_wait = getattr(args, "no_wait", False)

    sock_path = _resolve_sock_path(issue_id, workspace_root)
    if sock_path is not None and not no_wait:

        async def _do_resume() -> int:
            t0 = asyncio.get_event_loop().time()
            data = await _send_and_wait(sock_path, "resume", "", "Resumed", timeout=5.0)
            elapsed = asyncio.get_event_loop().time() - t0
            if data is not None:
                print(f"Agent resumed ({elapsed:.1f}s).")
                return 0
            else:
                print("Resume sent but no confirmation (5s). The agent may already be running.")
                return 0

        return asyncio.run(_do_resume())
    elif sock_path is not None and no_wait:
        return _write_control("resume", issue_id, workspace_root=workspace_root)
    else:
        print(f"Issue resume: sending resume command for {issue_id}")
        return _write_control("resume", issue_id, workspace_root=workspace_root)


# ---------------------------------------------------------------------------
# issue clarify
# ---------------------------------------------------------------------------


def _run_clarify(
    args: argparse.Namespace,
    *,
    registry_path: Path | None = None,
    workspace_root: Path | None = None,
) -> int:
    """Answer a clarification request. Idempotent — re-answering updates in place."""
    issue_id = getattr(args, "id", None)
    list_clarifications = bool(getattr(args, "list_clarifications", False))
    if not issue_id and not list_clarifications:
        print("error: --id is required", file=sys.stderr)
        return 2

    answer = getattr(args, "answer", None)
    forward = getattr(args, "forward_to_author", False)

    recheck = bool(getattr(args, "recheck", False))
    resolve = bool(getattr(args, "resolve", False))
    if not answer and not forward and not list_clarifications and not recheck and not resolve:
        print("error: --answer is required unless --forward-to-author is used", file=sys.stderr)
        return 2

    from extensions.orchestrator.clarification_queue import ClarificationQueue
    from extensions.orchestrator.issue_registry import IssueRegistry

    queue_path = Path(workspace_root) / ".clawcodex_clarification_queue.json" if workspace_root is not None else None
    queue = ClarificationQueue(queue_path)

    if list_clarifications:
        items = queue.list_items()
        if not items:
            print("No clarification records.")
            return 0
        for item in items:
            print(f"{item.issue_id}\t{item.status.value}\t{item.question}")
        return 0

    registry = IssueRegistry(registry_path) if registry_path is not None else None
    if recheck:
        queue.remove(issue_id)
        record = registry.get(issue_id) if registry is not None else None
        if record is None:
            print(f"Issue {issue_id} is not present in the registry.", file=sys.stderr)
            return 1
        record.clarification_status = None
        record.open_questions = []
        record.clarification_round = 0
        record.clarifier_fingerprint = None
        record.clarification_replies = []
        record.local_answer = None
        record.local_answer_source = None
        record.touch()
        registry._save()
        print(f"Issue {issue_id} will be rechecked on the next poll cycle.")
        return 0

    if resolve:
        queue.remove(issue_id)
        if registry is None:
            print("Could not locate the issue registry.", file=sys.stderr)
            return 1
        record = registry.get(issue_id)
        if record is None:
            print(f"Issue {issue_id} is not present in the registry.", file=sys.stderr)
            return 1
        registry.mark_clarification_resolved(
            issue_id,
            fingerprint=record.clarifier_fingerprint or "manual",
            answer=answer or "Manually resolved by operator",
            source="operator",
            status="manual_resolved",
        )
        print(f"Issue {issue_id} clarification marked resolved.")
        return 0

    if forward:
        item = queue.mark_awaiting_author(issue_id)
        if item is None:
            print(f"No pending clarification for issue {issue_id}.", file=sys.stderr)
            return 1
        print(f"Issue {issue_id} marked for author clarification.")
        return 0

    resolved = queue.resolve(issue_id, answer or "", source="clarification_queue")
    if resolved is None:
        print(f"Failed to write answer for issue {issue_id}.", file=sys.stderr)
        return 1

    print(f"Answer recorded for issue {issue_id}: {answer or '(forwarded to author)'}")
    print(f"Status: {resolved.status.value}")
    print("The orchestrator will pick this up on its next poll cycle.")
    return 0


# ---------------------------------------------------------------------------
# issue inject
# ---------------------------------------------------------------------------


def _run_workspace(args: argparse.Namespace) -> int:
    """View or modify workspace files. Workspace listing/view are pure reads."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    ws_path = _resolve_issue_workspace_path(issue_id)
    if ws_path is None:
        print(f"Could not find workspace for issue {issue_id}.", file=sys.stderr)
        return 1

    ls_flag = getattr(args, "ls", False)
    cat_flag = getattr(args, "cat", None)
    edit_flag = getattr(args, "edit", None)
    content = getattr(args, "content", None)

    if ls_flag:
        return _workspace_list_files(issue_id, ws_path)
    elif cat_flag:
        return _workspace_cat_file(issue_id, ws_path, cat_flag)
    elif edit_flag:
        if not content:
            print("error: --edit requires --with <content>", file=sys.stderr)
            return 2
        return _workspace_edit_file(issue_id, ws_path, edit_flag, content)
    else:
        return _workspace_list_files(issue_id, ws_path)


def _workspace_list_files(issue_id: str, ws_path: Path) -> int:
    """List files in workspace. Idempotent — pure read."""
    if not ws_path.exists():
        print(f"Workspace for issue {issue_id} not found.", file=sys.stderr)
        return 1

    exclude = {".metadata", ".orchestrator_control", ".operator_hints.md"}
    print(f"Workspace for issue {issue_id}: {ws_path}")
    print("-" * 60)

    files: list[str] = []
    dirs: list[str] = []
    for item in sorted(ws_path.iterdir()):
        if item.name in exclude:
            continue
        if item.is_dir():
            dirs.append(item.name + "/")
        else:
            size = item.stat().st_size
            files.append(f"{item.name} ({size} bytes)")

    for d in dirs:
        print(f"  [DIR]  {d}")
    for f in files:
        print(f"  {f}")
    if not files and not dirs:
        print("  (empty workspace)")
    return 0


def _workspace_cat_file(issue_id: str, ws_path: Path, filename: str) -> int:
    """Show file contents. Idempotent — pure read."""
    file_path = ws_path / filename
    if not file_path.exists():
        print(f"File not found: {filename}", file=sys.stderr)
        return 1
    if not file_path.is_file():
        print(f"Not a file: {filename}", file=sys.stderr)
        return 1
    try:
        content = file_path.read_text(encoding="utf-8")
        print(f"=== {filename} ===")
        print(content)
    except Exception as exc:
        print(f"Failed to read {filename}: {exc}", file=sys.stderr)
        return 1
    return 0


def _workspace_edit_file(issue_id: str, ws_path: Path, filename: str, content: str) -> int:
    """Write new content to a file."""
    file_path = ws_path / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_path.write_text(content, encoding="utf-8")
        print(f"Updated {filename} in issue {issue_id} workspace.")
        print("  The agent will see this change on its next tool call.")
        return 0
    except Exception as exc:
        print(f"Failed to write {filename}: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# issue review
# ---------------------------------------------------------------------------


def _tracker_from_workflow_arg(args: argparse.Namespace) -> Any | None:
    workflow_path = getattr(args, "workflow", None)
    if not workflow_path:
        return None
    try:
        from extensions.orchestrator.tracker import create_tracker_adapter
        from extensions.orchestrator.workflow import WorkflowLoader

        workflow, _ = WorkflowLoader.load(workflow_path)
        return create_tracker_adapter(workflow.tracker)
    except Exception as exc:
        print(f"Warning: could not initialize tracker from workflow: {exc}", file=sys.stderr)
        return None


def _mirror_intent_label(
    tracker: Any | None,
    issue_id: str,
    label: str,
    *,
    remove: bool,
) -> bool:
    """Best-effort mirror of CLI intent onto issue label.

    Calls ``tracker.add_label(issue_id, label)`` (default) or
    ``tracker.remove_label(issue_id, label)`` (when ``remove=True``)
    so the label-based intent path picks up the same intent as the
    local ``registry.intent``.

    The local ``registry.intent`` is the authoritative source of
    truth; this is belt-and-suspenders so a future registry reset
    does not silently drop the operator's intent. The function
    is intentionally permissive:

      * ``tracker is None`` → returns False (no-op).
      * Tracker does not implement the label method → returns False.
      * Async call raises or returns False → logs a warning and
        returns False. Never raises.

    Used by :func:`_run_retry` for ``--mode reset`` (add
    ``agent:retry``), ``--mode followup`` (add ``agent:follow-up``),
    and ``--mode unblock`` (remove ``agent:blocked``).
    """
    if tracker is None:
        return False
    from extensions.orchestrator.tracker import LabelCapability, supports

    if not supports(tracker, LabelCapability):
        return False
    method = tracker.remove_label if remove else tracker.add_label
    if method is None:
        return False
    try:

        async def call() -> bool:
            return bool(await method(issue_id, label))

        return asyncio.run(call())
    except Exception as exc:  # noqa: BLE001
        verb = "remove" if remove else "add"
        print(
            f"Warning: could not {verb} {label} label on issue {issue_id}: {exc}",
            file=sys.stderr,
        )
        return False


def _run_review(registry_path: Path | None, args: argparse.Namespace, workspace_root: str | Path | None = None) -> int:
    """Approve or reject a LocalTracker issue's changes."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    if not registry_path or not registry_path.exists():
        print(f"No registry found. Cannot review issue {issue_id}.", file=sys.stderr)
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry, IssueStatus

    registry = IssueRegistry(registry_path)
    record = registry.get(issue_id)
    if record is None:
        print(f"Issue {issue_id} not found in registry.", file=sys.stderr)
        return 1

    approve = getattr(args, "approve", False)
    reject = getattr(args, "reject", False)

    if not approve and not reject:
        print("error: specify --approve or --reject", file=sys.stderr)
        return 2

    recoverable_failed_completion = bool(
        record.status is IssueStatus.COMPLETED
        and (
            record.verification_status == "failed"
            or record.last_hook_error
            or record.session_end_reason == "empty_branch_no_commits"
        )
    )
    retry_already_queued = bool(
        reject
        and (
            record.status
            in {
                IssueStatus.PENDING,
                IssueStatus.FAILED,
                IssueStatus.VERIFICATION_FAILED,
            }
            or recoverable_failed_completion
        )
        and (
            getattr(record.intent, "value", record.intent) in {"retry", "followup"}
            or record.commit_sha
            or record.pr_number
            or record.pr_url
            or recoverable_failed_completion
        )
    )
    approve_already_recorded = approve and record.status is IssueStatus.COMPLETED
    if record.status is not IssueStatus.PENDING_REVIEW and not retry_already_queued and not approve_already_recorded:
        print(
            f"Issue {issue_id} is not pending review (status: {record.status.value}).",
            file=sys.stderr,
        )
        print(
            "Only issues with 'pending_review' status, or an already queued rejection retry, can be reviewed.",
            file=sys.stderr,
        )
        return 1

    if reject:
        feedback = getattr(args, "feedback", None)
        if not feedback:
            print("error: --reject requires --feedback", file=sys.stderr)
            return 2

        # The daemon owns its in-memory registry, lifecycle sets, tracker state,
        # and clarification queue. Send one durable control command so those
        # related mutations happen together on the next poll instead of
        # partially updating the same files through stale CLI-side objects.
        rc = _write_control("review_retry", issue_id, feedback, workspace_root=workspace_root)
        if rc != 0:
            return rc

        print(f"Issue {issue_id} rejected with feedback:")
        print(f'  "{feedback}"')
        print("Feedback queued — orchestrator will retry this issue.")
        return 0

    if approve:
        comment = getattr(args, "comment", None)
        rc = _write_control(
            "review_approve",
            issue_id,
            comment or "",
            workspace_root=workspace_root,
        )
        if rc != 0:
            return rc

        # The daemon is the sole owner of the registry, lifecycle sets and
        # remote tracker side effects.  Updating them here as well races its
        # in-memory snapshot and posts the optional approval comment twice.
        print(f"Issue {issue_id} approval queued — orchestrator will finalize it.")
        return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
# issue feedback
# ---------------------------------------------------------------------------

# Matches a repo web URL like
#   https://gitcode.com/Gideon_Zhao/perf-reference-ascend/merge_requests/3
#   https://gitee.com/acme/widget/pulls/9
#   https://github.com/acme/widget/pull/12
# capturing host / owner / repo so we can rebuild a comment permalink.
_PR_URL_RE = re.compile(r"^(?P<host>https?://[^/]+)/(?P<owner>[^/]+)/(?P<repo>[^/]+)/")


def _fallback_feedback_url(record: Any, feedback_id: str) -> str | None:
    """Reconstruct a comment URL when none was persisted.

    Used by ``issue feedback --list`` for records written before URL
    persistence, or items whose source has no html_url (GitCode's
    issue-comments endpoint omits it). Parses host/owner/repo from the
    record's ``pr_url`` and builds the platform's comment permalink:

      - gitcode / gitee: ``{host}/{owner}/{repo}/issues/{number}#tid-{id}``
      - github:          ``{host}/{owner}/{repo}/issues/{number}#issuecomment-{id}``

    Returns ``None`` for review_summary / ci sources (no comment anchor)
    or when the record has no parseable pr_url.
    """
    if not feedback_id or ":" not in feedback_id:
        return None
    source, _, raw_id = feedback_id.partition(":")
    if source not in {"conversation", "inline_review"} or not raw_id:
        return None
    pr_url = getattr(record, "pr_url", None)
    if not isinstance(pr_url, str) or not pr_url:
        return None
    m = _PR_URL_RE.match(pr_url)
    if not m:
        return None
    host = m.group("host")
    owner = m.group("owner")
    repo = m.group("repo")
    # Issue/PR number for the URL path. The tracker fetches conversation
    # comments via ``/issues/{effective_issue_id}/comments`` where
    # ``effective_issue_id = issue_id or pr_number`` (see
    # client.fetch_pull_request_feedback), so the comment lives under the
    # issue number. Prefer the record's issue_id when it is numeric
    # (GitCode stores the bare issue number there); fall back to pr_number
    # for GitHub/Gitee where issue_id may be a tracker key (AGENTSDK-15).
    number = ""
    raw_issue_id = str(getattr(record, "issue_id", "") or "").strip()
    if raw_issue_id.startswith("#"):
        raw_issue_id = raw_issue_id[1:]
    if raw_issue_id.isdigit():
        number = raw_issue_id
    else:
        pr_number = getattr(record, "pr_number", None)
        if isinstance(pr_number, str) and pr_number.strip().isdigit():
            number = pr_number.strip()
    if not number:
        return None
    anchor = f"#issuecomment-{raw_id}" if "github.com" in host else f"#tid-{raw_id}"
    return f"{host}/{owner}/{repo}/issues/{number}{anchor}"


def _run_feedback(
    registry_path: Path | None, args: argparse.Namespace, workspace_root: str | Path | None = None
) -> int:
    """List, approve, or dismiss pending PR review feedback."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    if not registry_path or not registry_path.exists():
        print(f"No registry found. Cannot manage feedback for issue {issue_id}.", file=sys.stderr)
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    record = registry.get(issue_id)
    if record is None:
        print(f"Issue {issue_id} not found in registry.", file=sys.stderr)
        return 1

    list_feedback = getattr(args, "list_feedback", False)
    approve = getattr(args, "approve", False)
    dismiss = getattr(args, "dismiss", False)

    if not list_feedback and not approve and not dismiss:
        print("error: specify --list, --approve, or --dismiss", file=sys.stderr)
        return 2

    if list_feedback:
        if not record.pending_feedback_ids:
            print(f"No pending feedback for issue {issue_id}.")
            return 0
        print(f"Pending feedback for issue {issue_id}:")
        print("(use the ID with --feedback-id to approve/dismiss a single item)")
        for i, fid in enumerate(record.pending_feedback_ids, 1):
            # Resolve the canonical comment/check URL when available
            # (persisted from the tracker's html_url). Fall back to
            # reconstructing it from pr_url + raw comment id (GitCode's
            # issue-comments API omits html_url). No URL for review_summary
            # / ci sources -> show the id alone.
            url = record.pending_feedback_urls.get(fid) or _fallback_feedback_url(record, fid)
            if url:
                print(f"  {i}. {fid}  ->  {url}")
            else:
                print(f"  {i}. {fid}")
        print(f"\nTotal: {len(record.pending_feedback_ids)} pending item(s)")
        return 0

    target_ids = getattr(args, "feedback_id", None) or list(record.pending_feedback_ids)
    if not target_ids:
        print(f"No pending feedback to process for issue {issue_id}.")
        return 0

    if dismiss:
        registry.mark_feedback_processed(issue_id, target_ids)
        print(f"Dismissed {len(target_ids)} feedback item(s) for issue {issue_id}.")
        return 0

    if approve:
        _write_control("review_followup", issue_id, ",".join(target_ids), workspace_root=workspace_root)
        print(f"Approved {len(target_ids)} feedback item(s) for issue {issue_id}.")
        print("Follow-up will be triggered on next orchestrator poll cycle.")
        return 0

    return 0


# ---------------------------------------------------------------------------
# issue diff
# ---------------------------------------------------------------------------


def _run_diff(registry_path: Path | None, args: argparse.Namespace) -> int:
    """Show code changes for an issue using git diff."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    if not registry_path or not registry_path.exists():
        print(f"No registry found. Cannot show diff for issue {issue_id}.", file=sys.stderr)
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    record = registry.get(issue_id)
    if record is None:
        print(f"Issue {issue_id} not found in registry.", file=sys.stderr)
        return 1

    branch_name = record.branch_name
    if not branch_name:
        print(f"Issue {issue_id} has no branch name recorded.", file=sys.stderr)
        return 1

    # Resolve workspace path
    workspace_root = getattr(args, "workspace", None)
    if workspace_root is None:
        workspace_root = os.environ.get("CLAWCODEX_WORKSPACE_ROOT")

    if not workspace_root:
        print(
            "Cannot resolve workspace root. Set CLAWCODEX_WORKSPACE_ROOT or use --workspace.",
            file=sys.stderr,
        )
        return 1

    ws_path = Path(workspace_root)
    if not ws_path.exists():
        print(f"Workspace not found: {ws_path}", file=sys.stderr)
        return 1

    previous_workspace = os.environ.get("CLAWCODEX_WORKSPACE_ROOT")
    os.environ["CLAWCODEX_WORKSPACE_ROOT"] = str(ws_path)
    try:
        issue_ws = _resolve_issue_workspace_path(issue_id)
    finally:
        if previous_workspace is None:
            os.environ.pop("CLAWCODEX_WORKSPACE_ROOT", None)
        else:
            os.environ["CLAWCODEX_WORKSPACE_ROOT"] = previous_workspace

    if issue_ws is None:
        for wd in ws_path.iterdir():
            if not wd.is_dir():
                continue
            metadata_file = wd / ".metadata"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    if metadata.get("issue_id") == issue_id:
                        issue_ws = wd
                        break
                except Exception:  # nosec B110
                    pass
            if wd.name == issue_id or issue_id in wd.name:
                issue_ws = wd
                break

    if issue_ws is None:
        print(f"Workspace not found for issue {issue_id}.", file=sys.stderr)
        return 1

    # Check if it's a git repository
    git_dir = issue_ws / ".git"
    if not git_dir.exists():
        # Not a git repo — show file tree instead
        return _show_diff_non_git(issue_ws, issue_id, args)

    import subprocess

    base_branch = record.base_branch or "main"

    # Get agent's run summary from comments (if available)
    agent_summary = _fetch_agent_summary(issue_id, ws_path)

    # Get diff compared to parent commit (this is what the agent actually changed)
    diff_target = _get_diff_target(issue_ws)

    # Get diff stat (summary)
    stat_result = subprocess.run(
        ["git", "diff", "--stat", diff_target],  # nosec B607
        cwd=str(issue_ws),
        capture_output=True,
        text=True,
        check=False,
    )

    # Also get the actual diff content
    diff_result = subprocess.run(
        ["git", "diff", "--no-color", diff_target],  # nosec B607
        cwd=str(issue_ws),
        capture_output=True,
        text=True,
        check=False,
    )

    show_full = getattr(args, "full", False)
    show_stat_only = getattr(args, "stat", False) and not show_full

    print(f"Issue {issue_id} — Changes")
    print(f"  Branch    : {branch_name}")
    print(f"  Base      : {base_branch}")
    if record.commit_sha:
        print(f"  Commit    : {record.commit_sha[:12]}")
    print()

    # Show agent summary if available
    if agent_summary:
        print("## Agent Summary")
        print(agent_summary)
        print()

    if stat_result.stdout.strip():
        print(stat_result.stdout)

    if show_full and diff_result.stdout.strip():
        print("--- Full Diff ---")
        print(diff_result.stdout)
    elif show_stat_only:
        pass  # stat already printed above
    else:
        # Default: show stat + first 50 lines of diff
        print("--- Diff Preview (use --full for complete output) ---")
        diff_lines = diff_result.stdout.strip().split("\n")
        if len(diff_lines) > 60:
            print("\n".join(diff_lines[:60]))
            print(f"\n  ... ({len(diff_lines) - 60} more lines, use --full to see all)")
        elif diff_lines:
            print("\n".join(diff_lines))

    return 0


def _get_diff_target(ws_path: Path) -> str:
    """Get the diff target (compare HEAD vs its parent commit)."""
    import subprocess

    # Get the parent commit hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],  # nosec B607
        cwd=str(ws_path),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        parent = result.stdout.strip()
        return f"{parent}...HEAD"

    # If no parent (first commit), show diff of working tree vs empty
    return "HEAD"


def _fetch_agent_summary(issue_id: str, ws_path: Path) -> str | None:
    """Fetch the agent's run summary from issue comments.

    Returns the first "## ClawCodex Run Complete" comment if found,
    otherwise returns None.
    """
    # Pattern to find safe stem for issue
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", issue_id.strip()).strip("-._")

    # Search in multiple possible locations for comments
    search_dirs = [
        ws_path,  # workspace root
        ws_path.parent / ".clawcodex_local_issues",
        ws_path.parent / ".clawcodex",
    ]

    for comments_dir in search_dirs:
        if not comments_dir.exists():
            continue

        # Find comment files matching this issue
        comment_files = list(comments_dir.glob(f"{safe_stem}*.comments.ndjson"))
        if not comment_files:
            # Also try with the issue directory name
            comment_files = list(comments_dir.glob(f"*{issue_id}*.comments.ndjson"))

        for cf in comment_files:
            try:
                for line in cf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    body = payload.get("body", "")
                    if "## ClawCodex Run Complete" in body:
                        # Extract the output excerpt section
                        if "**Output excerpt:**" in body:
                            idx = body.index("**Output excerpt:**")
                            return body[idx:]
                        elif body:
                            # Return the whole body as summary
                            return body[:500] if len(body) > 500 else body
            except Exception:  # nosec B110
                pass

    return None


def _has_origin(ws_path: Path) -> bool:
    """Check if the workspace has an origin remote."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "origin/HEAD"],  # nosec B607
        cwd=str(ws_path),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _show_diff_non_git(ws_path: Path, issue_id: str, args: argparse.Namespace) -> int:
    """Show file tree for non-git workspace."""
    print(f"Issue {issue_id} — Workspace Files (not a git repository)")
    print(f"  Workspace: {ws_path}")
    print()

    exclude = {".metadata", ".orchestrator_control", ".operator_hints.md"}

    files: list[tuple[str, str, int]] = []
    dirs: list[str] = []

    for item in sorted(ws_path.iterdir()):
        if item.name in exclude:
            continue
        if item.is_dir():
            dirs.append(item.name + "/")
        else:
            size = item.stat().st_size
            rel_path = item.relative_to(ws_path)
            files.append((str(rel_path), "file", size))

    if not files and not dirs:
        print("  (empty workspace)")
        return 0

    print(f"  {'FILE':<50} {'SIZE':>10}")
    print(f"  {'-' * 50} {'-' * 10}")

    for name, _, size in sorted(files):
        size_str = _format_size(size)
        print(f"  {name:<50} {size_str:>10}")

    for d in dirs:
        print(f"  {d:<50} {'[DIR]':>10}")

    print(f"\n  {len(files)} files, {len(dirs)} directories")
    print("\n  Note: This workspace is not a git repository — no diff available.")
    print("  Use 'clawcodex orchestrator issue workspace --id {} --cat <file>' to view file contents.".format(issue_id))
    return 0


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size // 1024}KB"
    else:
        return f"{size // (1024 * 1024)}MB"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_status_str(status) -> str:
    """Normalize status to string."""
    if hasattr(status, "value"):
        return status.value
    return str(status)


# ---------------------------------------------------------------------------
# issue retry (CLI fallback command)
# ---------------------------------------------------------------------------

# Single source of truth for the on-disk audit log location. Tests
# override this by monkey-patching `_DEFAULT_AUDIT_LOG_PATH` to a
# tempdir, so the production path is the only constant we expose.
_DEFAULT_AUDIT_LOG_PATH = Path.home() / ".clawcodex" / "orchestrator" / "audit.jsonl"


def _resolve_operator(explicit: str | None) -> str:
    """Resolve the operator login for audit logging.

    Priority: explicit --operator arg > $USER env > os.getlogin() > 'unknown'.
    """
    if explicit:
        return explicit
    env_user = os.environ.get("USER") or os.environ.get("USERNAME")
    if env_user:
        return env_user
    try:
        return os.getlogin()
    except Exception:
        return "unknown"


def _append_audit_log(
    *,
    issue_id: str,
    mode: str,
    reason: str,
    operator: str,
    force: bool,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Path | None:
    """Append a single JSONL line to the local audit log.

    Design: "~/.clawcodex/orchestrator/audit.jsonl records
    {ts, operator, issue_id, mode, reason} for traceability".

    Returns the path written, or None on I/O failure (the CLI surfaces
    audit failures to the operator as a warning but does not abort —
    the registry update is the user-visible side-effect).
    """

    target = path or _DEFAULT_AUDIT_LOG_PATH
    payload: dict[str, Any] = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operator": operator,
        "issue_id": issue_id,
        "mode": mode,
        "reason": reason,
        "force": force,
        "priority": "high" if force else "normal",
    }
    if extra:
        payload.update(extra)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return target
    except Exception as exc:
        print(
            f"warning: failed to write audit log {target}: {exc}",
            file=sys.stderr,
        )
        return None
