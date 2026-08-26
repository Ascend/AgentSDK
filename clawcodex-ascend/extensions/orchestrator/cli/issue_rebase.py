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

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from .issue_control import _resolve_sock_path, _send_and_wait, _write_control
from .issue_ops import _append_audit_log, _mirror_intent_label, _resolve_operator, _tracker_from_workflow_arg


def _run_rebase(registry_path: Path | None, args: argparse.Namespace, workspace_root: str | Path | None = None) -> int:
    """CLI fallback command - request a PR rebase via the built-in path.

    Unlike ``issue retry``, this command DOES NOT mutate the local
    registry intent directly. Instead it writes a control file that
    the daemon picks up on its next poll cycle and dispatches to
    ``_process_rebase_intent`` (which calls ``git_sync.rebase_for_pr``
    directly, with no external agent involvement when the rebase is
    clean). This avoids racing the daemon when it is mid-run.

    ``--force`` (default False) overrides two safe defaults:

      1. Uses plain ``git push --force`` instead of
         ``--force-with-lease``. May overwrite concurrent pushes.
      2. Bypasses the ``max_rebase_attempts_per_issue`` rate-limit
         gate. The audit entry is flagged high-priority in either
         case so the operator action is traceable.
    """
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required for rebase", file=sys.stderr)
        return 2
    force = bool(getattr(args, "force", False))
    reason = getattr(args, "reason", "") or ""
    operator = _resolve_operator(getattr(args, "operator", None))

    if registry_path is None or not registry_path.exists():
        print(
            "error: no issue registry found for this workspace.\n"
            "hint: run from a project root or pass --workspace / --workflow.",
            file=sys.stderr,
        )
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    record = registry.get_by_issue_ref(issue_id)
    if record is None:
        # Auto-register so the daemon can find the record on its next
        # poll. CLI rebase is a legitimate way to bootstrap an issue
        # record when the local daemon hasn't seen the issue yet.
        registry.register(
            issue_id=issue_id,
            issue_identifier=issue_id,
        )
        record = registry.get(issue_id)
        assert record is not None
    registry_issue_id = record.issue_id

    # Guard 1: the issue must have a known PR + workspace + branch.
    # Without these the rebase cannot be performed (no PR to push to,
    # no local workspace to operate on).
    if not record.pr_number or not record.workspace_path or not record.branch_name:
        print(
            f"error: issue {issue_id} ({record.issue_identifier}) has "
            f"no PR / workspace / branch registered. The rebase path "
            f"requires a previously-opened PR. Run a normal agent "
            f"cycle first to open a PR, then re-issue this command.",
            file=sys.stderr,
        )
        return 4

    # Mirror the intent onto the local registry so the daemon's
    # _resolve_intent sees REBASE even when the control file path is
    # unavailable (e.g. daemon already started its poll cycle).
    from extensions.orchestrator.tracker import Intent

    registry.mark_intent(
        registry_issue_id,
        Intent.REBASE,
        source="cli",
        command=f"cli:rebase:{reason[:64]}",
    )

    # Guard 2: best-effort rate-limit preview (the daemon enforces
    # the authoritative gate via _check_rebase_rate_limit). When the
    # current count already equals the configured cap and the
    # operator did NOT pass --force, warn but still write the control
    # file — the daemon's gate will surface a structured
    # ``rebase_rejected`` audit event instead of silently swallowing
    # the request.
    # Read the configured rebase attempt cap (fall back to default 3)
    max_attempts = 3
    workflow_path = getattr(args, "workflow", None)
    if workflow_path:
        try:
            from extensions.orchestrator.workflow import WorkflowLoader

            workflow, _ = WorkflowLoader.load(workflow_path)
            pr_conflict_scan = getattr(workflow, "pr_conflict_scan", None)
            if pr_conflict_scan is not None:
                max_attempts = getattr(pr_conflict_scan, "max_rebase_attempts_per_issue", max_attempts)
        except Exception:  # nosec B110
            pass
    if record.rebase_attempt_count >= max_attempts and not force:
        print(
            f"warning: issue {issue_id} has reached "
            f"rebase_attempt_count={record.rebase_attempt_count} >= "
            f"max_rebase_attempts_per_issue={max_attempts}. Pass "
            f"--force to bypass (logged as high-priority audit).",
            file=sys.stderr,
        )

    # Mirror the intent label onto the tracker (best-effort).
    tracker = _tracker_from_workflow_arg(args)
    if tracker is not None:
        _mirror_intent_label(tracker, issue_id, "agent:rebase", remove=False)

    # Append a JSONL entry to the local audit log so the operator
    # action is traceable.
    _append_audit_log(
        issue_id=issue_id,
        mode="rebase",
        reason=reason,
        operator=operator,
        force=force,
        extra={
            "issue_identifier": record.issue_identifier,
            "event": "rebase_requested",
            "priority": "high" if force else "normal",
            "push_method": "force" if force else "force-with-lease",
            "rebase_attempt_count": record.rebase_attempt_count,
            "max_rebase_attempts_per_issue": max_attempts,
            "pr_number": record.pr_number,
            "branch_name": record.branch_name,
            "base_branch": record.base_branch,
        },
    )

    # Write the control file that the daemon polls. Format:
    #   rebase\n<id>\nforce=0|1\n<reason>\n
    extra = f"force={'1' if force else '0'}\n{reason}"
    rc = _write_control("rebase", registry_issue_id, extra, workspace_root=workspace_root)
    if rc != 0:
        return rc

    print(f"Issue {issue_id} ({record.issue_identifier}): rebase requested.")
    print(f"  push method: {'--force' if force else '--force-with-lease'}")
    if reason:
        print(f"  reason: {reason}")
    print("  The orchestrator will run `rebase_for_pr` on its next poll cycle (default 30s).")
    return 0


def _run_retry(
    registry_path: Path | None,
    args: argparse.Namespace,
    *,
    workspace_root: str | Path | None = None,
) -> int:
    """CLI fallback command - record an operator-driven retry intent.

    Behaviour (per the design doc):

      * ``--mode reset``    — mark intent=RETRY, reset the registry record
                              to PENDING, and reopen the workflow tracker
                              issue so the daemon can pick it up.
      * ``--mode followup`` — mark intent=FOLLOWUP so Sub-C reuses the
                              existing branch.
      * ``--mode unblock``  — call IssueRegistry.unblock() to roll an
                              ABANDONED issue back to PENDING.

    All three branches append a JSONL entry to the local audit log
    (~/.clawcodex/orchestrator/audit.jsonl) so the action is
    traceable. ``--force`` flags the audit entry as high-priority
    and signals that the rate limit (Sub-F) was bypassed.
    """
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required for retry", file=sys.stderr)
        return 2
    mode = getattr(args, "mode", None)
    if mode not in {"reset", "followup", "unblock"}:
        print(f"error: --mode must be reset|followup|unblock, got {mode!r}", file=sys.stderr)
        return 2
    reason = getattr(args, "reason", "") or ""
    force = bool(getattr(args, "force", False))
    operator = _resolve_operator(getattr(args, "operator", None))
    max_retries = int(getattr(args, "max_retries", 3) or 3)

    if registry_path is None or not registry_path.exists():
        print(
            "error: no issue registry found for this workspace.\n"
            "hint: run from a project root or pass --workspace / --workflow.",
            file=sys.stderr,
        )
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry
    from extensions.orchestrator.tracker import Intent

    registry = IssueRegistry(registry_path)
    record = registry.get_by_issue_ref(issue_id)

    # --stop-first: if the agent is still running, stop it
    # before retrying. Equivalent to 'issue stop' + 'issue retry'.
    stop_first = bool(getattr(args, "stop_first", False))
    if stop_first and record is not None and record.status.value == "running":
        sock_path = _resolve_sock_path(issue_id, workspace_root)
        if sock_path is not None:
            print(f"Stopping running agent for {issue_id} before retry…")

            async def _stop_for_retry() -> bool:
                data = await _send_and_wait(sock_path, "stop", "", "SessionComplete", timeout=10.0)
                return data is not None

            stopped = asyncio.run(_stop_for_retry())
            if stopped:
                print("Agent stopped. Proceeding with retry.")
            else:
                print(
                    "warning: stop sent but agent may still be unwinding. Proceeding with retry anyway.",
                    file=sys.stderr,
                )
        else:
            print(
                f"warning: could not find control socket for {issue_id}. Writing stop control file as fallback.",
                file=sys.stderr,
            )
            _write_control("stop", issue_id, workspace_root=workspace_root)
    elif stop_first and record is not None and record.status.value != "running":
        print(f"Issue {issue_id} is not running (status: {record.status.value}). No need to stop before retry.")
    if record is None:
        # Auto-register so the daemon can find the record on its next
        # poll. CLI retry is a legitimate way to bootstrap an issue
        # record when the local daemon hasn't seen the issue yet.
        registry.register(
            issue_id=issue_id,
            issue_identifier=issue_id,
        )
        record = registry.get(issue_id)
        assert record is not None  # just registered
    registry_issue_id = record.issue_id

    # ``--mode reset`` is itself a fresh-start bypass: it clears
    # ``retry_count`` via ``reset_for_retry(reset_retry_count=True)``,
    # so the ``max_retries_per_issue`` cap does not apply (you cannot
    # be locked out of a command whose whole point is to wipe the
    # lock). ``--force`` is still accepted as an audit-priority
    # marker (the original design required high-priority entries
    # for cap bypasses) but no longer gates the cap check.
    #
    # Other retry paths (label-driven ``agent:retry``, comment-driven
    # ``/agent retry``) DO respect the cap — they live in
    # ``orchestrator._resolve_intent`` / ``mark_intent`` and call
    # ``reset_for_retry(increment_retry=True)`` to bump the budget
    # one tick at a time.
    control_rc = 0

    # Obtain the tracker once so we can
    # mirror the CLI intent onto the remote issue label AND
    # reopen the issue. The tracker is optional — operators
    # who run from a directory without a workflow.md will get
    # None and the local registry.intent (written just below)
    # is still the authoritative source of truth.
    tracker = _tracker_from_workflow_arg(args)
    if mode == "reset":
        registry.mark_intent(
            registry_issue_id,
            Intent.RETRY,
            source="cli",
            command=f"cli:reset:{reason[:64]}",
        )
        # ``mode=reset`` is semantically a fresh start: clear the
        # previous failure state AND reset the rate-limit budget so a
        # transient daemon/agent bug that consumed the previous
        # retries does not permanently lock the issue. Other retry
        # paths (label-driven ``agent:retry``, comment-driven
        # ``/agent retry``) keep the historical ``+= 1`` behaviour
        # via the default ``increment_retry=True``.
        registry.reset_for_retry(registry_issue_id, reset_retry_count=True)
        if tracker is not None:
            try:

                async def reopen_tracker_issue() -> None:
                    try:
                        await tracker.update_issue_state(issue_id, "open")
                    except FileNotFoundError:
                        if registry_issue_id == issue_id:
                            raise
                        await tracker.update_issue_state(registry_issue_id, "open")

                asyncio.run(reopen_tracker_issue())
            except Exception as exc:
                print(f"Warning: could not update tracker: {exc}", file=sys.stderr)
            # Mirror the retry intent onto the remote issue
            # label so label-based intent resolution sees the
            # same intent. Best-effort: the tracker may not
            # implement add_label (returns False), or the API
            # call may fail — both are non-fatal because the
            # local registry.intent is the authoritative
            # source.
            _mirror_intent_label(tracker, issue_id, "agent:retry", remove=False)
        # The CLI may run inside the IM gateway process while the
        # orchestrator daemon keeps a separate, already-loaded
        # IssueRegistry instance. Persisting the registry alone does
        # not update that in-memory state (notably its completed set),
        # so explicitly notify the daemon through its control queue.
        control_root = workspace_root or registry_path.parent
        control_rc = _write_control(
            "retry",
            registry_issue_id,
            reason,
            workspace_root=control_root,
        )
        action = "marked for reset" if control_rc == 0 else "reset persisted, but daemon notification failed"
    elif mode == "followup":
        registry.mark_intent(
            registry_issue_id,
            Intent.FOLLOWUP,
            source="cli",
            command=f"cli:followup:{reason[:64]}",
        )
        if tracker is not None:
            _mirror_intent_label(tracker, issue_id, "agent:follow-up", remove=False)
        action = "marked for follow-up"
    else:  # mode == "unblock"
        registry.unblock(registry_issue_id)
        if tracker is not None:
            _mirror_intent_label(tracker, issue_id, "agent:blocked", remove=True)
        action = "unblocked"
    audit_priority = "high" if force else "normal"
    audit_event = "retry" if mode == "reset" else mode

    _append_audit_log(
        issue_id=issue_id,
        mode=mode,
        reason=reason,
        operator=operator,
        force=force,
        extra={
            "issue_identifier": record.issue_identifier,
            "event": audit_event,
            "priority": audit_priority,
            "retry_count": record.retry_count,
            "max_retries_per_issue": max_retries,
        },
    )

    print(f"Issue {issue_id} ({record.issue_identifier}): {action}.")
    if reason:
        print(f"  reason: {reason}")
    print(f"  operator: {operator}")
    if control_rc == 0:
        print("  The orchestrator will pick this up on its next poll cycle.")
    else:
        print(
            "  The local reset was saved, but the orchestrator control command failed.",
            file=sys.stderr,
        )
    return control_rc


# ── issue init ───────────────────────────────────────────────────────


def _run_init(args: argparse.Namespace) -> int:
    """Scaffold an issue card from the issue-card.template.md."""
    # Locate template
    import extensions.orchestrator.templates as tpl_mod
    from datetime import datetime, timezone

    tpl = None
    for p in tpl_mod.__path__:  # type: ignore[attr-defined]
        candidate = Path(p) / "issue-card.template.md"
        if candidate.exists():
            tpl = candidate
            break

    if tpl is None:
        print("✗ Cannot locate issue-card.template.md — your install may be corrupt.", file=sys.stderr)
        return 1

    # Determine output path
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        print(f"✗ {out} already exists — remove it first or use --output", file=sys.stderr)
        return 1

    interactive = sys.stdin.isatty() and not args.non_interactive

    def val(flag_val: str, label: str, default: str = "") -> str:
        if flag_val:
            return flag_val
        if interactive:
            try:
                raw = input(f"  {label} [{default}]: ")
                return raw.strip() or default
            except (EOFError, KeyboardInterrupt):
                return default
        return default

    issue_id = val(args.id, "Issue ID (e.g. <ID>-pr-auto-fix)", "")
    identifier = val(args.identifier, "Short identifier (e.g. <id>)", "")
    title = val(args.title, "Issue title", "")
    priority = val(args.priority, "Priority (0-3)", "3")
    state = args.state or "open"
    category = val(args.category, "Category label (e.g. feature, bug, refactor)", "feature")
    branch_name = val(args.branch_name, "Preferred branch name (blank for auto)", "")
    base_branch = val(args.base_branch, "Base branch (e.g. main, dev-decoupling)", "")
    assignee = val(args.assignee, "Assignee / team", "")
    url = val(args.url, "Upstream issue / document URL", "")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Read and replace all <...> placeholders
    raw = tpl.read_text(encoding="utf-8")
    replacements = {
        "<ID>": issue_id,
        "<IDENTIFIER>": identifier,
        "<TITLE>": title,
        "<PRIORITY>": priority,
        "<STATE>": state,
        "<CATEGORY_TAG>": category,
        "<BRANCH_NAME>": branch_name,
        "<BASE_BRANCH>": base_branch,
        "<ASSIGNEE>": assignee,
        "<UPSTREAM_URL>": url,
        "<ISO8601>": now,
    }
    for key, replacement in replacements.items():
        raw = raw.replace(key, replacement)

    # Write
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw, encoding="utf-8")

    remaining = bool(re.search(r"<[A-Z_]+>", raw))
    print(f"✓ Generated {out}")
    print()
    print("  Next steps:")
    if remaining:
        print(f"    1. Edit {out.name} — review and fill any remaining <...> placeholders")
    else:
        print(f"    1. Review {out.name} — all placeholders have been filled")
    print("    2. Move it to your local tracker's issues path")
    print("    3. Start: clawcodex orchestrator server start --workflow workflow.md")
    return 0
