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

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any


from .events import EventLevel
from .issue import Issue
from .issue_registry import IssueStatus
from .tracker import (
    Command,
    CommandIntentCapability,
    Intent,
    command_to_intent,
    merge_intents_with_cli,
    supports,
)

if TYPE_CHECKING:
    from .tracker import CommandIntent

logger = logging.getLogger(__name__)

_CONTINUATION_RETRY_DELAY_MS = 1_000
_FAILURE_RETRY_BASE_MS = 10_000


def _operator_failure_detail(exc: BaseException) -> str:
    """Return a concise failure detail suitable for IM and registry records."""

    raw = " ".join(str(exc).split())
    body_detail = _extract_error_message_from_body(raw)
    if body_detail:
        status_code = _extract_status_code(raw)
        if raw.startswith("request_failed") and status_code:
            return f"request_failed status={status_code}: {body_detail}"
        return body_detail
    return raw or exc.__class__.__name__


def _extract_status_code(text: str) -> str | None:
    for part in text.split():
        if part.startswith("status="):
            status = part.removeprefix("status=").strip()
            if status:
                return status
    return None


def _extract_error_message_from_body(text: str) -> str | None:
    marker = "body="
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    body = text[marker_index + len(marker) :].strip()
    if not body:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    return _extract_error_message(payload)


def _extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in (
            "error_message",
            "message",
            "error_description",
            "detail",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return " ".join(error.split())
        nested = _extract_error_message(error)
        if nested:
            return nested
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors:
                nested = _extract_error_message(item)
                if nested:
                    return nested
    return None


class OrchestratorSessionMixin:
    def _derive_orchestrator_session_id(self) -> str:
        """Stable session id for the orchestrator daemon.

        Combines the workspace root path with a daily salt so all
        orchestrator daemons on a given day share the same id
        (the polling loop is one continuous session for telemetry
        purposes — restart on a new day = new session).
        """
        try:
            workspace = str(self._workspace_root) if self._workspace_root else ""
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            raw = f"orchestrator:{workspace}:{day}"
            return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        except Exception:
            return "orchestrator"

    async def _recover_stale_running_records(self) -> None:
        reason = "Recovered stale running issue on orchestrator startup"
        stale_records = self._registry.running_records()
        for record in stale_records:
            self._registry.mark_failed_with_reason(record.issue_id, reason)
            await self._sync_tracker_issue_state(record.issue_id, "failed")
            logger.warning(
                "Recovered stale running issue_id=%s on orchestrator startup",
                record.issue_id,
            )

    async def _metadata_heartbeat_loop(self) -> None:
        """Periodically rewrite metadata so CLI can always discover the orchestrator.

        If metadata.json is accidentally deleted, this recreates it within
        the heartbeat interval (30s), preventing the ``server start`` PID
        guard from being bypassed for a running instance.
        """
        from .workspace_locator import write_orchestrator_metadata

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=30.0,
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass

            write_orchestrator_metadata(
                workspace_root=self._workspace_root,
                workflow_path=self._workflow_path,
                started_at=self._metadata_started_at,
            )

    async def shutdown(self) -> None:
        """Signal graceful shutdown and clean up metadata."""
        self._shutdown_event.set()
        # Clean up orchestrator metadata
        from .workspace_locator import clear_orchestrator_metadata

        clear_orchestrator_metadata(self._workspace_root)

    def _workflow_mtime_ns(self) -> int | None:
        """Return the workflow modification time, if this daemon has a file."""
        if not self._workflow_path:
            return None
        try:
            return Path(self._workflow_path).stat().st_mtime_ns
        except OSError:
            return None

    def _refresh_dynamic_title_prefix_filter(self) -> None:
        """Apply changed title-prefix settings before fetching candidates.

        Only this lightweight tracker setting is hot-reloaded.  Other workflow
        settings can affect running work and retain the daemon-start snapshot.
        """
        mtime_ns = self._workflow_mtime_ns()
        if mtime_ns is None or mtime_ns == self._dynamic_tracker_config_mtime_ns:
            return
        # Record first so a malformed edit does not generate a warning every
        # polling interval. A subsequent file save will be retried.
        self._dynamic_tracker_config_mtime_ns = mtime_ns
        try:
            from .workflow import WorkflowLoader

            refreshed, _ = WorkflowLoader.load(self._workflow_path or "")
        except Exception as exc:
            logger.warning("workflow title-prefix reload failed: %s", exc)
            return

        if refreshed.tracker.kind != self.workflow.tracker.kind:
            logger.warning(
                "workflow tracker kind changed from %s to %s; title-prefix reload ignored",
                self.workflow.tracker.kind,
                refreshed.tracker.kind,
            )
            return
        configure = getattr(self.tracker, "configure_title_prefix_filter", None)
        if not callable(configure):
            logger.warning("tracker does not support dynamic title-prefix filtering")
            return
        configure(
            refreshed.tracker.title_prefixes,
            refreshed.tracker.title_prefix_match,
        )
        self.workflow.tracker.title_prefixes = refreshed.tracker.title_prefixes
        self.workflow.tracker.title_prefix_match = refreshed.tracker.title_prefix_match
        logger.info(
            "reloaded title-prefix filter: mode=%s prefixes=%s",
            refreshed.tracker.title_prefix_match,
            refreshed.tracker.title_prefixes,
        )

    async def _poll_and_dispatch(self) -> None:
        """Fetch candidates, respect concurrency limit, launch runs."""
        self.status_dashboard.on_poll_start()
        self._state.poll_check_in_progress = True

        try:
            self._refresh_dynamic_title_prefix_filter()
            # Process lifecycle control commands (pause/resume/stop/takeover)
            await self._process_control_commands()

            # Poll clarification answers (Channel 2 + Channel 3)
            await self._clarification_resolver.poll_clarification_answers()

            # Process retry queue first
            await self._process_retry_queue()

            # Handle escalated (clarification-exhausted) issues
            await self._process_escalated_issues()

            await self._process_review_feedback()

            # Launch agent_rebase for PRs with content conflicts
            await self._process_pending_rebase_conflicts()
            # Optional PR mergeable-state scan (opt-in via workflow.md)
            await self._process_pr_conflict_scan()

            # Fetch new candidate issues
            try:
                issues = await self.tracker.fetch_candidate_issues()
            except Exception as exc:
                logger.error("Failed to fetch candidate issues: %s", exc)
                return

            available_slots = self._state.max_concurrent_agents - len(self._state.running)
            if self._clarification_gate is not None:
                self._clarification_gate.begin_poll()

            # Pre-register all unregistered candidates with QUEUED status
            # so the dashboard / registry reflects the full backlog.
            for issue in issues:
                if not self._registry.get(issue.id or ""):
                    base_branch = getattr(issue, "base_branch", None) or self.workflow.workspace.base_branch or "main"
                    self._registry.register(
                        issue_id=issue.id or "",
                        issue_identifier=issue.identifier or "",
                        branch_name=issue.branch_name,
                        base_branch=base_branch,
                        status=IssueStatus.QUEUED,
                        author_login=issue.author_login,
                    )
                    # Notify the operator that a new issue was discovered.
                    # The Issue object (with url, title, identifier) is
                    # directly in scope here — all tracker adapters
                    # populate issue.url from the platform API response.
                    self._emit_im_event(
                        issue.id or "",
                        "issue.detected",
                        EventLevel.INFO,
                        "新增 ISSUE",
                        self._issue_payload(issue, url=issue.url),
                    )
                elif issue.author_login:
                    record = self._registry.get(issue.id or "")
                    if record is not None and not record.author_login:
                        record.author_login = issue.author_login
                        self._registry._save()

            if self.workflow.workspace.strategy == "sequential" and self._state.running:
                return

            launched_this_poll = 0
            for issue in issues:
                if launched_this_poll >= available_slots:
                    break
                if issue.id in self._state.running:
                    continue
                if issue.id in self._state.claimed:
                    continue

                # Intent resolution must
                # happen BEFORE the completed/pending_review skip so
                # operators can trigger RETRY / FOLLOWUP on completed
                # issues via labels, comments, or CLI.
                intent, command_intent_obj, intent_source = await self._resolve_intent(issue)

                if issue.id in self._state.completed or issue.id in self._state.pending_review:
                    if intent not in (Intent.RETRY, Intent.FOLLOWUP):
                        continue
                # `command_intent_obj` may carry the comment author
                # for role checks; the bare `Command` value
                # is in `command_intent_obj.command`.
                command = command_intent_obj.command if command_intent_obj is not None else None
                command_author = command_intent_obj.author_login if command_intent_obj is not None else None

                # Role check. If a comment command is
                # what triggered the intent, only the issue author or
                # a maintainer (or `allow_anyone_to_retry=True`) is
                # allowed to fire it. The check happens BEFORE the
                # acknowledgement comment is posted, so a rejected
                # command never advances the cursor.
                if (
                    command_intent_obj is not None
                    and intent in (Intent.RETRY, Intent.FOLLOWUP)
                    and not self._is_command_author_eligible(issue, command_author)
                ):
                    await self._reject_unauthorized_command(issue, command_intent_obj)
                    continue

                # Rate limit on RETRY intent. If the issue
                # has hit `max_retries_per_issue`, refuse the reset
                # (even with `--force`; only the label-based retry
                # honors force in the daemon path).
                if intent is Intent.RETRY:
                    if not self._check_retry_rate_limit(issue, force=False):
                        continue

                # When a comment command is honored, post
                # a bot acknowledgement so the operator sees the
                # intent was received, and record the command on the
                # registry for audit.
                if command is not None:
                    await self._post_command_acknowledgement(issue, command)
                    record = self._registry.get(issue.id or "")
                    if record is not None:
                        record.last_command = f"/agent {command.value}"
                        record.touch()
                        self._registry._save()
                    logger.info(
                        "Issue %s command received: /agent %s",
                        issue.id,
                        command.value,
                    )

                    # UNBLOCK is a meta-command: clear any BLOCKED
                    # state so the next poll re-applies the (now
                    # possibly cleared) label-based intent.
                    if command is Command.UNBLOCK:
                        record = self._registry.get(issue.id or "")
                        if record is not None and record.status is IssueStatus.ABANDONED:
                            logger.info(
                                "Issue %s unblocked, status reset to pending",
                                issue.id,
                            )
                            record.status = IssueStatus.PENDING
                            record.intent = Intent.NONE
                            record.intent_source = None
                            self._registry._save()

                if intent is Intent.BLOCKED:
                    logger.info(
                        "Issue %s blocked intent detected, marking abandoned",
                        issue.id,
                    )
                    record = self._registry.get(issue.id or "")
                    if record is None:
                        self._registry.register(
                            issue_id=issue.id or "",
                            issue_identifier=issue.identifier or "",
                            branch_name=getattr(issue, "branch_name", None) or "main",
                        )
                    self._registry.mark_intent(
                        issue.id or "",
                        intent,
                        # Preserve the source from
                        # _resolve_intent so CLI / comment / label
                        # origin is recorded on the record. The
                        # fallback only fires if intent_source is
                        # somehow None (defensive — should not be
                        # reachable when intent is RETRY/FOLLOWUP/
                        # BLOCKED).
                        source=(intent_source or ("command" if command is not None else "label")),
                        command=(f"/agent {command.value}" if command is not None else None),
                    )
                    self._registry.mark_abandoned(issue.id or "")
                    await self._sync_tracker_issue_state(issue.id or "", "abandoned")
                    self._state.completed.add(issue.id or "")
                    continue

                if intent is Intent.RETRY:
                    logger.info(
                        "Issue %s retry intent detected, will reset on launch",
                        issue.id,
                    )
                    self._registry.mark_intent(
                        issue.id or "",
                        intent,
                        # Preserve the source from
                        # _resolve_intent so CLI / comment / label
                        # origin is recorded on the record. The
                        # fallback only fires if intent_source is
                        # somehow None (defensive — should not be
                        # reachable when intent is RETRY/FOLLOWUP/
                        # BLOCKED).
                        source=(intent_source or ("command" if command is not None else "label")),
                        command=(f"/agent {command.value}" if command is not None else None),
                    )
                    # The reset+close path performs the actual reset.
                elif intent is Intent.FOLLOWUP:
                    logger.info(
                        "Issue %s follow-up intent detected, will reuse branch",
                        issue.id,
                    )
                    self._registry.mark_intent(
                        issue.id or "",
                        intent,
                        # Preserve the source from
                        # _resolve_intent so CLI / comment / label
                        # origin is recorded on the record. The
                        # fallback only fires if intent_source is
                        # somehow None (defensive — should not be
                        # reachable when intent is RETRY/FOLLOWUP/
                        # BLOCKED).
                        source=(intent_source or ("command" if command is not None else "label")),
                        command=(f"/agent {command.value}" if command is not None else None),
                    )
                    # The follow-up path performs the actual follow-up.

                if intent is Intent.REBASE:
                    # REBASE intent — the orchestrator itself
                    # performs the rebase (no agent for clean rebases).
                    # On content conflict, has_conflict is set and the
                    # next ``_process_pending_rebase_conflicts`` cycle
                    # launches an agent_rebase run.
                    logger.info(
                        "Issue %s rebase intent detected, running built-in rebase",
                        issue.id,
                    )
                    self._registry.mark_intent(
                        issue.id or "",
                        intent,
                        source=(intent_source or ("command" if command is not None else "label")),
                        command=(f"/agent {command.value}" if command is not None else None),
                    )
                    if not self._check_rebase_rate_limit(issue, force=False):
                        continue
                    await self._process_rebase_intent(issue)
                    # CLI is one-shot; clear so the next poll doesn't
                    # re-trigger. Audit + last_command are preserved.
                    if intent_source == "cli":
                        self._registry.clear_intent(issue.id or "")
                    continue

                # Skip terminal registry records even if the tracker still
                # exposes the issue in an active state. Explicit retry/follow-up
                # intents are the only daemon path that may reopen handled work.
                if intent is Intent.NONE and (
                    self._registry.is_terminal(issue.id or "") or self._registry.has_pr(issue.id or "")
                ):
                    logger.info("Issue %s already handled (registry), skipping", issue.id)
                    continue
                if not await self._dependencies_satisfied(issue):
                    continue
                if self._clarification_gate is not None:
                    try:
                        if not await self._clarification_gate.should_dispatch(issue):
                            logger.info("Issue %s is waiting for issue-clarifier clarification", issue.id)
                            continue
                    except Exception:
                        logger.exception("Issue-clarifier clarity gate failed for issue %s", issue.id)
                        if not bool(getattr(self.workflow.clarifier, "fail_open", True)):
                            continue
                self._state.claimed.add(issue.id)
                # Thread-local MDC for the orchestrator launch path —
                # the agent_runner will refill with run_id once available.
                from .logging_setup import set_log_context

                set_log_context(
                    issue_id=str(issue.id or ""),
                    issue_identifier=str(getattr(issue, "identifier", "")),
                )
                await self._launch_issue(issue)
                if issue.id in self._state.running:
                    launched_this_poll += 1
                    # CLI retry is a one-shot. The
                    # operator's `clawcodex-dev orchestrator issue
                    # retry --mode reset` already wrote `registry.intent`
                    # with `intent_source="cli"`; now that the launch
                    # has started, clear it so the next poll does NOT
                    # re-trigger. The audit trail (the original
                    # `last_command` text + the high-priority audit
                    # log entry written by the CLI) is preserved.
                    if intent_source == "cli":
                        self._registry.clear_intent(issue.id or "")

        finally:
            self._state.poll_check_in_progress = False
            self.status_dashboard.on_poll_end()
            # Broadcast clarification status to the dashboard after polling
            self._broadcast_clarification_status()

    async def _dependencies_satisfied(self, issue: Issue) -> bool:
        dependencies = [dep for dep in getattr(issue, "depends_on", []) if dep]
        if not dependencies:
            return True

        unresolved = [
            dependency
            for dependency in dependencies
            if not (self._registry.is_completed(dependency) or self._registry.has_pr(dependency))
        ]
        if unresolved:
            logger.info(
                "Issue %s waiting for dependencies: %s",
                issue.id,
                ", ".join(unresolved),
            )
            return False
        return True

    async def _resolve_intent(
        self,
        issue: Issue,
    ) -> tuple[Intent, "CommandIntent | None", str | None]:
        """Resolve the current operator intent for an issue.

        Merges three intent sources:
          1. Label-based intent (Sub-A: `agent:retry` / `agent:follow-up`
             / `agent:blocked`).
          2. Comment-based command (Sub-D: `/agent retry` / `/agent
             follow-up` / `/agent unblock`).
          3. Registry-based CLI intent (Sub-E: `clawcodex-dev
             orchestrator issue retry --mode reset|followup|unblock`
             writes `registry.intent` with `intent_source="cli"`).

        Priority (high → low): BLOCKED is sticky; CLI beats comment
        beats label. CLI is the operator's authoritative local command
        and must survive even when the remote issue tracker is
        unreachable / read-only / local-only (LocalTracker).

        Returns ``(intent, command_intent_obj, intent_source)``:
          * ``intent`` — merged Intent for the launch.
          * ``command_intent_obj`` — the raw CommandIntent (with the
            comment's author login for the role check) if a
            comment command was honored, else None.
          * ``intent_source`` — the source that won the merge
            (``"cli"`` | ``"command"`` | ``"label"`` | None) so the
            caller can preserve the audit trail in `mark_intent` and
            decide whether to clear the intent after launch.
        """
        labels = list(getattr(issue, "labels", None) or [])
        label_intent = Intent.NONE
        if labels:
            try:
                label_intent = await self.tracker.extract_intent_from_labels(labels)
            except Exception as exc:
                logger.warning(
                    "Failed to extract intent from labels for issue %s: %s",
                    issue.id,
                    exc,
                )

        # Comment command intent.
        command_intent_obj = await self._resolve_command_intent(issue)
        command = command_intent_obj.command if command_intent_obj is not None else None
        command_intent = command_to_intent(command) if command is not None else Intent.NONE

        # CLI fallback intent. The CLI is the operator's
        # authoritative local command, so we read it directly from
        # `registry.intent` whenever the record carries
        # `intent_source="cli"`. The CLI path does NOT require the
        # remote issue tracker to be reachable, so this is also the
        # only intent source that works for LocalTracker users and
        # for operators working offline.
        cli_intent = Intent.NONE
        record = self._registry.get(issue.id or "")
        if record is not None and getattr(record, "intent_source", None) == "cli":
            raw_intent = getattr(record, "intent", None)
            if raw_intent:
                try:
                    cli_intent = Intent(raw_intent)
                except ValueError:
                    logger.warning(
                        "Issue %s has unknown CLI intent %r, ignoring",
                        issue.id,
                        raw_intent,
                    )
                    cli_intent = Intent.NONE

        merged = merge_intents_with_cli(label_intent, command_intent, cli_intent)

        # Track which source won so downstream `mark_intent` calls
        # preserve the audit trail. The order matches the merge
        # priority (BLOCKED > CLI > command > label).
        intent_source: str | None = None
        if merged is Intent.BLOCKED:
            if label_intent is Intent.BLOCKED:
                intent_source = "label"
            elif command_intent is Intent.BLOCKED:
                intent_source = "command"
            elif cli_intent is Intent.BLOCKED:
                intent_source = "cli"
        elif cli_intent is not Intent.NONE and merged is cli_intent:
            intent_source = "cli"
        elif command_intent is not Intent.NONE and merged is command_intent:
            intent_source = "command"
        elif label_intent is not Intent.NONE and merged is label_intent:
            intent_source = "label"

        return merged, command_intent_obj, intent_source

    async def _resolve_command_intent(self, issue: Issue) -> "CommandIntent | None":
        """Fetch and parse the most recent /agent command.

        The returned `CommandIntent` carries the comment
        author so the caller can perform the role check. Adapters that
        don't expose author info will return `author_login=None`, in
        which case `_is_command_author_eligible` will reject the
        command (fail-closed) to avoid the LLM-self-trigger risk.
        """
        issue_id = issue.id or ""
        if not issue_id:
            return None
        record = self._registry.get(issue_id)
        cursor = record.command_cursor if record is not None else None
        if not supports(self.tracker, CommandIntentCapability):
            return None
        try:
            return await self.tracker.fetch_issue_command_intent(issue_id, cursor)
        except Exception as exc:
            logger.warning(
                "Failed to fetch issue command intent for %s: %s",
                issue_id,
                exc,
            )
            return None

    async def _post_command_acknowledgement(
        self,
        issue: Issue,
        command: "Command",
    ) -> str | None:
        """Post a bot confirmation comment and update cursor.

        The confirmation comment includes a metadata HTML comment
        with `command_cursor` so the next poll knows where to resume
        scanning. Returns the created comment ID, or None on
        failure.
        """
        issue_id = issue.id or ""
        body = f"## ClawCodex: 已受理 /agent {command.value}\n\n下一轮 poll 开始执行。\n"
        try:
            comment = await self.tracker.create_comment(issue_id, body)
        except Exception as exc:
            logger.warning(
                "Failed to post command acknowledgement for %s: %s",
                issue_id,
                exc,
            )
            return None
        comment_id = getattr(comment, "id", None) if comment is not None else None
        if comment_id:
            record = self._registry.get(issue_id)
            if record is not None:
                record.command_cursor = comment_id
                self._registry._save()
        return comment_id

    # ------------------------------------------------------------------
    # Role check + rate-limit guard
    # ------------------------------------------------------------------

    def _is_command_author_eligible(
        self,
        issue: Issue,
        author_login: str | None,
    ) -> bool:
        """Return True if `author_login` may trigger a retry/follow-up.

        Per the design doc: "comment commands require the issue
        author or repo maintainer by default". The check has three
        short-circuits:

          1. `workflow.agent.allow_anyone_to_retry` — disables the
             role check entirely (trusted-team mode).
          2. `author_login` is None — fail-closed. Adapters that
             don't expose author info cannot pass the check; this
             prevents the LLM-self-trigger risk where a bot
             accidentally writes `/agent retry` in its own reply
             and the daemon can't tell it wasn't a human.
          3. The bot itself (`clawcodex`) is always allowed so the
             CLI fallback (`/agent retry` from a local operator
             routed through the bot) isn't rejected. NOTE: the CLI
             path doesn't actually go through this code path; this
             branch is only here to be lenient on platform quirks
             where the bot appears as the author of its own ack
             comment.

        Otherwise, the author must equal the issue author login
        (kept in `IssueRecord.author_login`, populated by the
        clarification flow) or a maintainer login (platform
        metadata; we fall back to None for now and rely on the
        author check).
        """
        if getattr(self.workflow.agent, "allow_anyone_to_retry", False):
            return True
        if not author_login:
            # Fail-closed: if we don't know who wrote the command,
            # we cannot certify they are not the LLM itself.
            return False
        if author_login == "clawcodex":
            return True
        record = self._registry.get(issue.id or "")
        issue_author = getattr(record, "author_login", None) if record else None
        return bool(issue_author and author_login == issue_author)

    async def _reject_unauthorized_command(
        self,
        issue: Issue,
        command_intent: "CommandIntent",
    ) -> None:
        """Post a comment rejecting an unauthorized command.

        Per the design acceptance criteria: "when a user posts `/agent
        retry` in an issue comment and is not the original author, the
        daemon rejects it and comments `## ClawCodex: only the issue
        author or maintainer can trigger /agent retry`".
        """
        issue_id = issue.id or ""
        body = (
            f"## ClawCodex: 仅 issue 作者或 maintainer 可触发 "
            f"/agent {command_intent.command.value}\n\n"
            f"author=`{command_intent.author_login or '<unknown>'}` "
            f"not authorized; ignored.\n"
        )
        try:
            await self.tracker.create_comment(issue_id, body)
        except Exception as exc:
            logger.warning(
                "Failed to post unauthorized-command rejection for %s: %s",
                issue_id,
                exc,
            )
        logger.info(
            "Issue %s command rejected: /agent %s by %s (not authorized)",
            issue_id,
            command_intent.command.value,
            command_intent.author_login,
        )
        self._log_audit_event(
            issue_id=issue_id,
            event="unauthorized_command",
            mode=f"command:{command_intent.command.value}",
            reason="role_check_failed",
            author=command_intent.author_login or "unknown",
        )

    def _check_retry_rate_limit(
        self,
        issue: Issue,
        *,
        force: bool = False,
    ) -> bool:
        """Refuse a RETRY when retry_count >= max_retries_per_issue.

        Returns True if the retry is allowed (and bumps
        `retry_count` for the record), or False if the rate limit
        was hit. The caller is responsible for the actual reset
        work; this helper is a guard.

        On a hit, this method:
          * Logs the rejection.
          * Appends an `agent:retry-rejected` label to the issue
            (best-effort).
          * Posts a comment explaining the rejection.
          * Records a high-priority audit.jsonl entry.
        """
        issue_id = issue.id or ""
        max_retries = getattr(self.workflow.agent, "max_retries_per_issue", 3)
        record = self._registry.get(issue_id)
        current = record.retry_count if record else 0
        if current < max_retries:
            return True
        if force:
            # `force=True` is reserved for the CLI path, which
            # logs its own audit entry. The daemon path passes
            # `force=False` and is therefore rejected on the
            # `current >= max_retries` branch.
            return True
        # Rate limit hit; do the side-effects.
        logger.warning(
            "Issue %s retry rate limit hit: %d >= %d",
            issue_id,
            current,
            max_retries,
        )
        self._log_audit_event(
            issue_id=issue_id,
            event="retry_rejected",
            mode="label:agent:retry",
            reason=f"retry_count={current} >= max_retries_per_issue={max_retries}",
            author="daemon",
        )
        # Best-effort: add the agent:retry-rejected label and
        # post a comment. Failures here are logged but do not
        # change the verdict (False = reject).
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.create_task(self._post_retry_rejection(issue_id, current, max_retries))
        else:
            asyncio.run(self._post_retry_rejection(issue_id, current, max_retries))
        return False

    async def _post_retry_rejection(
        self,
        issue_id: str,
        current: int,
        max_retries: int,
    ) -> None:
        """Best-effort label + comment for rate-limit hits."""
        body = (
            f"## ClawCodex: retry rate limit reached\n\n"
            f"This issue has been retried {current} times "
            f"(limit: {max_retries}). The `agent:retry` label "
            f"is being ignored. Please review manually and "
            f"either remove the label or use "
            f"`clawcodex orchestrator issue retry --id {issue_id} "
            f"--mode reset --force` to bypass.\n"
        )
        try:
            await self.tracker.create_comment(issue_id, body)
        except Exception as exc:
            logger.warning(
                "Failed to post retry-rejection comment for %s: %s",
                issue_id,
                exc,
            )
        # Adding the rejection label is platform-specific. We use
        # `update_issue_state` as a no-op state-setter and try to
        # pass the label through the same channel; the adapter
        # implementations that support labels will route it.
        try:
            update_labels = getattr(self.tracker, "add_label", None)
            if callable(update_labels):
                result = update_labels(issue_id, "agent:retry-rejected")
                if hasattr(result, "__await__"):
                    await result
        except Exception as exc:
            logger.warning(
                "Failed to add agent:retry-rejected label to %s: %s",
                issue_id,
                exc,
            )

    def _log_audit_event(
        self,
        *,
        issue_id: str,
        event: str,
        mode: str,
        reason: str,
        author: str,
    ) -> None:
        """Write a daemon-side audit log entry.

        Best-effort: writes to `~/.clawcodex/orchestrator/audit.jsonl`
        (the same file the CLI uses). Failure to write is logged
        but does not affect the orchestrator's main loop.
        """
        try:
            log_path = Path.home() / ".clawcodex" / "orchestrator" / "audit.jsonl"
            payload = {
                "ts": time.time(),
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "operator": author,
                "issue_id": issue_id,
                "mode": mode,
                "reason": reason,
                "event": event,
                "force": False,
                "priority": "high",
            }
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "Failed to write daemon audit log: %s",
                exc,
            )
