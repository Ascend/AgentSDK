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

# pylint: disable=too-many-nested-blocks

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


from .agent_runner import AgentSession
from .debug_log import append_debug_event
from .events import EventLevel
from .git_sync import (
    GitSyncPostCommitError,
    HookFailedError,
    VerificationFailed,
)
from .premise_check import format_cannot_proceed_comment, read_cannot_proceed
from .prompt_builder import PromptBuilder
from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
    get_file_status,
    get_repo_root,
)
from .tracker import (
    PullRequestFeedbackCapability,
    supports,
)

if TYPE_CHECKING:
    pass

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


class OrchestratorRunMixin:
    async def _run_issue(self, session: AgentSession) -> None:
        """Run agent for one issue with concurrency control."""
        async with self._semaphore:
            ran_agent = False
            workspace_dirty: bool | None = None
            try:
                await self.workspace.run_before_run_hook(
                    session.workspace,
                    session.issue,
                )
                ran_agent = True
                try:
                    # Build a fresh per-session progress sink so
                    # concurrent issues no longer share the
                    # ``_current_task_id`` / ``_phase_count`` mutable
                    # state of the legacy :class:`ProgressReporter`
                    # singleton. ``AgentRunner.run`` is duck-typed on
                    # the kwarg: anything with ``on_phase_complete`` /
                    # ``on_turn_complete`` / ``on_session_complete``
                    # methods works.
                    progress_sink = self._build_session_sink(session.issue.id or "")

                    # Repro-first gate: before any fix work, a dedicated
                    # reproduction pass must demonstrate the described
                    # failure (executable check, non-zero exit). A closed
                    # gate fails the issue with a "cannot reproduce"
                    # report instead of an unverifiable fix MR.
                    if self._repro_gate_applies(session):
                        gate_open = await self._run_repro_gate(session, progress_sink)
                        if not gate_open:
                            return

                    # If workflow.yaml is configured, use the declarative workflow engine
                    # review_followup uses a dedicated prompt (render_review_feedback),
                    # not the full workflow.yaml stage flow, to avoid loops.
                    if self._workflow_orchestrator is not None and session.run_kind != "review_followup":
                        await self._run_issue_with_workflow(session, progress_sink)
                    else:
                        # Collaboration-mode dispatch. For the
                        # default ``single`` mode (the only one
                        # registered in Phase 1) we keep the legacy
                        # ``stage_runners[run_kind] or agent_runner``
                        # lookup so 270+ existing tests pass byte-
                        # identically. For non-single modes registered
                        # in later phases, we dispatch to the
                        # ``ModeRunner`` from the registry instead, and
                        runner = self._resolve_session_runner(session)
                        run_timeout_seconds = self.workflow.agent.run_timeout_ms / 1000.0
                        session.timeout_deadline_at = time.time() + run_timeout_seconds
                        await asyncio.wait_for(
                            runner.run(
                                session,
                                self.workflow,
                                status_dashboard=self.status_dashboard,
                                tracker=self.tracker,
                                comment_tracker=self.tracker,
                                clarification_resolver=self._clarification_resolver,
                                progress_reporter=progress_sink,
                                diagnostics_callback=self._update_run_diagnostics,
                            ),
                            timeout=run_timeout_seconds,
                        )
                    if session.status in (
                        "completed",
                        "stagnation",
                        "read_only_loop",
                        "loop_detected",
                        "max_turns_exceeded",
                    ):
                        # Honest-exit channel (defect R3): the agent declared
                        # the issue premise unfulfillable (e.g. it references
                        # a file that does not exist). Report the finding back
                        # to the issue and mark FAILED instead of falling
                        # through to git_sync — which would either open an MR
                        # around a fabricated fix or an empty branch.
                        _cannot = read_cannot_proceed(getattr(session.workspace, "path", None))
                        if _cannot is not None:
                            _reason = str(_cannot.get("reason", "cannot_proceed"))
                            session.status = "failed"
                            session.session_end_reason = "premise_not_met"
                            session.session_end_summary = str(_cannot.get("details", ""))[:500]
                            logger.warning(
                                "Issue %s: agent declared cannot_proceed (%s) — marking FAILED without creating a PR",
                                session.issue.id,
                                _reason,
                            )
                            self._registry.mark_failed_with_reason(
                                session.issue.id or "",
                                f"premise_not_met ({_reason}): agent declared the issue "
                                "cannot honestly be completed; no PR created.",
                            )
                            try:
                                await self.tracker.create_comment(
                                    session.issue.id or "",
                                    format_cannot_proceed_comment(session.issue, _cannot),
                                )
                            except Exception:
                                logger.warning(
                                    "Issue %s: failed to post cannot_proceed comment",
                                    session.issue.id,
                                    exc_info=True,
                                )
                            await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                            self.status_dashboard.on_session_complete(session.issue.id or "")
                            self._state.completed.add(session.issue.id or "")
                            self._state.failed.add(session.issue.id or "")
                            return
                        # Safety net: verify workspace has actual changes before git_sync.
                        # If agent reported "completed" but workspace is clean (no uncommitted
                        # changes, no HEAD change), mark as failed to avoid empty PRs.
                        if session.status == "completed" and session.session_end_reason not in (
                            "noop_completed",
                            "already_completed",
                            "task_complete",
                        ):
                            _has_changes = False
                            try:
                                _repo_root = get_repo_root(str(session.workspace.path))
                                if _repo_root:
                                    _file_status = get_file_status(_repo_root)
                                    _has_changes = bool(_file_status)
                                    if not _has_changes:
                                        _start_sha = getattr(session, "start_commit_sha", None)
                                        if _start_sha:
                                            from src.utils.git import _run_git as _git

                                            _head_out, _, _rc = _git(["rev-parse", "HEAD"], _repo_root)
                                            _has_changes = bool(
                                                _rc == 0 and _head_out.strip() and _head_out.strip() != _start_sha
                                            )
                            except Exception:
                                _has_changes = True  # fail-open
                            if not _has_changes:
                                logger.warning(
                                    "Session completed but workspace has no changes issue_id=%s — marking as failed",
                                    session.issue.id,
                                )
                                session.status = "failed"
                                session.session_end_reason = "no_changes_produced"
                                session.session_end_summary = (
                                    "Agent reported completed but workspace has no file changes"
                                )
                        # A followup run passes mode="followup"
                        # to git_sync so it reuses the existing branch + PR
                        # instead of creating a new one.
                        sync_mode = (
                            "followup"
                            if session.run_kind in ("agent_followup", "review_followup", "review_retry")
                            and not isinstance(
                                self.tracker,
                                __import__(
                                    "extensions.orchestrator.local_tracker.adapter",
                                    fromlist=["LocalTrackerAdapter"],
                                ).LocalTrackerAdapter,
                            )
                            else "default"
                        )
                        sync_result = await self.git_sync.sync(session, mode=sync_mode)
                        # Addendum: when the daemon triggers a read-only loop /
                        # stagnation termination, git_sync does not create a PR,
                        # and marks empty_branch_no_commits in session_end_reason.
                        # In that case do not mark_synced (would set SYNCED with no PR),
                        # instead mark_failed_with_reason so the issue goes to FAILED.
                        if sync_result is not None and sync_result.session_end_reason == "empty_branch_no_commits":
                            logger.warning(
                                "Issue %s ended with no reviewable commit "
                                "(session_end_reason=%s) — marking FAILED "
                                "without creating a PR",
                                session.issue.id,
                                sync_result.session_end_reason,
                            )
                            session.status = "failed"
                            session.session_end_reason = "empty_branch_no_commits"
                            session.session_end_summary = (
                                "Agent did not produce any file modifications; no PR was created."
                            )
                            session.verification_status = "failed"
                            session.verification_output = session.session_end_summary
                            session.last_hook_error = session.session_end_summary
                            return
                        if sync_result is not None:
                            self._registry.update_report(
                                session.issue.id or "",
                                report_path=getattr(session, "report_path", None),
                                verification_status=getattr(session, "verification_status", None),
                                verification_output=getattr(session, "verification_output", None),
                                summary_comment_id=getattr(session, "summary_comment_id", None),
                                # Root-cause fix: persist
                                # explicit session-end reason so the
                                # dashboard / verification can
                                # distinguish stagnation / loop from
                                # a clean success path.
                                session_end_reason=getattr(session, "session_end_reason", None),
                                session_end_summary=getattr(session, "session_end_summary", ""),
                            )
                            if session.run_kind == "review_followup":
                                self._registry.mark_feedback_processed(
                                    session.issue.id or "",
                                    list(getattr(session, "feedback_ids", [])),
                                    commit_sha=sync_result.commit_sha,
                                )
                                await self._reply_to_processed_feedback(session)
                                await self._post_feedback_summary(session, sync_result)
                                await self._apply_review_rules(session)
                            elif session.run_kind in ("agent_followup", "review_retry"):
                                # A follow-up keeps the
                                # existing pr_number / pr_url / status;
                                # only the followup_attempt_count and
                                # last_followup_commit_sha change.
                                self._registry.increment_followup_attempt(session.issue.id or "")
                                if sync_result.commit_sha:
                                    record = self._registry.get(session.issue.id or "")
                                    if record is not None:
                                        record.last_followup_commit_sha = sync_result.commit_sha
                                        self._registry._save()
                                    if session.run_kind == "review_retry":
                                        # Keep rejected-review feedback
                                        # available across failed attempts,
                                        # but consume it once a follow-up
                                        # commit has synced so a future reset
                                        # cannot replay stale advice.
                                        self._clarification_queue.consume_feedback(session.issue.id or "")
                                logger.info(
                                    "Issue %s followup committed: %s on %s",
                                    session.issue.id,
                                    sync_result.commit_sha,
                                    sync_result.branch_name,
                                )
                            else:
                                self._registry.mark_synced(
                                    session.issue.id or "",
                                    branch_name=sync_result.branch_name,
                                    commit_sha=sync_result.commit_sha,
                                    pr_number=sync_result.pull_request.number if sync_result.pull_request else None,
                                    pr_url=sync_result.pull_request.url if sync_result.pull_request else None,
                                )
                            pr_url = sync_result.pull_request.url if sync_result.pull_request is not None else None
                            if pr_url:
                                is_followup = session.run_kind in (
                                    "agent_followup",
                                    "review_followup",
                                    "review_retry",
                                )
                                self._emit_im_event(
                                    session.issue.id or "",
                                    "pr.updated" if is_followup else "pr.opened",
                                    EventLevel.INFO,
                                    "PR updated" if is_followup else "PR opened",
                                    self._session_payload(
                                        session,
                                        pr=pr_url,
                                        commit=getattr(sync_result, "commit_sha", None),
                                    ),
                                )
                            # Review gate: after commit, await human review before completion.
                            # Triggered when GitSyncResult.pending_review is True (LocalTracker
                            # by default, or any tracker when agent.review_required=True in workflow).
                            if sync_result.pending_review:
                                if self.workflow.agent.auto_approve:
                                    logger.info(
                                        "Issue %s auto-approved (auto_approve=True) — skipping pending_review gate",
                                        session.issue.id,
                                    )
                                else:
                                    self._registry.mark_pending_review(session.issue.id or "")
                                    await self._sync_tracker_issue_state(session.issue.id or "", "pending_review")
                                    self.status_dashboard.on_session_complete(session.issue.id or "")
                                    self._emit_im_event(
                                        session.issue.id or "",
                                        "pr.pending_review_gate",
                                        EventLevel.WARN,
                                        "pending human review",
                                        self._session_payload(session, pr=pr_url),
                                    )
                                    self._state.pending_review.add(session.issue.id or "")
                                    # Do NOT cleanup workspace — human needs to review it
                                    return

                        # ClawCodex downstream-deviation (TODO upstream-merge):
                        # salvage override — when the widened gate above let
                        # us attempt git_sync for a non-completed agent
                        # termination, but the sync actually produced a real
                        # commit + PR, treat the run as a successful salvage:
                        # override session.status to "completed" and record
                        # the actual termination reason in
                        # session_end_reason / session_end_summary so the
                        # audit trail is preserved. Without this, the
                        # post-`_run_issue` failure handler would still see
                        # status=stagnation/loop_detected/etc and route the
                        # run to retry/abandoned even though the work landed.
                        if session.status != "completed" and sync_result is not None and sync_result.commit_sha:
                            logger.warning(
                                "Issue %s session terminated with status=%s "
                                "but git_sync salvaged commit %s on branch "
                                "%s — overriding status to completed and "
                                "recording salvage reason",
                                session.issue.id,
                                session.status,
                                sync_result.commit_sha,
                                sync_result.branch_name,
                            )
                            session.session_end_reason = f"salvaged_after_{session.status}"
                            session.session_end_summary = (
                                f"agent terminated with status="
                                f"{session.status}; git_sync salvaged "
                                f"commit {sync_result.commit_sha[:12]} on "
                                f"branch {sync_result.branch_name}"
                            )
                            session.status = "completed"
                finally:
                    await self.workspace.run_after_run_hook(
                        session.workspace,
                        session.issue,
                    )
            except GitSyncPostCommitError as exc:
                sync_result = exc.result
                self._registry.update_report(
                    session.issue.id or "",
                    report_path=getattr(session, "report_path", None),
                    verification_status=getattr(session, "verification_status", None),
                    verification_output=getattr(session, "verification_output", None),
                    summary_comment_id=getattr(session, "summary_comment_id", None),
                    session_end_reason=getattr(session, "session_end_reason", None),
                    session_end_summary=getattr(session, "session_end_summary", ""),
                )
                if session.run_kind in ("agent_followup", "review_retry"):
                    record = self._registry.get(session.issue.id or "")
                    if record is not None and sync_result.commit_sha:
                        record.last_followup_commit_sha = sync_result.commit_sha
                        self._registry._save()
                elif session.run_kind != "review_followup":
                    self._registry.mark_synced(
                        session.issue.id or "",
                        branch_name=sync_result.branch_name,
                        commit_sha=sync_result.commit_sha,
                        pr_number=(sync_result.pull_request.number if sync_result.pull_request else None),
                        pr_url=(sync_result.pull_request.url if sync_result.pull_request else None),
                    )
                logger.warning(
                    "Post-commit sync failed issue_id=%s commit=%s: %s",
                    session.issue.id,
                    sync_result.commit_sha,
                    exc,
                )
                session.status = "verification_failed"
                session.verification_status = "failed"
                session.verification_output = exc.output
                if exc.hook_name:
                    session.last_hook_error = str(exc.cause)
                self._emit_im_event(
                    session.issue.id or "",
                    "post_commit_failed",
                    EventLevel.ERROR,
                    str(exc),
                    self._session_payload(
                        session,
                        pr=sync_result.pull_request.url if sync_result.pull_request is not None else None,
                        commit=getattr(sync_result, "commit_sha", None),
                    ),
                )
            except VerificationFailed as exc:
                logger.warning(
                    "Verification failed issue_id=%s: %s",
                    session.issue.id,
                    exc,
                )
                session.status = "verification_failed"
                session.verification_status = "failed"
                session.verification_output = exc.output
                self._emit_im_event(
                    session.issue.id or "",
                    "verification.failed",
                    EventLevel.WARN,
                    exc.output or str(exc),
                    self._session_payload(session),
                )
            except HookFailedError as exc:
                logger.warning(
                    "Hook failed issue_id=%s hook=%s: %s",
                    session.issue.id,
                    exc.hook_name,
                    exc,
                )
                session.status = "verification_failed"
                session.verification_status = "failed"
                session.verification_output = exc.output
                session.last_hook_error = str(exc)
                self._emit_im_event(
                    session.issue.id or "",
                    "verification.failed",
                    EventLevel.WARN,
                    f"{exc.hook_name}: {exc.output or exc}",
                    self._session_payload(session),
                )
            except asyncio.TimeoutError:
                reason = f"Agent run exceeded configured timeout ({self.workflow.agent.run_timeout_ms}ms)"
                logger.warning(
                    "Agent run timed out issue_id=%s timeout_ms=%s",
                    session.issue.id,
                    self.workflow.agent.run_timeout_ms,
                )
                workspace_dirty = bool(get_file_status(str(session.workspace.path)))
                append_debug_event(
                    getattr(session, "debug_log_path", None),
                    "orchestrator.timeout",
                    run_id=getattr(session, "run_id", None),
                    turn_count=getattr(session, "turn_count", 0),
                    tool_count=getattr(session, "tool_count", 0),
                    last_event_type=getattr(session, "last_agent_event", None),
                    last_tool=getattr(session, "last_tool_name", None),
                    output_len=len(getattr(session, "output_text", "") or ""),
                    workspace_dirty=workspace_dirty,
                    timeout_ms=self.workflow.agent.run_timeout_ms,
                )
                session.status = "agent_timeout"
                session.verification_status = "failed"
                session.verification_output = reason
                self._emit_im_event(
                    session.issue.id or "",
                    "issue.failed",
                    EventLevel.WARN,
                    reason,
                    self._session_payload(
                        session,
                        turns=getattr(session, "turn_count", None),
                    ),
                )
            except asyncio.CancelledError:
                # Root-cause fix: clean cancellation path.
                # When the stop command cancels the task, capture
                # the reason so the registry marks the issue as
                # cancelled instead of silently dropping it.
                # Also clean up the workspace immediately to avoid
                # leaking worktrees on unexpected cancellation.
                logger.warning(
                    "Agent run cancelled issue_id=%s — cleaning up workspace",
                    session.issue.id,
                )
                session.status = "cancelled"
                session.session_end_reason = "operator_stopped"
                session.session_end_summary = "cancelled by operator"
                session.verification_status = "cancelled"
                session.verification_output = "Operator requested stop"
                # Best-effort workspace cleanup on cancellation so
                # worktrees are not left dirty even if the outer
                # finally block is skipped or interrupted.
                try:
                    issue_record = self._registry.get(session.issue.id)
                    await self.workspace.cleanup(
                        session.issue,
                        end_status=session.status,
                        end_reason=session.session_end_reason,
                        agent_config=getattr(self, "_agent_config", None),
                        issue_record=issue_record,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Workspace cleanup on cancellation failed issue_id=%s: %s",
                        session.issue.id,
                        cleanup_exc,
                    )
            except Exception as exc:
                logger.exception(
                    "Agent run failed issue_id=%s: %s",
                    session.issue.id,
                    exc,
                )
                session.status = "before_run_failed" if not ran_agent else "failed"
                # Replace any prior success summary with the actual failure
                # detail so IM and registry records show the root cause.
                detail = _operator_failure_detail(exc)
                session.session_end_reason = session.status
                session.session_end_summary = detail
                session.verification_status = "failed"
                session.verification_output = detail
                session.last_hook_error = detail
                setattr(session, "operator_failure_detail", detail)
            finally:
                if workspace_dirty is not None:
                    session.run_workspace_dirty = workspace_dirty
                self._update_run_diagnostics(session)
                # Diagnostics saves are throttled; force the final
                # snapshot to disk in case this path (e.g. pending_review)
                # ends without a durable status mutation.
                self._registry.flush()

                if session.issue.id in self._state.running:
                    del self._state.running[session.issue.id]

                # Dashboard journal: one terminal event per run with the
                # final status plus the session/PR references the issue
                # accumulated. Best-effort — never raises.
                if self._viz_journal is not None:
                    try:
                        _iid = str(session.issue.id or "")
                        _rec = self._registry.get(_iid)
                        if getattr(session, "run_id", None):
                            self._viz_journal.write_event(
                                {
                                    "type": "session_ref",
                                    "issue_id": _iid,
                                    "session_id": str(session.run_id),
                                    "session_path": str(Path.home() / ".clawcodex" / "sessions" / str(session.run_id)),
                                }
                            )
                        if _rec is not None and _rec.pr_url:
                            self._viz_journal.write_event(
                                {
                                    "type": "pr_status",
                                    "issue_id": _iid,
                                    "pr_url": _rec.pr_url,
                                    "pr_number": _rec.pr_number,
                                }
                            )
                        _status = str(session.status or "")
                        if _status == "completed":
                            self._viz_journal.write_event(
                                {
                                    "type": "complete",
                                    "issue_id": _iid,
                                    "overall_status": "completed",
                                }
                            )
                        elif _status:
                            self._viz_journal.write_event(
                                {
                                    "type": "error",
                                    "issue_id": _iid,
                                    "error": getattr(session, "session_end_summary", "") or _status,
                                }
                            )
                    except Exception:
                        logger.debug("viz journal final event failed", exc_info=True)

                # Review gate: if the issue is already in pending_review
                # (set by the early return above), skip the final status
                # transition so the outer finally does NOT overwrite it with
                # COMPLETED. The human must run `orchestrator issue review
                # --id ... --approve` to move it to COMPLETED.
                if session.issue.id in self._state.pending_review:
                    # Issue is waiting for human review — do nothing further.
                    # Workspace preservation is handled by the early return.
                    logger.info(
                        "Issue %s left in pending_review state — human review required",
                        session.issue.id,
                    )
                elif session.status == "completed":
                    self.status_dashboard.on_session_complete(session.issue.id or "")
                    self._emit_im_event(
                        session.issue.id or "",
                        "issue.completed",
                        EventLevel.SUCCESS,
                        "任务完成",
                        self._session_payload(session),
                    )
                    self._state.completed.add(session.issue.id or "")
                    self._registry.mark_completed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "completed")
                elif session.status == "verification_failed":
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        str(session.status),
                    )
                    # Terminal IM event already emitted by the originating
                    # except handler (VerificationFailed / HookFailedError /
                    # GitSyncPostCommitError). ``verification_failed`` is
                    # only ever set there, so re-emitting here would double
                    # (e.g. ``post_commit_failed`` ERROR then
                    # ``verification.failed`` WARN). See review 🟡2.
                    self._registry.mark_verification_failed(
                        session.issue.id or "",
                        output=getattr(session, "verification_output", None),
                        hook_error=getattr(session, "last_hook_error", None),
                    )
                    await self._sync_tracker_issue_state(session.issue.id or "", "verification_failed")
                    await self._schedule_retry(session)
                elif session.status == "agent_timeout":
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        str(session.status),
                    )
                    # Terminal IM event already emitted by the
                    # ``asyncio.TimeoutError`` except handler; ``agent_timeout``
                    # is only ever set there. See review 🟡2.
                    self._registry.mark_failed_with_reason(
                        session.issue.id or "",
                        getattr(session, "last_hook_error", None)
                        or getattr(session, "verification_output", None)
                        or "Agent run timed out",
                    )
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    await self._schedule_retry(session)
                elif session.status == "max_turns_exceeded":
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        str(session.status),
                    )
                    self._emit_im_event(
                        session.issue.id or "",
                        "agent.max_turns_exceeded",
                        EventLevel.WARN,
                        "max turns exceeded",
                    )
                    self._registry.mark_failed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    await self._schedule_retry(
                        session,
                        delay_base_ms=self.workflow.agent.max_turns_retry_delay_ms,
                    )
                elif session.status == "rate_limit_circuit_open":
                    # The AgentRunner's 429 backoff circuit breaker tripped
                    # after ``rate_limit_max_retries`` consecutive rate
                    # limit hits. Surface it on the dashboard and hand it
                    # off to the inter-run retry queue with the longest
                    # configured base delay so the provider's rate window
                    # has a chance to reset before the next attempt.
                    backoff_s = self.workflow.agent.rate_limit_max_backoff_ms
                    logger.warning(
                        "Rate limit circuit open issue_id=%s — scheduling "
                        "inter-run retry with base delay %dms (session "
                        "spent %.1fs in in-turn backoff across %d hits)",
                        session.issue.id or "",
                        backoff_s,
                        getattr(session, "total_429_backoff_seconds", 0.0),
                        getattr(session, "consecutive_429_count", 0),
                    )
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        "rate_limit_circuit_open",
                    )
                    self._emit_im_event(
                        session.issue.id or "",
                        "agent.rate_limit_circuit_open",
                        EventLevel.ERROR,
                        "rate limit circuit open",
                    )
                    self._registry.mark_failed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    await self._schedule_retry(
                        session,
                        delay_base_ms=backoff_s,
                    )
                elif session.status in (
                    "stagnation",
                    "loop_detected",
                ):
                    # Root-cause fix: the agent loop detected it
                    # was no longer making progress (stagnation =
                    # consecutive no-op turns; loop_detected = same
                    # tool-call signature repeated within window).
                    # Mark the issue failed with the explicit
                    # session_end_reason so the dashboard / cron tick
                    # can distinguish these from ordinary crashes.
                    logger.warning(
                        "Agent %s issue_id=%s — %s",
                        session.status,
                        session.issue.id or "",
                        getattr(session, "session_end_summary", ""),
                    )
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        str(session.status),
                    )
                    self._emit_im_event(
                        session.issue.id or "",
                        f"agent.{session.status}",
                        EventLevel.WARN,
                        getattr(session, "session_end_summary", "") or str(session.status),
                    )
                    self._registry.mark_failed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    # No retry — same agent will likely repeat the
                    # same loop on retry without human intervention.
                    # The cron tick will mark the issue abandoned on
                    # the next pass and the operator can either
                    # adjust the issue / workflow or skip it.
                elif session.status == "cancelled":
                    logger.info(
                        "Issue %s cancelled by operator — skipping retry",
                        session.issue.id,
                    )
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        "cancelled",
                    )
                    self._emit_im_event(
                        session.issue.id or "",
                        "issue.cancelled",
                        EventLevel.WARN,
                        "cancelled by operator",
                    )
                    self._registry.mark_failed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    # Do NOT schedule retry — operator explicitly cancelled.
                else:
                    self.status_dashboard.on_session_failed(
                        session.issue.id or "",
                        str(session.status),
                    )
                    # Use the error detail (session_end_summary) as the
                    # message if available — str(session.status) is just
                    # "failed" with no context.  Truncate long error
                    # bodies (e.g. API JSON responses) for WeChat display.
                    detail = getattr(session, "session_end_summary", None) or str(session.status)
                    if len(detail) > 200:
                        detail = detail[:200] + "…"
                    self._emit_im_event(
                        session.issue.id or "",
                        "issue.failed",
                        EventLevel.WARN,
                        detail,
                        self._session_payload(
                            session,
                            turns=getattr(session, "turn_count", None),
                        ),
                    )
                    failure_detail = getattr(session, "operator_failure_detail", None)
                    if failure_detail:
                        self._registry.mark_failed_with_reason(
                            session.issue.id or "",
                            str(failure_detail),
                        )
                        self._registry.update_report(
                            session.issue.id or "",
                            session_end_reason=getattr(session, "session_end_reason", None),
                            session_end_summary=getattr(session, "session_end_summary", ""),
                        )
                    else:
                        self._registry.mark_failed(session.issue.id or "")
                    await self._sync_tracker_issue_state(session.issue.id or "", "failed")
                    # Schedule retry
                    await self._schedule_retry(session)

                # Update summary comment for non-completed paths
                if session.issue.id not in self._state.pending_review:
                    await self._update_issue_summary(session)

                # Cleanup workspace based on preservation policy
                try:
                    issue_record = self._registry.get(session.issue.id)
                    await self.workspace.cleanup(
                        session.issue,
                        end_status=getattr(session, "status", None),
                        end_reason=getattr(session, "session_end_reason", None),
                        agent_config=getattr(self, "_agent_config", None),
                        issue_record=issue_record,
                    )
                except Exception as exc:
                    logger.warning(
                        "Workspace cleanup failed issue_id=%s: %s",
                        session.issue.id,
                        exc,
                    )

                self._state.claimed.discard(session.issue.id or "")

    async def _update_issue_summary(self, session: AgentSession) -> None:
        """Update the issue summary comment with final status for failure paths."""
        comment_id = getattr(session, "summary_comment_id", None)
        if comment_id is None:
            return
        body_lines = [
            "## ClawCodex Run Summary",
            "",
            f"- Run: `{getattr(session, 'run_id', 'unknown')}`",
            f"- Status: `{getattr(session, 'status', 'unknown')}`",
            f"- Turns: {getattr(session, 'turn_count', 0)}",
            f"- Tool calls: {getattr(session, 'tool_count', 0)}",
        ]
        if getattr(session, "last_hook_error", None):
            body_lines.append(f"- Error: `{session.last_hook_error}`")
        body = "\n".join(body_lines)
        try:
            await self.tracker.update_comment(session.issue.id, comment_id, body)
        except Exception as exc:
            logger.warning("Failed to update summary comment issue_id=%s: %s", session.issue.id, exc)

    async def _apply_review_rules(self, session: AgentSession) -> None:
        """Ensure the review commit carries review metadata.

        Rule extraction has been removed from the follow-up pipeline and is
        now triggered manually via the CLI command ``clawcodex rules extract``.
        Review metadata (review-pr / review-id) is written into the commit
        message by ``GitSyncService``, which the CLI extract command parses
        when scanning the commit log.
        """
        pass

    async def _reply_to_processed_feedback(self, session: AgentSession) -> None:
        if not self.workflow.review_feedback.reply_to_comments:
            return
        pull_request = getattr(session, "pull_request", None)
        feedback_ids = set(getattr(session, "feedback_ids", []))
        if pull_request is None or not feedback_ids:
            return
        if not supports(self.tracker, PullRequestFeedbackCapability):
            return
        try:
            feedback = await self.tracker.fetch_pull_request_feedback(
                pull_request=pull_request,
                issue_id=session.issue.id,
                include_ci_failures=False,
            )
        except Exception as exc:
            logger.warning("Failed to refresh feedback for replies issue_id=%s: %s", session.issue.id, exc)
            return
        from .review_feedback import REPLY_MARKER

        body = REPLY_MARKER
        for item in feedback:
            if item.id not in feedback_ids:
                continue
            try:
                await self.tracker.reply_to_pull_request_feedback(
                    pull_request=pull_request,
                    feedback=item,
                    body=body,
                    issue_id=session.issue.id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to reply to PR feedback issue_id=%s feedback_id=%s: %s",
                    session.issue.id,
                    item.id,
                    exc,
                )

    async def _post_feedback_summary(self, session: AgentSession, sync_result: Any) -> None:
        """Post a processing summary comment to the PR after a review follow-up."""
        pull_request = getattr(session, "pull_request", None)
        feedback_ids = list(getattr(session, "feedback_ids", []))
        if pull_request is None or not feedback_ids:
            return
        record = self._registry.get(session.issue.id or "")
        attempt = record.followup_attempt_count if record else 1

        if not supports(self.tracker, PullRequestFeedbackCapability):
            return
        try:
            all_feedback = await self.tracker.fetch_pull_request_feedback(
                pull_request=pull_request,
                issue_id=session.issue.id,
                include_ci_failures=self.workflow.review_feedback.include_ci_failures,
            )
        except Exception as exc:
            logger.warning("Failed to fetch feedback for summary issue_id=%s: %s", session.issue.id, exc)
            all_feedback = []

        feedback_by_id = {item.id: item for item in all_feedback}
        processed = []
        skipped = []
        commit_sha = getattr(sync_result, "commit_sha", None)
        for fid in feedback_ids:
            fb = feedback_by_id.get(fid)
            if fb is None:
                continue
            if commit_sha:
                processed.append(fb)
            else:
                skipped.append({"feedback": fb, "reason": "No changes were committed"})

        summary = PromptBuilder.render_feedback_summary(
            attempt=attempt,
            processed=processed,
            skipped=skipped,
        )
        try:
            await self.tracker.create_comment(session.issue.id or "", summary)
        except Exception as exc:
            logger.warning("Failed to post feedback summary issue_id=%s: %s", session.issue.id, exc)
