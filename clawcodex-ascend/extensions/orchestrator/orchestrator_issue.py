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
import json
import logging
import time
from typing import TYPE_CHECKING, Any


from .agent_runner import AgentSession
from .issue import Issue
from . import modes as _modes
from .modes.base import DEFAULT_MODE, ModeDecision
from .repro_gate import (
    ReproGateResult,
    append_repro_hint,
    build_repro_prompt,
    evaluate_repro_gate,
    format_repro_gate_comment,
)
from .status_dashboard import SessionStatus
from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
    get_default_branch,
)
from .tracker import (
    Intent,
    PullRequestCapability,
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


class OrchestratorIssueMixin:
    async def _launch_issue(self, issue: Issue) -> None:
        """Create workspace and run agent for one issue."""
        if not await self._dependencies_satisfied(issue):
            self._state.claimed.discard(issue.id)
            return

        # If the registry carries a RETRY intent for this
        # issue, close the existing remote PR (best-effort) and reset
        # the local record so the new run starts from a clean slate.
        # This must happen BEFORE workspace creation so the new run
        # does not try to push a follow-up commit to a closed PR.
        await self._prepare_intent_reset(issue)

        workspace_strategy = self.workflow.workspace.strategy
        branch_name = getattr(issue, "branch_name", None)
        if not branch_name:
            branch_name = self.git_sync._default_branch_name(issue)
            issue.branch_name = branch_name

        try:
            workspace = await self.workspace.create_for_issue(issue)
        except Exception as exc:
            logger.error(
                "Workspace creation failed issue_id=%s: %s",
                issue.id,
                exc,
            )
            self._state.claimed.discard(issue.id)
            return

        # Register as pending so restart won't re-launch this issue
        base_branch = getattr(issue, "base_branch", None) or self.workflow.workspace.base_branch or "main"
        integration_branch = self.workflow.workspace.integration_branch
        if workspace_strategy == "sequential" and integration_branch:
            branch_name = integration_branch
        start_commit_sha = await self.workspace.current_head(workspace.path)
        base_commit_sha = start_commit_sha if workspace_strategy == "sequential" else None
        previous_issue_id = None
        sequence_index = None
        if workspace_strategy == "sequential":
            previous_record = self._registry.latest_sequential_record()
            previous_issue_id = previous_record.issue_id if previous_record else None
            sequence_index = (previous_record.sequence_index or 0) + 1 if previous_record else 1
        # In sequential mode the registry's workspace_path must
        # record the configured root (not whatever WorkspaceManager
        # happened to return for the current issue), so that subsequent
        # issues can resolve the previous commit chain against the same
        # path. In isolated / shared modes the per-issue workspace.path
        # is already the canonical location, so keep that.
        recorded_workspace_path = (
            str(self._workspace_root) if workspace_strategy == "sequential" else str(workspace.path)
        )
        self._registry.register(
            issue_id=issue.id or "",
            issue_identifier=issue.identifier or "",
            branch_name=branch_name,
            base_branch=base_branch,
            workspace_strategy=workspace_strategy,
            workspace_path=recorded_workspace_path,
            base_commit_sha=base_commit_sha,
            start_commit_sha=start_commit_sha,
            previous_issue_id=previous_issue_id,
            sequence_index=sequence_index,
            author_login=issue.author_login,
        )

        # Pre-check: verify issue is still in an active state and has no
        # existing PR (which would mean it was already handled) before running agent
        try:
            refreshed = await self.tracker.fetch_issue_states_by_ids([issue.id])
            refreshed_issue = refreshed.get(issue.id)
            if refreshed_issue is None:
                logger.info("Issue %s no longer exists, skipping", issue.id)
                self._state.claimed.discard(issue.id)
                return
            active_states = [s.strip().lower() for s in (getattr(self.tracker, "active_states", None) or [])]
            is_active = refreshed_issue.state is not None and refreshed_issue.state.strip().lower() in active_states
            if not is_active:
                logger.info(
                    "Issue %s is no longer active (state=%r), skipping",
                    issue.id,
                    refreshed_issue.state,
                )
                self._state.claimed.discard(issue.id)
                return
            # Check for existing PR (only for repository-backed trackers)
            branch_name = refreshed_issue.branch_name
            if branch_name and supports(self.tracker, PullRequestCapability):
                base_branch = getattr(refreshed_issue, "base_branch", "main") or "main"
                existing_pr = await self.tracker.find_pull_request(
                    head_branch=branch_name,
                    base_branch=base_branch,
                )
                if existing_pr is not None:
                    # Explicit follow-up and retry intents both bypass the
                    # ordinary existing-PR guard. Follow-up reuses the PR;
                    # retry already attempted to close it and must still
                    # proceed when that best-effort close was a no-op.
                    record = self._registry.get(issue.id or "")
                    if record and record.intent in (Intent.RETRY, Intent.FOLLOWUP):
                        logger.info(
                            "Issue %s %s intent on existing PR %s (%s), proceeding",
                            issue.id,
                            record.intent.value,
                            existing_pr.number,
                            existing_pr.url,
                        )
                    else:
                        logger.info(
                            "Issue %s already has PR %s (%s), skipping",
                            issue.id,
                            existing_pr.number,
                            existing_pr.url,
                        )
                        self._state.claimed.discard(issue.id)
                        # Also add to completed so we don't re-process after restart
                        self._state.completed.add(issue.id)
                        return

            # Registry-based guard: skip if the local registry already records
            # a PR or a terminal state for this issue.  The tracker-based check
            # above only fires when the issue body contains ``branch_name:``
            # — many issues lack that field, so this tag-team guard across all
            # entry points (poll, retry queue, escalation) catches the gap.
            #
            # The ``register()`` call at line 1217 preserves ``pr_number`` from
            # any previous run (see issue_registry.py:317), while explicit retry
            # intents (``_prepare_intent_reset`` → ``reset_for_retry``) clear it
            # beforehand so a deliberate re-run still passes through.
            if self._registry.has_pr(issue.id or "") or self._registry.is_terminal(issue.id or ""):
                # Explicit retry/follow-up intents deliberately bypass the
                # handled guard. Retry clears stale PR state before reaching
                # this point; follow-up reuses it.
                record = self._registry.get(issue.id or "")
                if record and record.intent in (Intent.RETRY, Intent.FOLLOWUP):
                    logger.info(
                        "Issue %s %s intent bypasses registry guard (has_pr=%s, is_terminal=%s), proceeding",
                        issue.id,
                        record.intent.value,
                        self._registry.has_pr(issue.id or ""),
                        self._registry.is_terminal(issue.id or ""),
                    )
                else:
                    logger.info(
                        "Issue %s already handled (registry: has_pr=%s, "
                        "is_terminal=%s), skipping via _launch_issue guard",
                        issue.id,
                        self._registry.has_pr(issue.id or ""),
                        self._registry.is_terminal(issue.id or ""),
                    )
                    self._state.claimed.discard(issue.id)
                    self._state.completed.add(issue.id)
                    return

            # Update issue with latest state
            issue.state = refreshed_issue.state
        except Exception as exc:
            logger.warning(
                "Could not verify issue state for %s: %s — proceeding anyway",
                issue.id,
                exc,
            )

        session = AgentSession(
            issue=issue,
            workspace=workspace,
            pause_resume_event=asyncio.Event(),
            event_queue=asyncio.Queue(),
        )

        # Wire pause-state notification so the socket path
        # (_drain_control_commands in agent_runner) can sync the
        # registry when pause/resume is processed.
        def _on_pause_change(issue_id: str, paused: bool, reason: str) -> None:
            if paused:
                self._registry.mark_paused(issue_id, reason=reason)
            else:
                self._registry.mark_resumed(issue_id)

        session._on_pause_state_change = _on_pause_change
        clarification_record = self._registry.get(issue.id or "")
        if clarification_record is not None and clarification_record.local_answer:
            session.clarification_answer = clarification_record.local_answer
            session.clarification_source = clarification_record.local_answer_source
            if clarification_record.question_history:
                session.clarification_question = "\n".join(
                    f"- {question}" for question in clarification_record.question_history
                )
        retry_attempt = self._state.retry_attempts.get(issue.id or "", 0)
        session.attempt = retry_attempt + 1
        session.issue_attempt = session.attempt
        session.workspace_strategy = workspace_strategy
        session.workspace_path = str(workspace.path)
        session.start_commit_sha = start_commit_sha
        session.base_commit_sha = base_commit_sha
        session.previous_issue_id = previous_issue_id
        session.sequence_index = sequence_index
        session.integration_branch = integration_branch
        session.base_branch = base_branch
        # Collaboration mode selection. Phase 1 ships only the
        # ``single`` mode; ModeSelector returns "single" unless the issue
        # carries a ``mode:<name>`` label that maps to a registered
        # runner. The decision is recorded on the session for the
        # dispatcher in ``_run_issue`` and on the registry record for
        # audit (`issue list --mode`, dashboard column).
        try:
            mode_decision = self._mode_selector.choose(issue)
        except Exception:
            logger.exception(
                "Issue %s ModeSelector.choose raised; defaulting to single",
                issue.id,
            )
            mode_decision = ModeDecision(
                mode=DEFAULT_MODE,
                reason="ModeSelector.choose raised; see logs",
                source="fallback",
            )
        session.collaboration_mode = mode_decision.mode
        session.mode_decision = mode_decision
        record = self._registry.get(issue.id or "")
        if record is not None:
            record.collaboration_mode = mode_decision.mode
            record.mode_decision_reason = mode_decision.reason
            record.touch()
            self._registry._save()
        logger.info(
            "Issue %s collaboration_mode=%s (source=%s, reason=%s)",
            issue.id,
            mode_decision.mode,
            mode_decision.source,
            mode_decision.reason,
        )
        if self._viz_journal is not None:
            self._viz_journal.write_event(
                {
                    "type": "issue_status",
                    "issue_id": str(issue.id or ""),
                    "status": "running",
                }
            )
            self._viz_journal.write_event(
                {
                    "type": "phase",
                    "issue_id": str(issue.id or ""),
                    "phase": f"mode:{mode_decision.mode}",
                }
            )
        # If the registry intent is FOLLOWUP, wire the
        # session so the agent + git_sync know to reuse the existing
        # branch / PR rather than create a new run.
        self._prepare_intent_session(session)
        # Retry context: propagate previous_run_ids from the registry
        # to the session so the prompt builder can inject them.
        prev_record = self._registry.get(issue.id or "")
        if prev_record and prev_record.previous_run_ids:
            session.previous_run_ids = list(prev_record.previous_run_ids)
        self._state.running[issue.id] = session

        # Update persistent registry so `issue list` reflects running state
        self._registry.mark_running(issue.id or "")

        # Sync .gitignore to workspace so unwanted files are excluded from commit
        self._sync_gitignore_to_workspace(session.workspace)

        self.status_dashboard.on_session_start(
            SessionStatus(
                issue_id=issue.id or "",
                issue_identifier=issue.identifier or "",
                max_turns=self.agent_runner.max_turns,
                workspace_path=str(workspace.path),
            )
        )

        task = asyncio.create_task(self._run_issue(session))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        # Register issue_id → task mapping so the stop command
        # can cancel a specific running issue via task.cancel().
        issue_id_task = issue.id or ""
        self._issue_tasks[issue_id_task] = task

        def _unregister_issue_task(t: asyncio.Task) -> None:
            self._issue_tasks.pop(issue_id_task, None)

        task.add_done_callback(_unregister_issue_task)

    async def _sync_tracker_issue_state(self, issue_id: str, state: str) -> bool:
        if not issue_id:
            return False
        try:
            await self.tracker.update_issue_state(issue_id, state)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to sync tracker state issue_id=%s state=%s: %s",
                issue_id,
                state,
                exc,
            )
            return False

    def _update_run_diagnostics(self, session: AgentSession) -> None:
        issue_id = session.issue.id or ""
        record = self._registry.update_run_diagnostics(
            issue_id,
            run_id=getattr(session, "run_id", None),
            debug_log_path=getattr(session, "debug_log_path", None),
            turn_count=getattr(session, "turn_count", 0),
            tool_count=getattr(session, "tool_count", 0),
            last_event=getattr(session, "last_agent_event", None),
            last_tool=getattr(session, "last_tool_name", None),
            output_len=len(getattr(session, "output_text", "") or ""),
            timeout_deadline_at=getattr(session, "timeout_deadline_at", None),
            workspace_dirty=getattr(session, "run_workspace_dirty", None),
        )
        if record is None:
            if not (issue_id or "").startswith("stage-"):
                logger.warning(
                    "Skipped run diagnostics update because registry record is missing issue_id=%s run_id=%s status=%s",
                    issue_id,
                    getattr(session, "run_id", None),
                    getattr(session, "status", None),
                )

    async def _run_issue_with_workflow(
        self,
        session: AgentSession,
        progress_sink: Any,
    ) -> None:
        """Handle the issue with the declarative workflow engine.

        Executes the issue via WorkflowOrchestrator through the DAG stages
        defined in workflow.yaml, each stage run by a synthetic Issue driven by AgentRunner.
        """
        workflow_orch = self._workflow_orchestrator
        if workflow_orch is None:
            logger.error("_run_issue_with_workflow called but no workflow orchestrator")
            session.status = "failed"
            return

        logger.info(
            "Running workflow for issue %s: %s",
            session.issue.identifier,
            session.issue.title,
        )

        # Ensure the workspace is on the issue branch (not main).
        # A retained workspace may still be on main or the last-run branch,
        # so switch to the correct issue branch before running the workflow.
        try:
            work_branch = self.git_sync._ensure_work_branch(
                str(session.workspace.path),
                session.issue,
                session.base_branch or get_default_branch(str(session.workspace.path)),
            )
            logger.info(
                "Workflow workspace on branch: %s (issue=%s)",
                work_branch,
                session.issue.identifier,
            )
        except Exception as exc:
            logger.warning(
                "Failed to ensure work branch for workflow issue %s: %s",
                session.issue.id,
                exc,
            )

        # Inject the orchestrator's ProgressSink into the workflow engine,
        # so stage progress is reflected live on the StatusDashboard
        workflow_orch.set_progress_sink(progress_sink)
        workflow_orch._stage_runner._progress_reporter = progress_sink

        try:
            result = await workflow_orch.run_for_issue(
                issue=session.issue,
                workspace_path=str(session.workspace.path),
            )
        except Exception as exc:
            logger.exception("Workflow execution failed for issue %s", session.issue.id)
            session.status = "failed"
            session.output_text = str(exc)
            return

        # Store stage output into the session for git_sync to write into the PR body
        session.workflow_stage_outputs = {}
        for stage_id, stage_result in result.stage_results.items():
            if stage_result.outputs:
                session.workflow_stage_outputs[stage_id] = {
                    "phase": getattr(workflow_orch.schema.get_stage(stage_id), "phase", ""),
                    "name": getattr(workflow_orch.schema.get_stage(stage_id), "name", f"Stage {stage_id}"),
                    "output": stage_result.outputs[0] if stage_result.outputs else "",
                }

        if result.success:
            session.status = "completed"
            session.output_text = (
                f"Workflow '{result.workflow_name}' completed: "
                f"{result.completed_stages}/{result.total_stages} stages, "
                f"cost=${result.total_cost_usd:.4f}, "
                f"duration={result.total_duration_seconds:.1f}s"
            )
        else:
            session.status = "failed"
            session.output_text = (
                f"Workflow '{result.workflow_name}' failed at stage "
                f"{result.completed_stages}/{result.total_stages}: {result.error}"
            )

        self._update_run_diagnostics(session)

    def _repro_gate_applies(self, session: AgentSession) -> bool:
        """The gate only fronts fresh issue runs (not retries of other
        run kinds), only when enabled, and — when ``labels`` is
        configured — only for issues carrying one of those labels.
        """
        config = self.workflow.agent.repro_first
        if not config.enabled or session.run_kind != "issue":
            return False
        if config.labels:
            issue_labels = {label.strip().lower() for label in (getattr(session.issue, "labels", None) or [])}
            wanted = {label.strip().lower() for label in config.labels}
            if not issue_labels & wanted:
                return False
        return True

    async def _run_repro_gate(self, session: AgentSession, progress_sink: Any) -> bool:
        """Run the reproduction stage; True means "bug demonstrated,
        proceed to the fix stage".

        On a closed gate the issue is marked FAILED with a
        "cannot reproduce" report posted to the tracker, mirroring the
        empty-branch failure path (no MR is opened).
        """
        issue = session.issue
        config = self.workflow.agent.repro_first
        session.run_kind = "repro"
        session.prompt_override = build_repro_prompt(issue)
        repro_timeout_seconds = config.timeout_ms / 1000.0
        session.timeout_deadline_at = time.time() + repro_timeout_seconds
        logger.info("Issue %s: repro-first gate starting", issue.id)
        timed_out = False
        try:
            await asyncio.wait_for(
                self.agent_runner.run(
                    session,
                    self.workflow,
                    status_dashboard=self.status_dashboard,
                    # The repro stage has its own executable completion
                    # contract below. Passing the tracker here makes the
                    # generic runner continue while the issue is still open,
                    # even after the repro artifacts are complete.
                    tracker=None,
                    comment_tracker=self.tracker,
                    clarification_resolver=self._clarification_resolver,
                    progress_reporter=progress_sink,
                    diagnostics_callback=self._update_run_diagnostics,
                ),
                timeout=repro_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning("Issue %s: repro stage timed out", issue.id)

        result = ReproGateResult(verdict="missing")
        if not timed_out:
            result = await evaluate_repro_gate(
                session.workspace.path,
                timeout_ms=config.command_timeout_ms,
            )

        if result.proceed:
            assert result.command is not None
            logger.info(
                "Issue %s: reproduction established (%s) — opening fix stage",
                issue.id,
                result.command,
            )
            session.repro_command = result.command
            append_repro_hint(session.workspace.path, result.command)
            # Reset per-run state so the fix stage gets a clean session
            # (mirrors the pipeline mode's between-stage reset).
            session.turn_count = 0
            session.status = "running"
            session.output_text = ""
            session.session_end_reason = None
            session.session_end_summary = ""
            session.run_id = None
            session.consecutive_429_count = 0
            session.rate_limit_pending_turn = None
            session.prompt_override = None
            session.run_kind = "issue"
            return True

        verdict = "repro_stage_timeout" if timed_out else result.verdict
        logger.warning(
            "Issue %s: repro-first gate closed (verdict=%s) — marking FAILED without attempting a fix",
            issue.id,
            verdict,
        )
        session.status = "failed"
        session.session_end_reason = "not_reproducible"
        session.session_end_summary = f"repro gate closed: {verdict}"
        self._registry.mark_failed_with_reason(
            issue.id or "",
            f"not_reproducible ({verdict}): the described behavior could not "
            "be demonstrated; no fix attempted, no PR created.",
        )
        try:
            await self.tracker.create_comment(
                issue.id or "",
                format_repro_gate_comment(issue, result),
            )
        except Exception:
            logger.warning(
                "Issue %s: failed to post repro-gate comment",
                issue.id,
                exc_info=True,
            )
        await self._sync_tracker_issue_state(issue.id or "", "failed")
        self.status_dashboard.on_session_complete(issue.id or "")
        self._state.completed.add(issue.id or "")
        self._state.failed.add(issue.id or "")
        return False

    def _resolve_session_runner(self, session: AgentSession) -> Any:
        """Resolve the requested runner without silently changing semantics."""
        collab_mode = getattr(session, "collaboration_mode", None) or DEFAULT_MODE
        if collab_mode != "single" and session.run_kind == "issue":
            try:
                return _modes.get(collab_mode)
            except KeyError as exc:
                raise RuntimeError(
                    f"Issue {session.issue.id} requested collaboration mode "
                    f"{collab_mode!r}, but that mode is not enabled in workflow.md"
                ) from exc
        return self.stage_runners.get(session.run_kind, self.agent_runner)
