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
from pathlib import Path
from typing import TYPE_CHECKING, Any


from .agent_runner import AgentSession
from .events import EventLevel
from .git_sync_rebase import (
    PRRebaseResult,
    rebase_for_pr,
)
from .issue import Issue
from .prompt_builder import PromptBuilder
from .review_feedback import ReviewFeedbackService, ReviewFollowup
from .status_dashboard import SessionStatus
from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
    _run_git,
    get_repo_root,
)
from .tracker import (
    Intent,
    PullRequestMaintenanceCapability,
    PullRequestRef,
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


class OrchestratorRebaseMixin:
    async def _prepare_intent_reset(self, issue: Issue) -> None:
        """Apply registry-side reset before launching an issue.

        Reads the persisted intent from the registry (set in
        `_poll_and_dispatch`) and, when intent == RETRY:
          1. Closes the existing remote PR (best-effort; failure is
             logged but does not block the reset).
          2. Calls `reset_for_retry(issue_id)` to clear local
             commit_sha / pr_number / pr_url / report_path / status.

        For Intent.FOLLOWUP, no reset is performed here — Sub-C will
        handle the follow-up commit path inside git_sync.sync().

        For Intent.NONE / Intent.BLOCKED, this is a no-op. The
        BLOCKED case never reaches `_launch_issue` because
        `_poll_and_dispatch` skips it.
        """
        issue_id = issue.id or ""
        if not issue_id:
            return
        record = self._registry.get(issue_id)
        if record is None:
            return
        intent = record.intent
        if intent is not Intent.RETRY:
            return

        # 1. Close the existing PR (best-effort).
        pr_number = record.pr_number
        pr_url = record.pr_url
        if pr_number:
            pr_ref = PullRequestRef(number=pr_number, url=pr_url)
            try:
                closed = await self.tracker.close_pull_request(pr_ref)
                if closed:
                    logger.info(
                        "Issue %s retry: closed remote PR %s",
                        issue_id,
                        pr_number,
                    )
                else:
                    logger.warning(
                        "Issue %s retry: tracker could not close PR %s; continuing with local reset",
                        issue_id,
                        pr_number,
                    )
            except Exception as exc:
                logger.warning(
                    "Issue %s retry: close_pull_request raised %s; continuing with local reset",
                    issue_id,
                    exc,
                )

        # 2. Reset the local registry entry. retry_count is bumped
        # inside reset_for_retry by default.
        self._registry.reset_for_retry(issue_id)
        logger.info(
            "Issue %s retry: registry reset (attempt %d)",
            issue_id,
            (self._registry.get(issue_id) or record).retry_count,
        )

    # ------------------------------------------------------------------
    # PR Conflict Auto-Resolution
    # ------------------------------------------------------------------

    def _check_rebase_rate_limit(
        self,
        issue: Issue,
        *,
        force: bool = False,
    ) -> bool:
        """Refuse a rebase when rebase_attempt_count exceeds the cap.

        Returns True if the rebase is allowed (and bumps the
        counter on the registry record), or False if the rate
        limit was hit. ``force=True`` is reserved for the CLI path;
        the daemon path passes ``force=False`` so a hit produces a
        ``rebase_rejected`` audit entry instead of silently
        swallowing the request.
        """
        issue_id = issue.id or ""
        limit = self.workflow.pr_conflict_scan.max_rebase_attempts_per_issue
        record = self._registry.get(issue_id)
        current = record.rebase_attempt_count if record else 0
        if current < limit:
            if record is not None:
                self._registry.increment_rebase_attempt(issue_id)
            return True
        if force:
            return True
        logger.warning(
            "Issue %s rebase rate limit hit: %d >= %d",
            issue_id,
            current,
            limit,
        )
        self._log_audit_event(
            issue_id=issue_id,
            event="rebase_rejected",
            mode="rebase",
            reason=(f"rebase_attempt_count={current} >= max_rebase_attempts_per_issue={limit}"),
            author="daemon",
        )
        return False

    async def _process_rebase_intent(
        self,
        issue: Issue,
        *,
        force: bool | None = None,
    ) -> PRRebaseResult | None:
        """The built-in non-agent rebase path.

        Direct ``asyncio.to_thread(rebase_for_pr, ...)`` — no
        agent / session / provider involved. On a clean rebase
        the registry is cleared and ``rebase_completed`` is
        audited; on a content conflict ``has_conflict`` is set
        and ``rebase_conflict`` is audited (the daemon will pick
        it up in the next ``_process_pending_rebase_conflicts``
        cycle and launch an ``agent_rebase`` run).
        """
        issue_id = issue.id or ""
        record = self._registry.get(issue_id)
        if record is None:
            logger.warning(
                "Issue %s rebase skipped: registry record missing",
                issue_id,
            )
            return None
        if not record.workspace_path or not record.branch_name:
            logger.warning(
                "Issue %s rebase skipped: workspace_path=%r branch_name=%r",
                issue_id,
                record.workspace_path,
                record.branch_name,
            )
            return None
        base_branch = record.base_branch or self.workflow.workspace.base_branch or "main"
        use_force = self.workflow.pr_conflict_scan.use_force_push if force is None else force
        result = await asyncio.to_thread(
            rebase_for_pr,
            workspace_path=record.workspace_path,
            branch_name=record.branch_name,
            base_branch=base_branch,
            force=use_force,
        )
        if result.has_conflict:
            self._registry.mark_conflict(issue_id, result.conflict_files)
            # When the operator used --force, reset the rebase attempt
            # counter so _process_pending_rebase_conflicts can launch
            # the conflict-resolution agent on the next poll cycle.
            if use_force and record is not None:
                record.rebase_attempt_count = 0
                self._registry._save()
            logger.warning(
                "Issue %s rebase left conflicts: %s",
                issue_id,
                ", ".join(result.conflict_files),
            )
            self._log_audit_event(
                issue_id=issue_id,
                event="rebase_conflict",
                mode="force" if use_force else "force-with-lease",
                reason=",".join(result.conflict_files),
                author="daemon",
            )
            return result
        if result.rebased:
            self._registry.clear_conflict(issue_id)
            if result.new_head_sha:
                record.commit_sha = result.new_head_sha
                record.touch()
                self._registry._save()
            logger.info(
                "Issue %s rebase completed pushed=%s method=%s head=%s",
                issue_id,
                result.pushed,
                result.push_method,
                result.new_head_sha,
            )
            self._log_audit_event(
                issue_id=issue_id,
                event="rebase_completed",
                mode=result.push_method,
                reason="pushed" if result.pushed else "already_up_to_date",
                author="daemon",
            )
        else:
            logger.warning("Issue %s rebase did not complete", issue_id)
            self._log_audit_event(
                issue_id=issue_id,
                event="rebase_failed",
                mode="force" if use_force else "force-with-lease",
                reason="git_rebase_or_push_failed",
                author="daemon",
            )
        return result

    async def _process_pending_rebase_conflicts(self) -> None:
        """Launch ``agent_rebase`` for records with content conflicts.

        Iterates the registry, picks records with ``has_conflict=True``
        that are not already running/claimed and not rate-limited, and
        invokes ``_launch_rebase_resolution``. Each resolution opens a
        fresh ``AgentSession`` whose prompt is built by
        ``PromptBuilder.render_rebase``.
        """
        available_slots = self._state.max_concurrent_agents - len(self._state.running)
        if available_slots <= 0:
            logger.debug("No concurrency slots for rebase-resolution")
            return

        records_snapshot = list(self._registry._records.values())
        for record in records_snapshot:
            issue_id = record.issue_id or ""
            if not issue_id:
                continue
            if issue_id in self._state.running or issue_id in self._state.claimed:
                continue
            if not record.has_conflict:
                continue
            if not self._check_rebase_rate_limit(Issue(id=issue_id, identifier=record.issue_identifier)):
                continue
            issue = await self.tracker.fetch_issue_states_by_ids([issue_id])
            issue_obj = issue.get(issue_id) if issue else None
            if issue_obj is None:
                issue_obj = Issue(
                    id=issue_id,
                    identifier=record.issue_identifier,
                    title="(unknown)",
                    branch_name=record.branch_name,
                )
            try:
                ws = await self.workspace.create_for_issue(issue_obj)
                ws_path = getattr(ws, "path", None) or record.workspace_path
                if ws_path and not record.workspace_path:
                    record.workspace_path = str(ws_path)
                    self._registry._save()
            except Exception as exc:
                logger.warning(
                    "Issue %s rebase-resolution: workspace create failed %s; using record.workspace_path",
                    issue_id,
                    exc,
                )
            await self._launch_rebase_resolution(issue_obj)

    async def _process_pr_conflict_scan(self) -> None:
        """Optional daemon scan of PR mergeable state.

        Default-disabled (opt-in via ``workflow.pr_conflict_scan.enabled``).
        When enabled, polls each open PR in the registry, asks the
        tracker for mergeability, and triggers ``_process_rebase_intent``
        if conflicts are reported.

        GitCode fallback: ``MergeableStatus(mergeable=None, has_conflicts=False)``
        is silently skipped — operators on GitCode must use CLI / label /
        comment triggers.
        """
        cfg = self.workflow.pr_conflict_scan
        if not cfg.enabled:
            return
        now = time.monotonic()
        interval_s = cfg.poll_interval_ms / 1000.0
        if now - self._state.pr_conflict_scan_last_run < interval_s:
            return
        self._state.pr_conflict_scan_last_run = now

        for record in list(self._registry._records.values()):
            issue_id = record.issue_id or ""
            if not issue_id or not record.pr_number or not record.branch_name:
                continue
            pr_state = getattr(record, "pr_state", None)
            if pr_state and pr_state not in cfg.scan_states:
                continue
            if not self._check_rebase_rate_limit(Issue(id=issue_id, identifier=record.issue_identifier)):
                continue
            pr_ref = PullRequestRef(
                number=record.pr_number,
                url=record.pr_url,
            )
            if not supports(self.tracker, PullRequestMaintenanceCapability):
                continue
            try:
                status = await self.tracker.fetch_pull_request_mergeable(pr_ref)
            except Exception as exc:
                logger.warning(
                    "PR conflict scan: tracker fetch failed for %s: %s",
                    issue_id,
                    exc,
                )
                continue
            if status is None or not status.has_conflicts:
                continue
            issue = await self.tracker.fetch_issue_states_by_ids([issue_id])
            issue_obj = issue.get(issue_id) if issue else None
            if issue_obj is None:
                issue_obj = Issue(
                    id=issue_id,
                    identifier=record.issue_identifier,
                    title="(unknown)",
                    branch_name=record.branch_name,
                )
            await self._process_rebase_intent(issue_obj)

    async def _launch_rebase_resolution(self, issue: Issue) -> None:
        """Launch an ``agent_rebase`` session to resolve a content conflict.

        Mirrors ``_launch_issue`` for the conflict-resolution path.
        The session is tagged with ``run_kind="agent_rebase"`` so the
        agent runner can route the run through a rebase-tailored
        prompt and dispatch policy.
        """
        record = self._registry.get(issue.id or "")
        workspace_path = record.workspace_path if record else None
        # Synthesize a minimal Workspace stub when no real workspace
        # is available; the agent runner only needs ``workspace.path``
        # to be present for prompt injection.
        if workspace_path:
            from .workspace import Workspace as _Ws

            workspace = _Ws(path=Path(workspace_path), issue_identifier=issue.identifier or "")
        else:
            from .workspace import Workspace as _Ws

            workspace = _Ws(path=Path("/tmp"), issue_identifier=issue.identifier or "")  # nosec B108
        session = AgentSession(
            issue=issue,
            workspace=workspace,
            pause_resume_event=asyncio.Event(),
            event_queue=asyncio.Queue(),
        )
        clarification_record = self._registry.get(issue.id or "")
        if clarification_record is not None and clarification_record.local_answer:
            session.clarification_answer = clarification_record.local_answer
            session.clarification_source = clarification_record.local_answer_source
            if clarification_record.question_history:
                session.clarification_question = "\n".join(
                    f"- {question}" for question in clarification_record.question_history
                )
        session.run_kind = "agent_rebase"
        # Route the run through the purpose-built rebase prompt
        # (resolve markers -> git add -> git rebase --continue ->
        # --force-with-lease push, "do NOT open a new PR"). Without this
        # the session ran the generic issue prompt and the agent never
        # knew it was supposed to resolve the rebase conflict.
        rebase_branch = (record.branch_name if record else None) or issue.branch_name or ""
        rebase_base = (record.base_branch if record else None) or self.workflow.workspace.base_branch or "main"
        rebase_conflicts = tuple(record.conflict_files) if record else ()
        session.prompt_override = PromptBuilder.render_rebase(
            issue=issue,
            branch_name=rebase_branch,
            base_branch=rebase_base,
            conflict_files=rebase_conflicts,
        )
        self._prepare_rebase_session(session)
        self._state.running[issue.id or ""] = session
        try:
            progress_sink = self._build_session_sink(issue.id or "")
            run_timeout_seconds = self.workflow.agent.run_timeout_ms / 1000.0
            session.timeout_deadline_at = time.time() + run_timeout_seconds
            await asyncio.wait_for(
                self.agent_runner.run(
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
        except Exception as exc:
            logger.error(
                "Issue %s rebase-resolution: run_session raised %s",
                issue.id,
                exc,
            )
        finally:
            self._state.running.pop(issue.id or "", None)
            # Completion handling. Without this the record kept
            # has_conflict=True forever -> the next poll re-launched an
            # agent_rebase run in an infinite loop (repeated "Run in
            # progress" placeholder comments + task started/task completed
            # oscillation on IM), and the PR link never reached IM.
            # Detect resolution via git ground-truth (not session.status),
            # clear the conflict on success, and emit a PR-link-bearing
            # event either way.
            try:
                await self._finalize_rebase_resolution(issue, session)
            except Exception:
                logger.exception(
                    "Issue %s rebase-resolution finalizer failed",
                    issue.id,
                )

    async def _finalize_rebase_resolution(
        self,
        issue: Issue,
        session: AgentSession,
    ) -> None:
        """Post-run completion handling for an ``agent_rebase`` session.

        ``_launch_rebase_resolution`` historically popped the session out of
        ``_state.running`` and did nothing else. That left ``has_conflict``
        set on the registry record, so ``_process_pending_rebase_conflicts``
        re-launched a fresh agent_rebase run on every poll -> an infinite
        loop (repeated "## ClawCodex Run Summary / Run in progress."
        placeholder comments, and task started/task completed oscillation on IM), and
        because the rebase path never runs ``git_sync`` or emits a
        ``pr=``-bearing event, the PR link never reached Feishu/IM.

        This checks git ground-truth (NOT ``session.status`` - the
        agent_runner completion heuristics are tuned for normal issue
        runs and can misclassify a successful rebase+push as
        "no_changes_produced") and either clears the conflict + emits a
        PR-link-bearing ``pr.updated`` event, or records an unresolved
        failure so the operator can intervene.
        """
        issue_id = issue.id or ""
        record = self._registry.get(issue_id)
        workspace_path = record.workspace_path if record else None
        resolved, new_head = await self._rebase_conflict_resolved(
            workspace_path,
            previous_head=record.commit_sha if record else None,
            base_branch=record.base_branch if record else None,
            branch_name=(record.branch_name if record else None) or issue.branch_name,
        )
        pr_url = record.pr_url if record else None
        if resolved:
            self._registry.clear_conflict(issue_id)
            if new_head and record is not None:
                record.commit_sha = new_head
                record.touch()
                self._registry._save()
            self.status_dashboard.on_session_complete(issue_id)
            self._state.completed.add(issue_id)
            self._emit_im_event(
                issue_id,
                "pr.updated",
                EventLevel.SUCCESS,
                "rebase 冲突已解决，PR 已更新",
                self._issue_payload(issue, pr=pr_url, commit=new_head),
            )
            self._log_audit_event(
                issue_id=issue_id,
                event="rebase_resolved",
                mode="agent_rebase",
                reason=f"conflicts resolved, head={new_head}",
                author="daemon",
            )
            logger.info(
                "Issue %s rebase-resolution succeeded head=%s pr=%s",
                issue_id,
                new_head,
                pr_url,
            )
            return
        # Conflict not resolved - keep has_conflict so the next poll cycle
        # can retry (bounded by max_rebase_attempts_per_issue). Surface a
        # failure event WITH the PR link so the operator can intervene.
        self.status_dashboard.on_session_failed(issue_id, "rebase_unresolved")
        self._emit_im_event(
            issue_id,
            "issue.failed",
            EventLevel.WARN,
            "rebase 冲突未解决，请人工介入",
            self._issue_payload(issue, pr=pr_url),
        )
        self._log_audit_event(
            issue_id=issue_id,
            event="rebase_unresolved",
            mode="agent_rebase",
            reason="conflicts remain after agent_rebase run",
            author="daemon",
        )
        logger.warning(
            "Issue %s rebase-resolution did not resolve conflicts; has_conflict stays set for retry",
            issue_id,
        )

    async def _rebase_conflict_resolved(
        self,
        workspace_path: str | None,
        *,
        previous_head: str | None = None,
        base_branch: str | None = None,
        branch_name: str | None = None,
    ) -> tuple[bool, str | None]:
        """Check git ground-truth for whether the agent finished the rebase.

        Returns ``(resolved, new_head_sha)``. ``resolved=True`` only when
        there are no unmerged files or active sequencer, the expected base
        is an ancestor of HEAD, and the pushed remote feature ref equals the
        local HEAD.  This distinguishes a completed rebase from
        ``git rebase --abort`` and from a local-only rebase whose push failed.

        We trust git state over ``session.status``: the agent_runner
        completion heuristics (stagnation / read_only_loop /
        no_changes_produced) are tuned for normal issue runs, not rebase
        resolution - a successful conflict resolution that pushes and
        leaves a clean tree can be misclassified as "no changes produced".
        """
        if not workspace_path or not base_branch or not branch_name:
            return False, None
        repo_root = await asyncio.to_thread(get_repo_root, workspace_path)
        if not repo_root:
            return False, None

        def _check() -> tuple[bool, str | None]:
            # Unmerged files -> conflict markers still present in the worktree.
            unmerged, _, _ = _run_git(["diff", "--name-only", "--diff-filter=U"], repo_root)
            if unmerged.strip():
                return False, None
            # REBASE_HEAD is deliberately not used here. Git can retain that
            # pseudo-ref after a completed rebase, so its presence caused
            # successfully resolved conflicts to be reported as failures.
            # The sequencer's state directories are the authoritative signal
            # that a merge- or apply-backed rebase is still active.
            for state_name in ("rebase-merge", "rebase-apply"):
                state_out, _, state_rc = _run_git(
                    ["rev-parse", "--git-path", state_name],
                    repo_root,
                )
                if state_rc != 0 or not state_out:
                    return False, None
                state_path = Path(state_out)
                if not state_path.is_absolute():
                    state_path = Path(repo_root) / state_path
                if state_path.exists():
                    return False, None
            head_out, _, head_rc = _run_git(["rev-parse", "HEAD"], repo_root)
            if head_rc != 0 or not head_out.strip():
                return False, None
            head = head_out.strip()
            if previous_head and head == previous_head:
                return False, None

            current_branch, _, branch_rc = _run_git(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                repo_root,
            )
            if branch_rc != 0 or current_branch.strip() != branch_name:
                return False, None

            # Query both refs together so the ancestry decision uses the
            # current remote base rather than a potentially stale
            # ``origin/<base>`` left from the initial conflict attempt.
            remote_out, _, remote_rc = _run_git(
                [
                    "ls-remote",
                    "--heads",
                    "origin",
                    f"refs/heads/{base_branch}",
                    f"refs/heads/{branch_name}",
                ],
                repo_root,
            )
            if remote_rc != 0:
                return False, None
            remote_lines = [line.split() for line in remote_out.splitlines() if line.strip()]
            remote_heads = {parts[1]: parts[0] for parts in remote_lines if len(parts) >= 2}
            remote_base = remote_heads.get(f"refs/heads/{base_branch}")
            remote_feature = remote_heads.get(f"refs/heads/{branch_name}")
            if not remote_base or remote_feature != head:
                return False, None

            # A completed rebase must contain the current target tip.  An
            # aborted rebase returns to the old feature head and fails this
            # ancestry check even though the worktree itself is clean.
            _, _, ancestor_rc = _run_git(
                ["merge-base", "--is-ancestor", remote_base, head],
                repo_root,
            )
            if ancestor_rc != 0:
                return False, None
            return True, head

        return await asyncio.to_thread(_check)

    def _prepare_rebase_session(self, session: AgentSession) -> None:
        """Copy registry conflict metadata onto the session.

        Sets ``session.conflict_files`` from the registry so the
        agent runner / prompt builder can read which files git
        left in conflict state and inject them into the prompt.
        """
        record = self._registry.get(session.issue.id or "")
        if record is None:
            session.conflict_files = ()
            return
        session.conflict_files = tuple(record.conflict_files)

    async def _handle_rebase_control(self, issue_id: str, extra: str) -> None:
        """Handle a CLI-written rebase control file.

        Format::

            rebase
            <issue_id>
            force=0|1
            <reason>

        Routes through ``_process_rebase_intent`` so the orchestrator
        itself performs the rebase (no agent for clean rebases).
        Conflict results flow back into the registry and are picked
        up by ``_process_pending_rebase_conflicts`` on the next
        poll.
        """
        if not issue_id:
            return
        record = self._registry.get(issue_id)
        if record is None:
            logger.warning("rebase control: issue %s not in registry", issue_id)
            return
        if issue_id in self._state.running:
            logger.info(
                "rebase control: issue %s already running, skipping",
                issue_id,
            )
            return
        if not record.pr_number or not record.workspace_path or not record.branch_name:
            logger.warning(
                "rebase control: issue %s missing pr_number/workspace/branch",
                issue_id,
            )
            return

        force = False
        reason = ""
        if extra:
            for line in extra.split("\n"):
                token = line.strip()
                if token.startswith("force="):
                    force = token.split("=", 1)[1].strip() in ("1", "true", "yes")
                elif token:
                    reason = token
        logger.info(
            "rebase control: dispatching issue_id=%s force=%s reason=%r",
            issue_id,
            force,
            reason,
        )
        issue = await self.tracker.fetch_issue_states_by_ids([issue_id])
        issue_obj = issue.get(issue_id) if issue else None
        if issue_obj is None:
            issue_obj = Issue(
                id=issue_id,
                identifier=record.issue_identifier,
                title="(unknown)",
                branch_name=record.branch_name,
            )
        # The CLI already enforced the rate-limit preview; honor the
        # operator's explicit --force when set.
        await self._process_rebase_intent(issue_obj, force=force)

    def _prepare_intent_session(self, session: AgentSession) -> None:
        """Wire the session for an intent-driven run.

        Called from `_launch_issue` immediately after the AgentSession
        is constructed. Reads the registry's intent field and:

          - Intent.FOLLOWUP → set `run_kind = "agent_followup"`, copy
            the existing PR (number + url) and base_branch onto the
            session, and pin `issue.branch_name` to the registry
            branch so `_ensure_work_branch` reuses it.
          - Intent.RETRY → the registry was already reset by
            `_prepare_intent_reset`; nothing more to do here. The
            session is a fresh issue-style run.
          - Intent.NONE / Intent.BLOCKED → no-op.

        Sub-C mirrors the review_followup pattern (see
        `_launch_review_followup`): we reuse the same branch + PR
        and append a commit via git_sync(mode="followup").
        """
        issue_id = session.issue.id or ""
        if not issue_id:
            return
        record = self._registry.get(issue_id)
        if record is None or record.intent is not Intent.FOLLOWUP:
            return

        session.run_kind = "review_retry" if record.last_command == "/issue review --reject" else "agent_followup"

        # Wire the existing PR so git_sync reuses it instead of
        # creating a new one.
        if record.pr_number:
            session.pull_request = PullRequestRef(
                number=record.pr_number,
                url=record.pr_url,
            )

        # Pin base_branch so git_sync.push targets the right base.
        if record.base_branch:
            session.base_branch = record.base_branch

        # Pin issue.branch_name so _ensure_work_branch reuses the
        # existing feature branch (otherwise it would fall back to
        # the default name and create a new one).
        if record.branch_name and hasattr(session.issue, "branch_name"):
            try:
                session.issue.branch_name = record.branch_name
            except Exception:
                # Issue is a frozen dataclass in some contexts; in
                # that case the registry's branch_name still wins
                # because git_sync.sync also reads from the
                # registry-aware session.base_branch.
                logger.debug(
                    "Could not pin issue.branch_name for followup issue %s; relying on session.base_branch",
                    issue_id,
                )

        # Wire feedback metadata so git_sync writes review-id /
        # review-body into the commit message.  pending_feedback_ids
        # are the unprocessed review comments that prompted this
        # follow-up; feedback_commit_body is unavailable here (the
        # registry stores IDs, not body text) so review-body: is
        # omitted for agent_followup — review-pr: is still written.
        session.feedback_ids = list(record.pending_feedback_ids)

        logger.info(
            "Issue %s followup: session wired (branch=%s pr=%s base=%s)",
            issue_id,
            getattr(session.issue, "branch_name", None),
            getattr(getattr(session, "pull_request", None), "number", None),
            session.base_branch,
        )

    async def _process_review_feedback(self) -> None:
        config = self.workflow.review_feedback
        if not config.enabled:
            return
        available_slots = self._state.max_concurrent_agents - len(self._state.running)
        if available_slots <= 0:
            return

        service = ReviewFeedbackService(
            tracker=self.tracker,
            registry=self._registry,
            config=config,
        )
        try:
            followups = await service.collect_followups(available_slots)
        except Exception as exc:
            logger.error("Failed to collect PR review feedback: %s", exc)
            return

        for followup in followups:
            issue_id = followup.issue.id or ""
            if issue_id in self._state.running or issue_id in self._state.claimed:
                continue
            if config.mode != "auto":
                self._registry.mark_feedback_pending(
                    issue_id,
                    [item.id for item in followup.feedback],
                    feedback_urls={item.id: item.url for item in followup.feedback if item.url},
                )
                logger.info(
                    "PR feedback pending manual follow-up issue_id=%s feedback_count=%d",
                    issue_id,
                    len(followup.feedback),
                )
                continue
            self._state.claimed.add(issue_id)
            await self._launch_review_followup(followup)

    async def _launch_review_followup(self, followup: ReviewFollowup) -> None:
        issue = followup.issue
        issue.branch_name = followup.record.branch_name
        prompt = PromptBuilder.render_review_feedback(
            issue=issue,
            pull_request=followup.pull_request,
            branch_name=followup.record.branch_name or "",
            feedback=followup.feedback,
        )
        try:
            workspace = await self.workspace.create_for_issue(issue)
        except Exception as exc:
            logger.error("Workspace creation failed for PR follow-up issue_id=%s: %s", issue.id, exc)
            self._state.claimed.discard(issue.id or "")
            return
        start_commit_sha = await self.workspace.current_head(workspace.path)

        session = AgentSession(
            issue=issue,
            workspace=workspace,
            pause_resume_event=asyncio.Event(),
            event_queue=asyncio.Queue(),
            prompt_override=prompt,
            run_kind="review_followup",
        )
        session.pull_request = followup.pull_request
        session.base_branch = followup.record.base_branch
        session.start_commit_sha = start_commit_sha
        session.feedback_ids = [item.id for item in followup.feedback]
        # Use the first feedback body as the commit message for descriptive titles
        first_body = (followup.feedback[0].body or "").strip() if followup.feedback else ""
        session.feedback_commit_body = first_body
        self._state.running[issue.id or ""] = session
        if self._registry.mark_running(issue.id or "") is None:
            logger.warning(
                "Review follow-up started without registry record issue_id=%s",
                issue.id,
            )
        followup_record = self._registry.increment_followup_attempt(issue.id or "")
        session.issue_attempt = max(1, getattr(followup.record, "attempt_count", 0) + 1)
        session.followup_attempt = followup_record.followup_attempt_count if followup_record is not None else 1
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
        # Root-cause fix: register issue_id → task mapping so the
        # stop command can cancel a specific running issue.
        issue_id = issue.id or ""
        self._issue_tasks[issue_id] = task

        def _unregister_issue_task(t: asyncio.Task) -> None:
            self._issue_tasks.pop(issue_id, None)

        task.add_done_callback(_unregister_issue_task)
