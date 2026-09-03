#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Post-run git sync for repository-backed workspaces."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extensions.orchestrator_runtime.adapters.clawcodex_compat import (  # pylint: disable=import-error,no-name-in-module
    _run_git,
    get_current_branch,
    get_default_branch,
    get_file_status,
    get_repo_root,
)
from .config.schema import AgentConfig, HooksConfig, PrTemplateConfig  # pylint: disable=import-error,no-name-in-module
from .issue import Issue  # pylint: disable=import-error
from .prompt_context import resolve_python_executable  # pylint: disable=import-error
from .tracker import (  # pylint: disable=import-error,no-name-in-module
    PullRequestCapability,
    PullRequestMaintenanceCapability,
    PullRequestRef,
    TrackerAdapter,
    supports,
)
from .workspace import Workspace
from .git_sync_ops import GitSyncOpsMixin  # pylint: disable=import-error,no-name-in-module
from .git_sync_rebase import GitSyncError  # pylint: disable=import-error,no-name-in-module

logger = logging.getLogger(__name__)

_OUTPUT_TAIL_CHARS = 4_000


def _tail(output: str) -> str:
    """Return the last chunk of output — enough context for a report without megabytes of logs."""
    if len(output) <= _OUTPUT_TAIL_CHARS:
        return output
    return f"…(truncated)…\n{output[-_OUTPUT_TAIL_CHARS:]}"


@dataclass(frozen=True)
class GitSyncResult:
    """Result of post-run git synchronization."""

    branch_name: str
    base_branch: str
    commit_sha: str | None = None
    pull_request: PullRequestRef | None = None
    committed: bool = False
    pushed: bool = False
    has_conflict: bool = False
    conflict_files: tuple[str, ...] = field(default_factory=tuple)
    pending_review: bool = False  # True for LocalTracker after successful commit
    # When no reviewable commit exists (daemon read-only loop termination),
    # mark the reason so the orchestrator calls mark_failed_with_reason.
    session_end_reason: str | None = None


class VerificationFailed(GitSyncError):
    """Raised when configured verification commands fail."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


class HookFailedError(GitSyncError):
    """Raised when a configured sync hook fails."""

    def __init__(self, hook_name: str, message: str, output: str = "") -> None:
        super().__init__(message)
        self.hook_name = hook_name
        self.output = output


class GitSyncPostCommitError(GitSyncError):
    """Raised when post-commit sync steps fail after a commit exists."""

    def __init__(self, cause: VerificationFailed | HookFailedError, result: GitSyncResult) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.result = result
        self.output = getattr(cause, "output", "")
        self.hook_name = getattr(cause, "hook_name", None)


class GitSyncService(GitSyncOpsMixin):
    """Perform commit, push, and PR creation after a run."""

    def __init__(
        self,
        tracker: TrackerAdapter,
        branch_prefix: str | None = None,
        gitignore_patterns: list[str] | None = None,
        agent_config: AgentConfig | None = None,
        hooks_config: HooksConfig | None = None,
        git_username: str | None = None,
        git_email: str | None = None,
        upstream_clone_url: str | None = None,
        fork_clone_url: str | None = None,
        pr_template: PrTemplateConfig | None = None,
    ) -> None:
        self.tracker = tracker
        self._branch_prefix = branch_prefix
        self._agent_config = agent_config or AgentConfig()
        self._hooks_config = hooks_config or HooksConfig()
        self._git_username = git_username
        self._git_email = git_email
        self._upstream_clone_url = upstream_clone_url
        self._fork_clone_url = fork_clone_url
        self._pr_template = pr_template or PrTemplateConfig()
        self._gitignore_patterns = list(
            gitignore_patterns
            or [
                ".event_streams",
                ".orchestrator_control",
                ".orchestrator_workspace",
                ".operator_hints.md",
                ".reports",
                ".clawcodex_clarification_queue.json",
                ".clawcodex_issue_registry.json",
                ".clawcodex_workspace.lock",
                "*.pyc",
                "__pycache__",
                "*.egg-info",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "*.log",
                "analysis.md",
                "changes_summary.md",
                "implementation_notes.md",
                "verification_report.md",
            ]
        )
        # Agent-generated PR artifacts are orchestration metadata, never part
        # of the implementation commit. Keep that invariant even when callers
        # provide their own gitignore list.
        for artifact in (
            "analysis.md",
            "changes_summary.md",
            "implementation_notes.md",
            "verification_report.md",
        ):
            if artifact not in self._gitignore_patterns:
                self._gitignore_patterns.append(artifact)

    def _fork_mode(self) -> bool:
        """Return True when in fork workflow mode (upstream and fork differ)."""
        upstream = self._upstream_clone_url
        if not upstream:
            return False
        fork = self._fork_clone_url
        if not fork:
            return False
        return upstream.rstrip("/") != fork.rstrip("/")

    @staticmethod
    def _extract_owner_from_url(clone_url: str) -> str | None:
        """Extract owner from clone URL (e.g. https://gitcode.com/owner/repo.git -> owner)."""
        m = re.search(r"[:/]([^/]+?)/([^/]+?)(?:\.git)?$", clone_url.rstrip("/"))
        return m.group(1) if m else None

    @staticmethod
    def _extract_owner_repo_from_url(clone_url: str) -> str | None:
        """Extract owner/repo from clone URL (e.g. https://gitcode.com/owner/repo.git → owner/repo)."""
        m = re.search(r"[:/]([^/]+?)/([^/]+?)(?:\.git)?$", clone_url.rstrip("/"))
        return f"{m.group(1)}/{m.group(2)}" if m else None

    async def sync(
        self,
        session: Any,
        *,
        mode: str = "default",
    ) -> GitSyncResult | None:
        """Commit/push/PR sync.

        When `mode == "followup"`, the session is expected
        to already carry a `pull_request` attribute (set by the
        orchestrator from the registry record) and the run is treated
        as a same-branch follow-up commit. The commit message uses
        the "fix:" prefix (vs. "feat:" for new runs) and the existing
        `update_pull_request` path appends a `## ClawCodex Follow-up
        #N` section to the PR body (already in place).

        Other modes (default / future) are unchanged.
        """
        # Validate followup-mode prerequisites BEFORE any
        # workspace / repo_root I/O. A follow-up that forgot to wire
        # the existing PR would otherwise silently open a brand-new
        # PR, which is exactly what follow-up is trying to avoid.
        if mode == "followup":
            existing_pr = getattr(session, "pull_request", None)
            if existing_pr is None:
                raise GitSyncError(
                    "GitSyncService.sync(mode='followup') requires "
                    "session.pull_request to be set; orchestrator "
                    "should populate it from the IssueRegistry record"
                )

        workspace: Workspace = session.workspace
        issue: Issue = session.issue

        repo_root = await asyncio.to_thread(get_repo_root, str(workspace.path))
        if not repo_root:
            return None

        # Check if tracker is LocalTrackerAdapter — skip push/PR for local-only repos
        from .local_tracker.adapter import LocalTrackerAdapter

        is_local_tracker = isinstance(self.tracker, LocalTrackerAdapter)
        workspace_strategy = getattr(session, "workspace_strategy", "isolated")
        is_sequential = workspace_strategy == "sequential"
        self._sync_git_exclude(repo_root)
        no_push = is_local_tracker or is_sequential

        followup_pr = getattr(session, "pull_request", None)
        base_branch = getattr(session, "base_branch", None)
        if not base_branch:
            base_branch = await asyncio.to_thread(get_default_branch, repo_root)
        if is_sequential:
            branch_name = getattr(session, "integration_branch", None)
            if not branch_name:
                branch_name = await asyncio.to_thread(get_current_branch, repo_root) or base_branch
        else:
            branch_name = await asyncio.to_thread(self._ensure_work_branch, repo_root, issue, base_branch)
        changed = bool(await asyncio.to_thread(get_file_status, repo_root))

        commit_sha: str | None = None
        committed = False
        has_run_commit = False
        pushed = False
        has_conflict = False
        conflict_files: tuple[str, ...] = ()
        if changed:
            await asyncio.to_thread(self._ensure_commit_identity, repo_root)
            if is_sequential:
                await self._run_pre_commit_hook(repo_root, session)
            await asyncio.to_thread(self._run_git_checked, ["add", "-A"], repo_root)
            await asyncio.to_thread(self._unstage_orchestrator_artifacts, repo_root)
            await asyncio.to_thread(self._apply_file_whitelist, repo_root)

            # Check if there are staged changes after add/unstage/whitelist
            # If not, agent may have already committed (e2e workflow)
            has_staged = await asyncio.to_thread(self._has_staged_changes, repo_root)
            agent_committed = False
            if not has_staged:
                # No staged changes - check if agent already committed by comparing HEAD
                current_sha = await asyncio.to_thread(self._run_git_output, ["rev-parse", "HEAD"], repo_root)
                start_commit_sha = getattr(session, "start_commit_sha", None)
                has_run_commit = bool(start_commit_sha and current_sha != start_commit_sha)

                if has_run_commit:
                    # Agent already committed, skip auto-commit
                    agent_committed = True
                    commit_sha = current_sha
                    # Amend agent's commit with review metadata (safe before push)
                    if followup_pr is not None:
                        await asyncio.to_thread(self._ensure_review_metadata, repo_root, session, followup_pr)
                        commit_sha = await asyncio.to_thread(self._run_git_output, ["rev-parse", "HEAD"], repo_root)
                else:
                    # No staged changes and HEAD unchanged - likely whitelist filtered everything
                    # Fall through to normal commit flow (which will create empty commit or skip)
                    commit_sha = None
            else:
                commit_message = self._build_commit_message(
                    issue,
                    followup=followup_pr is not None,
                    feedback_body=getattr(session, "feedback_commit_body", None),
                    session=session,
                )
                await asyncio.to_thread(self._run_git_checked, ["commit", "-m", commit_message], repo_root)
                commit_sha = await asyncio.to_thread(self._run_git_output, ["rev-parse", "HEAD"], repo_root)
                committed = True
            try:
                if not is_sequential and not agent_committed:
                    await self._run_pre_commit_hook(repo_root, session)
                    commit_sha = await asyncio.to_thread(self._run_git_output, ["rev-parse", "HEAD"], repo_root)
                await self._run_pre_push_verification(repo_root, session)
            except (VerificationFailed, HookFailedError) as exc:
                # Roll back the just-created commit since verification failed
                # But only if we actually created one this run (`committed`);
                # otherwise HEAD~1 would pop a pre-existing baseline commit.
                if committed and not agent_committed:
                    try:
                        await asyncio.to_thread(self._run_git_checked, ["reset", "--mixed", "HEAD~1"], repo_root)
                    except GitSyncError:
                        pass  # No commit to rollback or reset failed — proceed anyway
                    committed = False
                raise self._post_commit_error(
                    exc,
                    branch_name=branch_name,
                    base_branch=base_branch,
                    commit_sha=commit_sha,
                    committed=committed,
                    pushed=pushed,
                    has_conflict=has_conflict,
                    conflict_files=conflict_files,
                    pull_request=followup_pr,
                    is_local_tracker=is_local_tracker,
                ) from exc
            if no_push:
                # LocalTracker: no remote, skip push but record branch info
                pass
            else:
                pushed, has_conflict, conflict_files = await asyncio.to_thread(
                    self._push_with_recovery,
                    repo_root,
                    branch_name,
                )
        else:
            commit_sha = await asyncio.to_thread(self._run_git_output, ["rev-parse", "HEAD"], repo_root)
            start_commit_sha = getattr(session, "start_commit_sha", None)
            has_run_commit = bool(start_commit_sha and commit_sha != start_commit_sha)
            try:
                await self._run_pre_push_verification(repo_root, session)
            except (VerificationFailed, HookFailedError) as exc:
                raise self._post_commit_error(
                    exc,
                    branch_name=branch_name,
                    base_branch=base_branch,
                    commit_sha=commit_sha,
                    committed=has_run_commit,
                    pushed=False,
                    has_conflict=False,
                    conflict_files=(),
                    pull_request=followup_pr,
                    is_local_tracker=is_local_tracker,
                ) from exc
            if not has_run_commit:
                commit_sha = None
            # No staged changes but branch may have diverged from origin — still push
            if branch_name and not no_push:
                # For follow-up PRs, push directly without rebase —
                # the agent already committed on the existing PR branch.
                if followup_pr is not None:
                    pushed, has_conflict, conflict_files = await asyncio.to_thread(
                        self._push_directly,
                        repo_root,
                        branch_name,
                    )
                else:
                    pushed, has_conflict, conflict_files = await asyncio.to_thread(
                        self._push_with_recovery,
                        repo_root,
                        branch_name,
                    )

        pr_ref: PullRequestRef | None = followup_pr
        pr_title = self._build_pr_title(issue)
        # Prevent empty PR: skip creation when no reviewable commit exists.
        has_reviewable_commit = committed or has_run_commit
        if pr_ref is None and branch_name != base_branch and not no_push and has_reviewable_commit:
            # Fork workflow: head ref needs fork owner/repo prefix
            head_ref = branch_name
            if self._fork_mode() and self._fork_clone_url:
                fork_owner_repo = self._extract_owner_repo_from_url(self._fork_clone_url)
                if fork_owner_repo:
                    head_ref = f"{fork_owner_repo}:{branch_name}"
            if supports(self.tracker, PullRequestCapability):
                pr_ref = await self.tracker.ensure_pull_request(
                    issue=issue,
                    head_branch=head_ref,
                    base_branch=base_branch,
                    title=pr_title,
                    body=self._build_pr_body(
                        issue,
                        commit_sha,
                        branch_name,
                        base_branch,
                        session=session,
                        pull_request=None,
                    ),
                )
            # GitCode PR creation may not return number/url immediately;
            # fall back to listing open PRs and matching by head branch.
            if pr_ref is not None and (not pr_ref.number or not pr_ref.url):
                pr_ref = await self._find_pr_fallback(
                    pr_ref,
                    head_branch=head_ref,
                    base_branch=base_branch,
                )

        report_result = self._write_report(
            session=session,
            branch_name=branch_name,
            base_branch=base_branch,
            commit_sha=commit_sha,
            pull_request=pr_ref,
        )

        if pr_ref is not None and not no_push:
            # PR body/title is user-owned after first creation. Follow-up
            # results are posted as thread replies, not body overwrites.
            if followup_pr is not None:
                updated_pr = None
            elif supports(self.tracker, PullRequestMaintenanceCapability):
                updated_pr = await self.tracker.update_pull_request(
                    pull_request=pr_ref,
                    title=pr_title,
                    body=self._build_pr_body(
                        issue,
                        commit_sha,
                        branch_name,
                        base_branch,
                        session=session,
                        pull_request=pr_ref,
                    ),
                )
            else:
                # Tracker lacks PR maintenance capability (e.g. Linear)
                updated_pr = None
            if updated_pr is not None:
                pr_ref = self._merge_pr_ref(updated_pr, pr_ref)
                if not pr_ref.number or not pr_ref.url:
                    pr_ref = await self._find_pr_fallback(
                        pr_ref,
                        head_branch=branch_name,
                        base_branch=base_branch,
                    )
                self._write_report(
                    session=session,
                    branch_name=branch_name,
                    base_branch=base_branch,
                    commit_sha=commit_sha,
                    pull_request=pr_ref,
                )

        has_reviewable_commit = committed or has_run_commit
        try:
            await self._run_post_sync_hook(repo_root, session)
        except (VerificationFailed, HookFailedError) as exc:
            if has_reviewable_commit:
                raise self._post_commit_error(
                    exc,
                    branch_name=branch_name,
                    base_branch=base_branch,
                    commit_sha=commit_sha,
                    committed=has_reviewable_commit,
                    pushed=pushed,
                    has_conflict=has_conflict,
                    conflict_files=conflict_files,
                    pull_request=pr_ref,
                    is_local_tracker=is_local_tracker,
                ) from exc
            raise

        # Only post summary comment when there is a reviewable commit or existing PR.
        if has_reviewable_commit or pr_ref is not None:
            await self._update_summary_comment(
                session=session,
                branch_name=branch_name,
                base_branch=base_branch,
                commit_sha=commit_sha,
                pull_request=pr_ref,
                committed=has_reviewable_commit,
                pushed=pushed if not no_push else False,
                report_path=(report_result.persistent_markdown_path if report_result is not None else None),
            )

        # Mark session_end_reason so the orchestrator can decide between
        # mark_synced and mark_failed_with_reason.
        session_end_reason: str | None = None
        if not has_reviewable_commit and pr_ref is None:
            session_end_reason = "empty_branch_no_commits"

        return GitSyncResult(
            branch_name=branch_name,
            base_branch=base_branch,
            commit_sha=commit_sha,
            pull_request=pr_ref,
            committed=has_reviewable_commit,
            pushed=pushed,
            has_conflict=has_conflict,
            conflict_files=conflict_files,
            pending_review=bool((is_local_tracker or self._agent_config.review_required) and has_reviewable_commit),
            session_end_reason=session_end_reason,
        )

    def _post_commit_error(
        self,
        cause: VerificationFailed | HookFailedError,
        *,
        branch_name: str,
        base_branch: str,
        commit_sha: str | None,
        committed: bool,
        pushed: bool,
        has_conflict: bool,
        conflict_files: tuple[str, ...],
        pull_request: PullRequestRef | None,
        is_local_tracker: bool,
    ) -> GitSyncPostCommitError:
        result = GitSyncResult(
            branch_name=branch_name,
            base_branch=base_branch,
            commit_sha=commit_sha,
            pull_request=pull_request,
            committed=committed,
            pushed=pushed,
            has_conflict=has_conflict,
            conflict_files=conflict_files,
            pending_review=bool((is_local_tracker or self._agent_config.review_required) and committed),
        )
        return GitSyncPostCommitError(cause, result)

    async def _run_pre_commit_hook(self, repo_root: str, session: Any) -> None:
        command = self._hooks_config.pre_commit
        if not command:
            return
        output = await self._run_shell(command, repo_root, self._hooks_config.timeout_ms)
        if (
            await asyncio.to_thread(get_file_status, repo_root)
            and getattr(session, "workspace_strategy", "isolated") != "sequential"
        ):
            await asyncio.to_thread(self._run_git_checked, ["add", "-A"], repo_root)
            await asyncio.to_thread(self._run_git_checked, ["commit", "--amend", "--no-edit"], repo_root)
        setattr(session, "pre_commit_output", output)

    async def _run_pre_push_verification(self, repo_root: str, session: Any) -> None:
        outputs: list[str] = []
        verification_status = "passed"
        for label, command in (
            ("test", self._agent_config.test_command),
            ("build", self._agent_config.build_command),
            ("lint", self._agent_config.lint_command),
        ):
            if not command:
                continue
            try:
                output = await self._run_shell(
                    command,
                    repo_root,
                    self._agent_config.verification.timeout_ms,
                )
            except VerificationFailed as exc:
                raise VerificationFailed(f"{label} verification failed", exc.output) from exc  # pylint: disable=bad-exception-cause
            outputs.append(f"## {label}\n{output}".strip())
        # Regression guard (defect R1): with no test_command configured the
        # loop above runs nothing and verification used to pass vacuously.
        # Fall back to an auto-detected test run compared against the
        # pre-change baseline so net-new failures block the push.
        if not self._agent_config.test_command and self._agent_config.verification.regression_guard:
            verification_status, guard_output = await self._run_regression_guard(repo_root, session)
            if guard_output:
                outputs.append(f"## regression_guard\n{guard_output}".strip())
        # Repro-first gate follow-through: the reproduction command that
        # demonstrated the bug (non-zero exit before the fix) must have
        # turned green. A still-failing reproduction blocks the push: the
        # fix did not fix the observed behavior.
        repro_command = getattr(session, "repro_command", None)
        if repro_command:
            try:
                output = await self._run_shell(
                    repro_command,
                    repo_root,
                    self._agent_config.verification.timeout_ms,
                )
            except VerificationFailed as exc:
                raise VerificationFailed(  # pylint: disable=bad-exception-cause
                    "repro verification failed: the reproduction command still exits non-zero after the fix",
                    exc.output,
                ) from exc
            outputs.append(f"## repro\n$ {repro_command}\n{output}".strip())
            # A green reproduction is an executable verification of the
            # reported bug even when the repository has no conventional
            # test suite for the fallback regression guard to discover.
            # Keep the guard's note in verification_output, but do not
            # downgrade the successful repro contract to skipped_no_tests.
            if verification_status == "skipped_no_tests":
                verification_status = "passed"
        hook_command = self._hooks_config.pre_push
        if hook_command:
            before = await asyncio.to_thread(self._status_snapshot, repo_root)
            try:
                output = await self._run_shell(
                    hook_command,
                    repo_root,
                    self._hooks_config.timeout_ms,
                )
            except VerificationFailed as exc:
                raise HookFailedError("pre_push", "pre_push hook failed", exc.output) from exc  # pylint: disable=bad-exception-cause
            if await asyncio.to_thread(self._status_snapshot, repo_root) != before:
                raise HookFailedError(
                    "pre_push",
                    "pre_push hook modified the workspace",
                    output,
                )
            outputs.append(f"## pre_push\n{output}".strip())
        setattr(session, "verification_status", verification_status)
        setattr(session, "verification_output", "\n\n".join(outputs))

    # ------------------------------------------------------------------
    # Regression guard (defect R1)
    # ------------------------------------------------------------------

    # Short-summary lines emitted by ``pytest -q``:
    #   FAILED tests/test_x.py::test_y[case] - AssertionError: ...
    #   ERROR tests/test_z.py - ImportError: ...
    _PYTEST_FAILURE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)

    async def _run_regression_guard(self, repo_root: str, session: Any) -> tuple[str, str]:
        """Run the fallback test suite and gate on **net-new** failures.

        Returns ``(verification_status, output_note)``. Statuses:

        - ``passed`` — suite is green after the change.
        - ``passed_preexisting_failures`` — suite is red, but every
          failure already fails at the session's start commit; the
          change introduced nothing new. Running the *same command in
          the same environment* on both sides is what makes the
          comparison honest — environment quirks fail identically on
          both sides and cancel out.
        - ``skipped_no_tests`` — no test suite detected (or the runner
          itself is unavailable). Deliberately NOT reported as
          ``passed``: reviewers see that nothing was verified.

        Raises :class:`VerificationFailed` when the change introduces
        failures that the baseline does not have.
        """
        command = self._detect_fallback_test_command(repo_root)
        if not command:
            logger.info("regression guard: no test suite detected in %s", repo_root)
            return (
                "skipped_no_tests",
                "no test suite detected — verification did not run",
            )
        timeout_ms = self._agent_config.verification.timeout_ms
        after_rc, after_output = await self._run_shell_result(command, repo_root, timeout_ms)
        if after_rc == 0:
            return ("passed", f"$ {command}\n{_tail(after_output)}")
        if self._looks_like_missing_runner(after_rc, after_output):
            logger.info(
                "regression guard: test runner unavailable (rc=%s) in %s",
                after_rc,
                repo_root,
            )
            return (
                "skipped_no_tests",
                f"test runner unavailable (rc={after_rc}) — verification did not run",
            )
        after_failures = set(self._PYTEST_FAILURE_RE.findall(after_output))
        baseline_failures = await self._baseline_failures(repo_root, session, command)
        if baseline_failures is not None and after_failures:
            net_new = sorted(after_failures - baseline_failures)
            if not net_new:
                note = (
                    f"$ {command}\n"
                    f"{len(after_failures)} failing test(s), all of which already "
                    f"fail at the session start commit — no regression introduced.\n"
                    f"{_tail(after_output)}"
                )
                return ("passed_preexisting_failures", note)
            listed = "\n".join(f"- {item}" for item in net_new[:50])
            raise VerificationFailed(
                f"regression guard: {len(net_new)} net-new failing test(s) introduced by this change",
                f"$ {command}\n\nNet-new failures:\n{listed}\n\n{_tail(after_output)}",
            )
        # No baseline to compare against (missing start sha, worktree
        # failure, or the failure list could not be parsed). Be
        # conservative: a red suite blocks the push.
        raise VerificationFailed(
            f"regression guard: test suite failed (rc={after_rc}) and no baseline was available for comparison",
            f"$ {command}\n{_tail(after_output)}",
        )

    def _detect_fallback_test_command(self, repo_root: str) -> str:
        """Pick the fallback test command for the workspace.

        Explicit ``verification.fallback_test_command`` wins. Otherwise
        detect a pytest suite (``pytest.ini`` / ``tests|test`` directory
        containing ``test_*.py`` / ``*_test.py``). Returns ``""`` when
        nothing is detected.
        """
        explicit = self._agent_config.verification.fallback_test_command
        if explicit:
            return explicit
        root = Path(repo_root)
        has_pytest_marker = (root / "pytest.ini").is_file()
        if not has_pytest_marker:
            for tests_dir in ("tests", "test"):
                candidate = root / tests_dir
                if not candidate.is_dir():
                    continue
                try:
                    has_pytest_marker = any(candidate.rglob("test_*.py")) or any(candidate.rglob("*_test.py"))
                except OSError:
                    has_pytest_marker = False
                if has_pytest_marker:
                    break
        if not has_pytest_marker:
            return ""
        python = resolve_python_executable(
            workspace_path=root,
            agent_cfg=self._agent_config,
            workspace_cfg=None,
        )
        interpreter = python or "python3"
        return f'"{interpreter}" -m pytest -q --color=no -p no:cacheprovider'

    @staticmethod
    def _looks_like_missing_runner(rc: int, output: str) -> bool:
        """True when the failure means "pytest isn't usable here", not
        "tests failed" — rc 127 (command not found), rc 5 (no tests
        collected) or the interpreter reporting the module is absent.
        """
        if rc in (5, 127):
            return True
        lowered = output.lower()
        return "no module named pytest" in lowered or "not recognized as" in lowered

    async def _baseline_failures(self, repo_root: str, session: Any, command: str) -> set[str] | None:
        """Run ``command`` against the session's start commit in a
        temporary worktree and return its failing-test set.

        ``None`` means "baseline unavailable" (no start sha recorded, or
        the worktree could not be created) — the caller must then treat
        every current failure as blocking.
        """
        start_sha = getattr(session, "start_commit_sha", None)
        if not start_sha:
            return None
        tmp_dir = tempfile.mkdtemp(prefix="clawcodex-baseline-")
        added = False
        try:
            _, err, rc = await asyncio.to_thread(
                _run_git,
                ["worktree", "add", "--detach", tmp_dir, str(start_sha)],
                repo_root,
            )
            if rc != 0:
                logger.warning("regression guard: baseline worktree failed: %s", err)
                return None
            added = True
            baseline_rc, baseline_output = await self._run_shell_result(
                command,
                tmp_dir,
                self._agent_config.verification.timeout_ms,
            )
            if baseline_rc == 0:
                return set()
            return set(self._PYTEST_FAILURE_RE.findall(baseline_output))
        except VerificationFailed:
            # Baseline run timed out — treat as unavailable rather than
            # letting a slow baseline mask the after-run result.
            logger.warning("regression guard: baseline run timed out")
            return None
        finally:
            if added:
                await asyncio.to_thread(
                    _run_git,
                    ["worktree", "remove", "--force", tmp_dir],
                    repo_root,
                )

    async def _run_post_sync_hook(self, repo_root: str, session: Any) -> None:
        command = self._hooks_config.post_sync
        if not command:
            return
        before = await asyncio.to_thread(self._status_snapshot, repo_root)
        try:
            output = await self._run_shell(command, repo_root, self._hooks_config.timeout_ms)
        except VerificationFailed as exc:
            raise HookFailedError("post_sync", "post_sync hook failed", exc.output) from exc  # pylint: disable=bad-exception-cause
        if await asyncio.to_thread(self._status_snapshot, repo_root) != before:
            raise HookFailedError(
                "post_sync",
                "post_sync hook modified the workspace",
                output,
            )
        setattr(session, "post_sync_output", output)

    async def _run_shell(self, command: str, repo_root: str, timeout_ms: int) -> str:
        rc, output = await self._run_shell_result(command, repo_root, timeout_ms)
        if rc != 0:
            raise VerificationFailed(
                f"command failed with exit code {rc}: {command}",
                output,
            )
        return output

    async def _run_shell_result(self, command: str, repo_root: str, timeout_ms: int) -> tuple[int, str]:
        """Like ``_run_shell`` but reports a non-zero exit code instead of
        raising, so callers that need to interpret the code (the
        regression guard) can. A timeout still raises.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError as exc:
            raise VerificationFailed(
                f"command timed out after {timeout_ms}ms: {command}",
                "",
            ) from exc
        output = "\n".join(part.decode("utf-8", errors="replace").strip() for part in (stdout, stderr) if part).strip()
        return proc.returncode or 0, output

    def _status_snapshot(self, repo_root: str) -> str:
        return "\n".join(sorted(s.path for s in get_file_status(repo_root)))

    def _sync_gitignore(self, repo_root: str) -> None:
        self._sync_git_exclude(repo_root)

    def _sync_git_exclude(self, repo_root: str) -> None:
        exclude_path = Path(repo_root) / ".git" / "info" / "exclude"
        self._append_ignore_patterns(exclude_path)

    def _append_ignore_patterns(self, path: Path) -> None:
        existing: set[str] = set()
        if path.exists():
            existing = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
        new_patterns = [pattern for pattern in self._gitignore_patterns if pattern not in existing]
        if not new_patterns:
            return
        with path.open("a", encoding="utf-8") as handle:
            if path.exists() and path.stat().st_size > 0:
                handle.write("\n")
            handle.write("# ClawCodeX managed — do not edit manually\n")
            for pattern in new_patterns:
                handle.write(f"{pattern}\n")


# Re-exported for backward compatibility — these symbols moved to
# git_sync_rebase.py but external code still imports them from git_sync.
from .git_sync_rebase import PRRebaseResult, _ahead_behind, _git_rebase_abort, rebase_for_pr  # noqa: F401,E402  # pylint: disable=import-error,no-name-in-module
