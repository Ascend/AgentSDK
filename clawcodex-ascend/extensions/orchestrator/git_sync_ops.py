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

"""``GitSyncOpsMixin`` — commit/push/PR/report operations for ``GitSyncService``.

Extracted from ``git_sync.py`` to keep ``GitSyncService`` under 1k
lines.  These methods are mixed into ``GitSyncService`` and access
instance attributes (``self.tracker``, ``self._agent_config``,
``self._pr_template``, etc.) set by the service's ``__init__``.

Dependency graph::

    git_sync_rebase  (GitSyncError, _git_rebase_abort, _slugify)
        ↑
    git_sync_ops     (this module — imports from git_sync_rebase)
        ↑
    git_sync         (imports GitSyncOpsMixin)
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from extensions.orchestrator_runtime.adapters.clawcodex_compat import (  # pylint: disable=import-error,no-name-in-module
    _run_git,
    get_current_branch,
)

from . import report_writer
from .git_sync_rebase import GitSyncError, _git_rebase_abort, _slugify  # pylint: disable=import-error,no-name-in-module
from .issue import Issue  # pylint: disable=import-error
from .tracker import PullRequestRef

logger = logging.getLogger(__name__)


class GitSyncOpsMixin:
    """Commit/push/PR/report operations mixed into ``GitSyncService``.

    All methods access instance attributes via ``self.*``, expecting
    ``GitSyncService.__init__`` to have set them.  This is a pure
    mixin — it does not define ``__init__``.
    """

    _ORCHESTRATOR_ARTIFACTS: tuple[str, ...] = (
        ".orchestrator_control",
        ".orchestrator_workspace",
        ".reports",
        ".operator_hints.md",
        ".clawcodex_issue_registry.json",
        ".clawcodex_clarification_queue.json",
        ".clawcodex_workspace.lock",
        ".event_streams",
        "daemon.pid",
        "analysis.md",
        "changes_summary.md",
        "verification_report.md",
    )

    _WORKFLOW_ARTIFACT_PATTERNS: tuple[str, ...] = (
        "ANALYSE_REPORT",
        "ANALYSIS_REPORT",
        "CHANGE_SUMMARY",
        "WORKFLOW_REPORT",
        "STAGE_REPORT",
    )

    def _push_directly(
        self,
        repo_root: str,
        branch_name: str,
    ) -> tuple[bool, bool, tuple[str, ...]]:
        """Push branch directly without rebase — used for followup PRs."""
        try:
            self._run_git_checked(
                ["fetch", "origin", branch_name],
                repo_root,
            )
        except Exception:
            pass  # nosec B110
        try:
            self._run_git_checked(
                ["push", "origin", branch_name],
                repo_root,
            )
        except Exception:
            return False, False, ()
        return True, False, ()

    def _push_with_recovery(
        self,
        repo_root: str,
        branch_name: str,
    ) -> tuple[bool, bool, tuple[str, ...]]:
        """Push branch, recovering from non-fast-forward with rebase."""
        stdout, stderr, rc = _run_git(
            ["push", "-u", "origin", branch_name],
            repo_root,
        )
        if rc == 0:
            return True, False, ()

        if not self._is_non_fast_forward(stderr):
            raise GitSyncError(f"git push failed: {stderr or stdout}")

        # Attempt fetch + rebase
        self._run_git_checked(["fetch", "origin"], repo_root)
        # Defensively clear any leftover REBASE_HEAD before
        # starting a fresh rebase — if a previous run aborted mid-
        # rebase, this prevents compounding conflict markers.
        _git_rebase_abort(repo_root)
        stdout, stderr, rc = _run_git(
            ["rebase", f"origin/{branch_name}"],
            repo_root,
        )
        if rc != 0:
            # Check if remote branch doesn't exist (shallow clone scenario)
            if "fatal: invalid upstream" in stderr or "couldn't find remote ref" in stderr:
                # Remote branch doesn't exist - force push to create it
                self._run_git_checked(["push", "-u", "origin", branch_name, "--force"], repo_root)
                return True, False, ()
            conflict_files = self._detect_conflicts(repo_root)
            if conflict_files:
                # Leave the half-finished rebase in place so
                # the follow-up agent run can resume with
                # ``git rebase --continue`` after resolving the
                # conflict markers.
                return False, True, conflict_files
            # Non-conflict rebase failure (auth / network) —
            # abort the half-finished rebase so the workspace
            # doesn't stay stuck in REBASE_HEAD.
            _git_rebase_abort(repo_root)
            raise GitSyncError(f"git rebase failed: {stderr or stdout}")

        # Retry push after successful rebase
        self._run_git_checked(["push", "-u", "origin", branch_name], repo_root)
        return True, False, ()

    def _is_non_fast_forward(self, stderr: str) -> bool:
        if not stderr:
            return False
        return (
            "non-fast-forward" in stderr.lower()
            or "fetch first" in stderr.lower()
            or "Updates were rejected" in stderr
            or "shallow update" in stderr.lower()
            or "deny updating a hidden branch" in stderr.lower()
        )

    def _detect_conflicts(self, repo_root: str) -> tuple[str, ...]:
        """Return list of files with conflict markers."""
        stdout, _, _ = _run_git(
            ["diff", "--name-only", "--diff-filter=U"],
            repo_root,
        )
        if not stdout.strip():
            return ()
        return tuple(f.strip() for f in stdout.strip().splitlines() if f.strip())

    def _ensure_work_branch(
        self,
        repo_root: str,
        issue: Issue,
        base_branch: str,
    ) -> str:
        current_branch = get_current_branch(repo_root)
        branch_name = issue.branch_name or self._default_branch_name(issue)

        if current_branch == branch_name:
            return branch_name
        if current_branch and current_branch != "HEAD" and current_branch != base_branch:
            return current_branch

        stdout, stderr, rc = _run_git(["checkout", branch_name], repo_root)
        if rc == 0:
            return branch_name

        # Branch doesn't exist locally — determine best creation strategy
        # Case 1: remote branch exists → checkout with --track to wire it to origin
        # Case 2: completely new branch → create from upstream/base (fork mode) or locally
        remote_ref = f"origin/{branch_name}"
        check_remote = self._run_git_output(["rev-parse", "--verify", f"refs/remotes/{remote_ref}"], repo_root)
        if check_remote:
            # Remote branch exists — wire it up with --track
            stdout, stderr, rc = _run_git(
                ["checkout", "--track", remote_ref],
                repo_root,
            )
        elif self._fork_mode():
            # Fork workflow: create new branch from upstream/base_branch
            stdout, stderr, rc = _run_git(
                ["checkout", "-b", branch_name, f"upstream/{base_branch}"],
                repo_root,
            )
        else:
            # No remote branch → create new local branch
            stdout, stderr, rc = _run_git(
                ["checkout", "-b", branch_name],
                repo_root,
            )
        if rc != 0:
            raise GitSyncError(f"Failed to checkout work branch {branch_name}: {stderr or stdout}")
        return branch_name

    def _ensure_commit_identity(self, repo_root: str) -> None:
        if self._git_email:
            self._run_git_checked(["config", "user.email", self._git_email], repo_root)
        elif self._git_username:
            self._run_git_checked(
                ["config", "user.email", f"{self._git_username}@gitcode.com"],
                repo_root,
            )
        else:
            self._run_git_checked(
                ["config", "user.email", "clawcodex-bot@local.invalid"],
                repo_root,
            )
        if self._git_username:
            self._run_git_checked(["config", "user.name", self._git_username], repo_root)
        else:
            self._run_git_checked(["config", "user.name", "ClawCodex Bot"], repo_root)

    def _unstage_orchestrator_artifacts(self, repo_root: str) -> None:
        """Remove orchestrator-internal files from the staging area.

        Safety net: even if ``.git/info/exclude`` patterns are bypassed
        (e.g. the agent overwrites the exclude file), these files will
        never enter a commit.

        Also removes workflow-generated report files (e.g. ANALYSE_REPORT.md)
        that the agent may have created during stage execution. These are
        analysis artifacts, not code changes.
        """
        stdout, _, rc = _run_git(["diff", "--cached", "--name-only"], repo_root)
        if rc != 0 or not stdout.strip():
            return
        staged = {f.strip() for f in stdout.strip().splitlines() if f.strip()}
        to_unstage: list[str] = []
        for path in staged:
            for artifact in self._ORCHESTRATOR_ARTIFACTS:
                if path == artifact or path.startswith(f"{artifact}/"):
                    to_unstage.append(path)
                    break
            else:
                basename = Path(path).stem.upper()
                for pattern in self._WORKFLOW_ARTIFACT_PATTERNS:
                    if pattern in basename:
                        to_unstage.append(path)
                        break
        if to_unstage:
            self._run_git_checked(["reset", "--", *to_unstage], repo_root)

    def _apply_file_whitelist(self, repo_root: str) -> None:
        """Unstage files outside the allowed whitelist before commit.

        When ``agent.allowed_changed_files`` is configured, only the
        specified glob patterns may enter the commit.  Any other staged
        file is reset to unstaged (``git reset -- <path>``).  If all
        files are filtered out the commit is still attempted — it will
        simply produce no commit (no staged changes), which the caller
        already handles gracefully.
        """
        whitelist = self._agent_config.allowed_changed_files
        if not whitelist:
            return
        import fnmatch

        stdout, _, rc = _run_git(["diff", "--cached", "--name-only"], repo_root)
        if rc != 0 or not stdout.strip():
            return
        staged = [f.strip() for f in stdout.strip().splitlines() if f.strip()]
        to_unstage = [f for f in staged if not any(fnmatch.fnmatch(f, pat) for pat in whitelist)]
        if to_unstage:
            self._run_git_checked(["reset", "--", *to_unstage], repo_root)

    def _build_commit_message(
        self,
        issue: Issue,
        *,
        followup: bool = False,
        feedback_body: str | None = None,
        session: Any | None = None,
    ) -> str:
        identifier = (issue.identifier or "issue").strip().lstrip("#")
        prefix = "fix" if followup else "feat"
        if followup and feedback_body:
            # Use the review comment as the commit title
            title = feedback_body.strip()[:72]
        else:
            title = (issue.title or "automated update").strip()
        message = f"{prefix}: {identifier} {title}"

        # Append review metadata for later rules extraction.
        if followup and session is not None:
            pr_ref = getattr(session, "pull_request", None)
            pr_num = getattr(pr_ref, "number", None) or getattr(pr_ref, "id", "")
            lines = [message, ""]
            if pr_num:
                lines.append(f"review-pr: #{pr_num}")
            feedback_ids = getattr(session, "feedback_ids", None) or []
            for fid in feedback_ids:
                lines.append(f"review-id: {fid}")
            feedback_body = (getattr(session, "feedback_commit_body", None) or "").strip()
            if feedback_body:
                lines.append(f"review-body: {feedback_body}")
            if len(lines) > 2:
                message = "\n".join(lines)
        return message[:1024] if followup else message[:72]

    def _ensure_review_metadata(self, repo_root: str, session: Any, followup_pr: Any) -> None:
        """Amend agent's commit to add review metadata if missing (safe before push)."""
        current_msg = self._run_git_output(["log", "-1", "--format=%B"], repo_root)
        if "review-pr:" in current_msg:
            return
        pr_num = getattr(followup_pr, "number", None) or getattr(followup_pr, "id", "")
        feedback_body = (getattr(session, "feedback_commit_body", None) or "").strip()
        lines = [current_msg.strip(), "", f"review-pr: #{pr_num}"]
        feedback_ids = getattr(session, "feedback_ids", None) or []
        for fid in feedback_ids:
            lines.append(f"review-id: {fid}")
        if feedback_body:
            lines.append(f"review-body: {feedback_body}")
        new_msg = "\n".join(lines)
        self._run_git_checked(["commit", "--amend", "-m", new_msg], repo_root)
        logger.info(
            "Amended commit with review metadata (PR=%s, body=%s)",
            pr_num,
            feedback_body[:40],
        )

    def _build_pr_title(self, issue: Issue) -> str:
        if self._pr_template.title:
            title = (
                self._render_pr_template(
                    self._pr_template.title,
                    self._pr_template_context(issue=issue),
                )
                .replace("\n", " ")
                .strip()
            )
            if title:
                return title
        identifier = (issue.identifier or "issue").strip()
        title = (issue.title or "Automated update").strip()
        return f"{identifier}: {title}"

    def _build_pr_body(
        self,
        issue: Issue,
        commit_sha: str | None,
        branch_name: str,
        base_branch: str,
        *,
        session: Any,
        pull_request: PullRequestRef | None,
    ) -> str:
        if self._pr_template.body:
            return self._render_pr_template(
                self._pr_template.body,
                self._pr_template_context(
                    issue=issue,
                    commit_sha=commit_sha,
                    branch_name=branch_name,
                    base_branch=base_branch,
                    session=session,
                    pull_request=pull_request,
                ),
            ).strip()

        report_path = getattr(session, "report_path", None)
        verification_status = getattr(session, "verification_status", None) or "skipped"
        workspace_path = getattr(session.workspace, "path", None) if hasattr(session, "workspace") else None
        lines = [
            "## ClawCodex Automated Change",
            "",
            f"- Issue: {issue.identifier or issue.id or 'unknown'}",
            f"- Branch: `{branch_name}`",
            f"- Base: `{base_branch}`",
            f"- Commit: `{commit_sha or 'n/a'}`",
            f"- Verification: `{verification_status}`",
            f"- Report: `{report_path or 'n/a'}`",
        ]
        # Add workspace path if available (useful for manual verification)
        if workspace_path:
            lines.append(f"- Workspace: `{workspace_path}`")
        if issue.url:
            lines.append(f"- Source issue: {issue.url}")
        if pull_request and pull_request.url:
            lines.append(f"- Pull request: {pull_request.url}")

        # Read agent's commit message for e2e verification results
        if commit_sha:
            try:
                workspace_path = getattr(session.workspace, "path", None)
                if workspace_path:
                    commit_msg = self._run_git_output(
                        ["log", "-1", "--format=%B", commit_sha],
                        str(workspace_path),
                    )
                    if commit_msg and commit_msg.strip():
                        lines.extend(["", "---", ""])
                        e2e_section = self._extract_section(commit_msg, "E2E Verification")
                        changes_section = self._extract_section(commit_msg, "Changes")

                        if changes_section:
                            lines.extend(["## Changes", "", changes_section])
                        if e2e_section:
                            lines.extend(["", "## E2E Verification", "", e2e_section])

                        if not e2e_section and not changes_section:
                            lines.extend(["## Agent Notes", "", commit_msg.strip()])
            except Exception:
                pass  # nosec B110

        # Include regression test output summary
        verification_output = getattr(session, "verification_output", None)
        if verification_output:
            summary_lines = [
                line for line in verification_output.strip().splitlines() if "passed" in line or "failed" in line
            ]
            if summary_lines:
                lines.extend(["", "## Regression Tests", "", "```", summary_lines[-1], "```"])

        # Include workflow stage outputs (analysis, implementation notes, etc.)
        workspace_path = getattr(session.workspace, "path", None)

        # Read analysis.md (Stage 1 output) for PR body
        if workspace_path:
            analysis_file = Path(workspace_path) / "analysis.md"
            if analysis_file.exists():
                try:
                    analysis = analysis_file.read_text(encoding="utf-8")
                    if analysis.strip():
                        lines.extend(["", "## Analysis", "", analysis.strip()])
                except Exception:
                    pass  # nosec B110

        # Prefer changes_summary.md over raw stage outputs for clean PR body.
        changes_summary_text = None
        if workspace_path:
            summary_file = Path(workspace_path) / "changes_summary.md"
            if summary_file.exists():
                try:
                    raw = summary_file.read_text(encoding="utf-8")
                    if raw.strip():
                        changes_summary_text = self._strip_think_blocks(raw).strip()
                except Exception:
                    pass  # nosec B110

        # Read verification_report.md (Stage 3 output) if available
        if workspace_path:
            verify_file = Path(workspace_path) / "verification_report.md"
            if verify_file.exists():
                try:
                    verify_text = verify_file.read_text(encoding="utf-8").strip()
                    if verify_text:
                        lines.extend(["", verify_text])
                except Exception:
                    pass  # nosec B110

        if changes_summary_text:
            lines.extend(["", "## Changes", "", changes_summary_text])
        else:
            workflow_outputs = getattr(session, "workflow_stage_outputs", None)
            if workflow_outputs:
                for stage_id in sorted(workflow_outputs.keys()):
                    if stage_id == 1:  # skip raw analysis conversation
                        continue
                    info = workflow_outputs[stage_id]
                    output = self._strip_think_blocks(info.get("output", "").strip())
                    if output:
                        name = info.get("name", f"Stage {stage_id}")
                        lines.extend(["", f"## {name}", ""])
                        if len(output) > 3000:
                            lines.append(output[:3000])
                            lines.append("\n... (truncated)")
                        else:
                            lines.append(output)

        if report_path:
            lines.extend(["", f"<!-- metadata: report_path={report_path} -->"])
        return "\n".join(lines)

    def _pr_template_context(
        self,
        *,
        issue: Issue,
        commit_sha: str | None = None,
        branch_name: str = "",
        base_branch: str = "",
        session: Any | None = None,
        pull_request: PullRequestRef | None = None,
    ) -> dict[str, str]:
        """Return the safe, data-only variables exposed to PR templates."""
        workspace_path = getattr(getattr(session, "workspace", None), "path", None)
        changes_summary = self._read_pr_artifact(workspace_path, "changes_summary.md")
        implementation_notes = self._read_pr_artifact(workspace_path, "implementation_notes.md")
        verification_report = self._read_pr_artifact(workspace_path, "verification_report.md")
        verification_status = getattr(session, "verification_status", None) or "skipped"
        verification_output = getattr(session, "verification_output", None) or ""
        verification_summary = verification_report or self._verification_summary(verification_output)
        return {
            "issue.id": str(issue.id or ""),
            "issue.identifier": str(issue.identifier or ""),
            "issue.title": str(issue.title or ""),
            "issue.url": str(issue.url or ""),
            "branch_name": branch_name,
            "base_branch": base_branch,
            "commit_sha": commit_sha or "",
            "verification_status": str(verification_status),
            "verification_summary": verification_summary,
            "changes_summary": changes_summary,
            "implementation_notes": implementation_notes,
            "pull_request.url": str(getattr(pull_request, "url", None) or ""),
            "pull_request.number": str(getattr(pull_request, "number", None) or ""),
        }

    @staticmethod
    def _render_pr_template(template: str, context: dict[str, str]) -> str:
        """Replace ``{{ variable }}`` tokens without evaluating template code."""
        return re.sub(
            r"{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}",
            lambda match: context.get(match.group(1), ""),
            template,
        )

    def _read_pr_artifact(self, workspace_path: str | Path | None, filename: str) -> str:
        if not workspace_path:
            return ""
        try:
            text = (Path(workspace_path) / filename).read_text(encoding="utf-8")
        except OSError:
            return ""
        return self._strip_think_blocks(text).strip()

    @staticmethod
    def _verification_summary(output: str) -> str:
        lines = [line for line in output.strip().splitlines() if "passed" in line or "failed" in line]
        return lines[-1] if lines else output.strip()

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """Remove <think>...</think> blocks from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str | None:
        """Extract a named section from a structured commit message.

        Looks for `## Section Name` followed by content until the next `##` or EOF.
        """
        pattern = rf"## {re.escape(section_name)}\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _write_report(
        self,
        *,
        session: Any,
        branch_name: str,
        base_branch: str,
        commit_sha: str | None,
        pull_request: PullRequestRef | None,
    ) -> report_writer.ReportResult | None:
        run_id = getattr(session, "run_id", None)
        workspace = getattr(session, "workspace", None)
        issue = getattr(session, "issue", None)
        if not run_id or workspace is None or issue is None:
            return None
        result = report_writer.write(
            run_id=run_id,
            workspace_path=Path(workspace.path),
            tracker=getattr(self.tracker, "platform", self.tracker.__class__.__name__),
            owner=getattr(self.tracker, "owner", None),
            repo=getattr(self.tracker, "repo", None),
            issue=issue,
            status=getattr(session, "status", "unknown"),
            branch_name=branch_name,
            base_branch=base_branch,
            commit_sha=commit_sha,
            pr_number=str(pull_request.number) if pull_request and pull_request.number is not None else None,
            pr_url=pull_request.url if pull_request else None,
            turn_count=getattr(session, "turn_count", 0),
            tool_count=getattr(session, "tool_count", 0),
            verification_status=getattr(session, "verification_status", None),
            verification_output=getattr(session, "verification_output", None),
            output_text=getattr(session, "output_text", ""),
            # Forward the per-tool audit log path so report_writer
            # can dual-write the NDJSON into the persistent layer.
            tool_events_path=getattr(session, "tool_events_path", None),
        )
        setattr(session, "report_path", result.persistent_markdown_path)
        return result

    async def _update_summary_comment(
        self,
        *,
        session: Any,
        branch_name: str,
        base_branch: str,
        commit_sha: str | None,
        pull_request: PullRequestRef | None,
        committed: bool,
        pushed: bool,
        report_path: str | None,
    ) -> None:
        issue = session.issue
        if not issue.id:
            return

        body = self._build_summary_comment_body(
            session=session,
            branch_name=branch_name,
            base_branch=base_branch,
            commit_sha=commit_sha,
            pull_request=pull_request,
            committed=committed,
            pushed=pushed,
            report_path=report_path,
        )
        comment_id = getattr(session, "summary_comment_id", None)
        if comment_id:
            updated = await self.tracker.update_comment(issue.id, comment_id, body)
            if updated is not None:
                return
        created = await self.tracker.create_comment(issue.id, body)
        if created is not None and getattr(created, "id", None):
            setattr(session, "summary_comment_id", created.id)

    def _build_summary_comment_body(
        self,
        *,
        session: Any,
        branch_name: str,
        base_branch: str,
        commit_sha: str | None,
        pull_request: PullRequestRef | None,
        committed: bool,
        pushed: bool,
        report_path: str | None,
    ) -> str:
        verification_status = getattr(session, "verification_status", None) or "skipped"
        body_lines = [
            "## ClawCodex Run Summary",
            "",
            f"- Run: `{getattr(session, 'run_id', 'unknown')}`",
            f"- Status: `{getattr(session, 'status', 'unknown')}`",
            f"- Branch: `{branch_name}`",
            f"- Base: `{base_branch}`",
            f"- Committed: {'yes' if committed else 'no'}",
            f"- Pushed: {'yes' if pushed else 'no'}",
            f"- Verification: `{verification_status}`",
            f"- Report: `{report_path or 'n/a'}`",
        ]
        if commit_sha:
            body_lines.append(f"- Commit: `{commit_sha}`")
        if pull_request and pull_request.url:
            body_lines.append(f"- Pull request: {pull_request.url}")
        if report_path:
            body_lines.extend(["", f"<!-- metadata: report_path={report_path} -->"])
        return "\n".join(body_lines)

    def _default_branch_name(self, issue: Issue) -> str:
        identifier = issue.identifier or issue.id or "issue"
        title = issue.title or "update"
        slug = _slugify(f"{identifier}-{title}")[:48]
        prefix = self._branch_prefix or "clawcodex"
        return f"{prefix}/{slug}"

    def _merge_pr_ref(
        self,
        updated: PullRequestRef,
        existing: PullRequestRef,
    ) -> PullRequestRef:
        return PullRequestRef(
            number=updated.number or existing.number,
            url=updated.url or existing.url,
            title=updated.title or existing.title,
        )

    def _run_git_output(self, args: list[str], repo_root: str) -> str:
        stdout, stderr, rc = _run_git(args, repo_root)
        if rc != 0:
            return ""
        return stdout.strip()

    def _run_git_checked(self, args: list[str], repo_root: str) -> str:
        stdout, stderr, rc = _run_git(args, repo_root)
        if rc != 0:
            raise GitSyncError(f"git {' '.join(args)} failed: {stderr or stdout}")
        return stdout.strip()

    def _has_staged_changes(self, repo_root: str) -> bool:
        """Check if there are staged changes ready to commit.

        Returns True if `git diff --cached --quiet` exits with non-zero
        (meaning there are staged changes), False otherwise.
        """
        _, _, rc = _run_git(["diff", "--cached", "--quiet"], repo_root)
        # rc=0 means no staged changes, rc=1 means there are staged changes
        return rc != 0

    async def _find_pr_fallback(
        self,
        pr_ref: PullRequestRef,
        *,
        head_branch: str,
        base_branch: str,
    ) -> PullRequestRef:
        """Find a just-created PR when the initial response lacks number/url.

        Some trackers (notably GitCode) return a pull-request object where
        ``number`` and ``url`` are empty right after creation.  This method
        polls the tracker's open-PR list and matches by ``head_branch``.
        """
        for _ in range(15):
            try:
                found = await self.tracker.find_pull_request(
                    head_branch=head_branch,
                    base_branch=base_branch,
                )
            except Exception:
                found = None
            if found is not None and (found.number or found.url):
                return self._merge_pr_ref(found, pr_ref)

            try:
                open_prs = await self.tracker.list_pull_requests(
                    state="open",
                    head=head_branch,
                )
            except (TypeError, AttributeError):
                # Tracker doesn't support head filtering — try unfiltered.
                try:
                    open_prs = await self.tracker.list_pull_requests(state="open")
                except Exception:
                    return pr_ref
            except Exception:
                return pr_ref
            if open_prs:
                for candidate in open_prs:
                    candidate_head = (
                        getattr(candidate, "head_ref", None)
                        or getattr(candidate, "head_branch", None)
                        or getattr(candidate, "source_branch", None)
                        or ""
                    )
                    if candidate_head == head_branch:
                        return candidate
            await asyncio.sleep(2)
        return pr_ref
