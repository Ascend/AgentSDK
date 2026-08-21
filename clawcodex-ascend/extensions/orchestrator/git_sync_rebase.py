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

"""PR conflict auto-resolution — rebase a stale-base feature branch.

Extracted from ``git_sync.py`` so the rebase flow (which is a set of
free functions, not ``GitSyncService`` methods) lives in its own
module.  ``GitSyncError`` is defined here because it is the base
exception needed by both ``rebase_for_pr`` (this module) and the
exception subclasses in ``git_sync.py`` — placing it here keeps the
dependency graph acyclic:

    git_sync_rebase  (no internal deps)
        ↑
    git_sync_ops     (imports from git_sync_rebase)
        ↑
    git_sync         (imports from both)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from extensions.orchestrator_runtime.adapters.clawcodex_compat import (  # pylint: disable=import-error,no-name-in-module
    _run_git,
    get_current_branch,
)


class GitSyncError(RuntimeError):
    """Raised when post-run git sync fails."""


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._/-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "issue-update"


# ---------------------------------------------------------------------------
# PR conflict auto-resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRRebaseResult:
    """Result of `rebase_for_pr`.

    Attributes:
      * ``rebased`` — ``True`` if the local branch is now based on
        the latest base. ``False`` if the rebase did not happen (no
        work to do, push failed, or pre-flight check rejected).
      * ``has_conflict`` — ``True`` if the rebase left content
        conflicts in the workspace. The orchestrator's daemon
        scan path uses this to schedule a follow-up agent run
        (``run_kind="agent_rebase"``).
      * ``conflict_files`` — list of files containing conflict
        markers (from ``git diff --name-only --diff-filter=U``).
        Empty tuple when ``has_conflict`` is False.
      * ``new_head_sha`` — the local commit SHA after a successful
        rebase+push. ``None`` when no push happened.
      * ``pushed`` — ``True`` if the rebased branch was pushed to
        the remote. ``False`` when no rebase was needed (already
        up-to-date) or the push failed.
      * ``push_method`` — one of ``"force_with_lease"`` /
        ``"force"`` / ``"none"``. Lets the audit log distinguish
        operator-explicit ``--force`` runs from the default
        safe-with-lease path.
      * ``workspace_clean`` — ``True`` when no
        ``.git/REBASE_HEAD`` is left behind. ``False`` indicates
        the operator should run ``git rebase --abort`` manually
        (defensive cleanup paths in ``rebase_for_pr`` are
        best-effort).
    """

    rebased: bool
    has_conflict: bool = False
    conflict_files: tuple[str, ...] = field(default_factory=tuple)
    new_head_sha: str | None = None
    pushed: bool = False
    push_method: str = "none"  # "force_with_lease" | "force" | "none"
    workspace_clean: bool = True


def _git_rebase_abort(repo_root: str) -> None:
    """Best-effort ``git rebase --abort``.

    Used by ``rebase_for_pr`` to clear a stuck rebase state when
    pre-flight checks fail (e.g. fetch returned 0 commits, or the
    rebase exited with a non-conflict error like auth failure). The
    command is allowed to fail silently — when no rebase is in
    progress ``git rebase --abort`` returns a non-zero exit code
    with a "No rebase in progress?" message; we don't want that to
    raise and mask the real error.
    """
    stdout, stderr, _rc = _run_git(["rebase", "--abort"], repo_root)
    # ``_run_git`` does not raise on non-zero rc; we explicitly
    # ignore the result.
    del stdout, stderr


def _ahead_behind(repo_root: str, branch: str, base: str) -> tuple[int, int]:
    """Return ``(ahead, behind)`` commit counts of ``branch`` vs ``base``.

    Wraps ``git rev-list --left-right --count branch...base`` and
    parses the two integers from stdout. On parse failure returns
    ``(0, 0)`` so the caller can short-circuit ("nothing to do")
    rather than crash.
    """
    stdout, _stderr, rc = _run_git(
        ["rev-list", "--left-right", "--count", f"{branch}...{base}"],
        repo_root,
    )
    if rc != 0:
        return (0, 0)
    parts = stdout.strip().split()
    if len(parts) != 2:
        return (0, 0)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (0, 0)


def rebase_for_pr(
    *,
    workspace_path: str,
    branch_name: str,
    base_branch: str,
    force: bool = False,
) -> PRRebaseResult:
    """Resolve a stale-base PR by rebasing the feature branch.

    Flow:
      1. **Pre-flight** — ``git checkout <branch>`` (if not already
         there). Abort any leftover ``.git/REBASE_HEAD`` so the
         fresh rebase doesn't compound a half-finished one.
      2. **Fetch** — ``git fetch --prune origin <base>:<base>`` so
         ``origin/<base>`` reflects the current remote tip.
      3. **Ahead/behind check** — if ``behind_by == 0`` the branch
         is already up-to-date; return immediately (no-op success).
      4. **Rebase** — ``git rebase origin/<base>``. If the command
         exits 0 we're on a fast-forward-friendly history; on
         non-zero exit, ``_detect_conflicts`` reports which files
         have conflict markers. If the failure is non-conflict
         (auth / network), abort the rebase and return
         ``rebased=False, has_conflict=False``.
      5. **Push** — ``git push --force-with-lease=origin/<branch>:
         <remote_sha>`` by default. When ``force=True``, fall
         back to plain ``--force`` (operator-explicit override).
         Push failure rolls back via
         ``git reset --hard origin/<branch>`` so the next
         retry starts from a known good state.
      6. **Return** — a ``PRRebaseResult`` summarizing the
         outcome. The CLI / daemon converts this to audit-log
         lines and ``IssueRecord.mark_conflict`` /
         ``clear_conflict`` calls.

    This function is sync (no ``await``) because it is a thin
    wrapper around ``subprocess.run``-based ``_run_git`` calls.
    Callers in async code paths can ``await asyncio.to_thread(
    rebase_for_pr, ...)`` if they need to yield the event loop
    during long fetches.
    """
    repo_root = workspace_path
    # 0. Defensively abort any leftover .git/REBASE_HEAD BEFORE
    #    checkout, because git refuses to checkout when the index
    #    has unresolved conflicts from a previous aborted rebase.
    _git_rebase_abort(repo_root)
    current_branch = get_current_branch(repo_root)
    if current_branch != branch_name:
        co_stdout, co_stderr, co_rc = _run_git(["checkout", branch_name], repo_root)
        if co_rc != 0:
            raise GitSyncError(f"git checkout {branch_name} failed: {co_stderr or co_stdout}")
    # Best-effort: clear any REBASE_HEAD that the checkout may have
    # resurrected (e.g. via git worktree or orphaned sequencer state).
    _git_rebase_abort(repo_root)

    # 1. fetch base
    fetch_stdout, fetch_stderr, fetch_rc = _run_git(
        ["fetch", "--prune", "origin", f"{base_branch}:{base_branch}"],
        repo_root,
    )
    if fetch_rc != 0:
        # Stale workspace: leave REBASE_HEAD absent and report
        # no-op (the operator can re-run with a corrected base
        # branch or refresh the workspace manually).
        return PRRebaseResult(
            rebased=False,
            push_method="none",
            workspace_clean=True,
        )

    # 2. ahead/behind short-circuit
    ahead, behind = _ahead_behind(repo_root, branch_name, f"origin/{base_branch}")
    if behind == 0:
        # Already up-to-date — no rebase needed.
        head_stdout, _, head_rc = _run_git(["rev-parse", "HEAD"], repo_root)
        head = head_stdout.strip() if head_rc == 0 else None
        return PRRebaseResult(
            rebased=True,
            has_conflict=False,
            conflict_files=(),
            new_head_sha=head,
            pushed=False,
            push_method="none",
            workspace_clean=True,
        )

    # 3. rebase
    rebase_stdout, rebase_stderr, rebase_rc = _run_git(
        ["rebase", f"origin/{base_branch}"],
        repo_root,
    )
    if rebase_rc != 0:
        # Inline the conflict check (the upstream helper is a
        # method on GitSyncService and we are a free function).
        diff_stdout, _, _ = _run_git(
            ["diff", "--name-only", "--diff-filter=U"],
            repo_root,
        )
        conflict_files = tuple(f.strip() for f in diff_stdout.strip().splitlines() if f.strip())
        if conflict_files:
            # Leave the rebase in progress — the follow-up agent
            # run will read the conflict markers and resolve
            # them, then ``git rebase --continue`` +
            # ``git push --force-with-lease``.
            return PRRebaseResult(
                rebased=False,
                has_conflict=True,
                conflict_files=conflict_files,
                push_method="none",
                workspace_clean=False,
            )
        # Rare: rebase failed but no conflicts. Could be auth,
        # missing remote, or filesystem permission. Abort the
        # half-finished rebase and report no-op.
        _git_rebase_abort(repo_root)
        return PRRebaseResult(
            rebased=False,
            has_conflict=False,
            push_method="none",
            workspace_clean=True,
        )

    # 4. push (force-with-lease by default; --force on operator request)
    # Capture the remote SHA BEFORE pushing so --force-with-lease
    # refuses if the remote moved between fetch and push.
    remote_sha_stdout, _, remote_sha_rc = _run_git(["rev-parse", f"origin/{branch_name}"], repo_root)
    remote_sha = remote_sha_stdout.strip() if remote_sha_rc == 0 else ""
    if force:
        push_stdout, push_stderr, push_rc = _run_git(
            ["push", "--force", "origin", branch_name],
            repo_root,
        )
        push_method = "force"
    elif remote_sha:
        # NOTE: --force-with-lease uses the SHORT ref name (no
        # `origin/` prefix). `git push --force-with-lease=origin/foo:X`
        # is parsed as an extra refspec and silently downgrades to a
        # non-fast-forward rejection. The correct form is
        # `--force-with-lease=foo:<expected-sha>`.
        push_stdout, push_stderr, push_rc = _run_git(
            [
                "push",
                f"--force-with-lease={branch_name}:{remote_sha}",
                "origin",
                branch_name,
            ],
            repo_root,
        )
        push_method = "force_with_lease"
    else:
        # Remote branch doesn't exist yet — fall back to plain
        # ``push -u`` (this is a fresh branch, no history to clobber).
        push_stdout, push_stderr, push_rc = _run_git(
            ["push", "-u", "origin", branch_name],
            repo_root,
        )
        push_method = "none"

    if push_rc != 0:
        # Roll back to the pre-rebase remote tip. The local
        # working tree will be left at the rebased commits; the
        # ``git reset --hard origin/<branch>`` rewinds to the
        # remote so the next attempt starts from a known state.
        rb_stdout, rb_stderr, rb_rc = _run_git(
            ["reset", "--hard", f"origin/{branch_name}"],
            repo_root,
        )
        if rb_rc != 0:
            # Reset failed — surface as a no-op with
            # workspace_clean=False so the operator knows the
            # local tree is in an unknown state.
            return PRRebaseResult(
                rebased=False,
                has_conflict=False,
                push_method="none",
                workspace_clean=False,
            )
        return PRRebaseResult(
            rebased=False,
            has_conflict=False,
            push_method="none",
            workspace_clean=True,
        )

    new_head_stdout, _, new_head_rc = _run_git(["rev-parse", "HEAD"], repo_root)
    new_head = new_head_stdout.strip() if new_head_rc == 0 else None
    return PRRebaseResult(
        rebased=True,
        has_conflict=False,
        conflict_files=(),
        new_head_sha=new_head,
        pushed=True,
        push_method=push_method,
        workspace_clean=True,
    )
