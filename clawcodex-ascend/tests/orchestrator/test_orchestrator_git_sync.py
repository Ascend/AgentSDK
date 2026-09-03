#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Full test suite for GitSyncService (17 tests).

Unmigrated deps (clawcodex_compat, config.schema, issue, prompt_builder,
intent, tracker_kinds) are stubbed via sys.modules.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

# --- Stubs for unmigrated dependencies ---

_SKIP = {"extensions", "extensions.orchestrator"}


def _reg(n: str, **kw: object) -> None:
    try:
        import_module(n)
        return
    except ImportError:
        pass
    p = n.split(".")
    for i in range(1, len(p)):
        x = ".".join(p[:i])
        if x not in _SKIP and x not in sys.modules:
            sys.modules[x] = types.ModuleType(x)
    m = types.ModuleType(n)
    for k, v in kw.items():
        setattr(m, k, v)
    sys.modules[n] = m


def _run_git(a: list[str], cwd: str | None = None, t: float = 30.0) -> tuple[str, str, int]:
    try:
        r = subprocess.run(
            ["git", *a], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd, timeout=t
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except Exception:
        return "", "", -1


def _ok(a: list[str], cwd: str | None = None) -> str:
    s, _, r = _run_git(a, cwd)
    return s if r == 0 else ""


def _gfs(cwd: str | None = None) -> list:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd or ".",
        timeout=30.0,
    )
    return (
        [SimpleNamespace(path=line[3:]) for line in r.stdout.splitlines() if len(line) >= 3]
        if r.returncode == 0 and r.stdout.strip()
        else []
    )


_reg(
    "extensions.orchestrator_runtime.adapters.clawcodex_compat",
    _run_git=_run_git,
    get_file_status=_gfs,
    get_current_branch=lambda cwd=None: _ok(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or None,
    get_default_branch=lambda cwd=None: (
        _ok(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd).rsplit("/", 1)[-1]
        if _ok(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd)
        else "main"
    ),
    get_repo_root=lambda cwd=None: _ok(["rev-parse", "--show-toplevel"], cwd) or None,
)


class _AgentConfig:
    def __init__(
        self,
        test_command="",
        build_command="",
        lint_command="",
        review_required=False,
        verification=None,
        repro_first=None,
        allowed_changed_files=None,
    ):
        self.test_command, self.build_command, self.lint_command = test_command, build_command, lint_command
        self.review_required = review_required
        self.verification = verification or SimpleNamespace(
            timeout_ms=600_000, regression_guard=True, fallback_test_command=""
        )
        self.repro_first = repro_first or SimpleNamespace(enabled=False)
        self.allowed_changed_files = allowed_changed_files or []


class _HooksConfig:
    def __init__(self, pre_commit=None, pre_push=None, post_sync=None, timeout_ms=60_000):
        self.pre_commit, self.pre_push, self.post_sync, self.timeout_ms = pre_commit, pre_push, post_sync, timeout_ms


class _PrTemplateConfig:
    def __init__(self, title="", body=""):
        self.title, self.body = title, body


class _WorkflowConfig:
    def __init__(self) -> None:
        self.pr_template = _PrTemplateConfig()

    @classmethod
    def from_dict(cls, raw: dict) -> "_WorkflowConfig":
        w = cls()
        pt = raw.get("pr_template", {})
        w.pr_template = _PrTemplateConfig(pt.get("title", ""), pt.get("body", ""))
        return w


_reg(
    "extensions.orchestrator.config.schema",
    AgentConfig=_AgentConfig,
    HooksConfig=_HooksConfig,
    PrTemplateConfig=_PrTemplateConfig,
    WorkflowConfig=_WorkflowConfig,
)


class _Issue:
    def __init__(self, id=None, identifier=None, title=None, description=None, url=None, branch_name=None, labels=None):
        self.id, self.identifier, self.title = id, identifier, title
        self.description, self.url, self.branch_name = description, url, branch_name
        self.labels = labels or []


_reg("extensions.orchestrator.issue", Issue=_Issue)
_reg("extensions.orchestrator.prompt_builder", resolve_python_executable=lambda: sys.executable)


class _Intent(str, Enum):
    NONE = "none"
    RETRY = "retry"
    FOLLOWUP = "followup"
    BLOCKED = "blocked"
    REBASE = "rebase"


class _Command(str, Enum):
    RETRY = "retry"
    FOLLOWUP = "followup"
    BLOCKED = "blocked"
    REBASE = "rebase"


_reg(
    "extensions.orchestrator.intent",
    Intent=_Intent,
    Command=_Command,
    DEFAULT_INTENT_LABELS={},
    command_to_intent=lambda c: _Intent.NONE,
    intent_from_label_set=lambda labels: _Intent.NONE,
    merge_intents=lambda a, b: _Intent.NONE,
    merge_intents_with_cli=lambda a, b, c=None: _Intent.NONE,
    parse_agent_command=lambda b: None,
)

_reg(
    "extensions.orchestrator.tracker_kinds",
    SUPPORTED_TRACKERS=frozenset({"linear", "github", "gitee", "gitcode", "local"}),
    TrackerConfigError=type("TrackerConfigError", (ValueError,), {}),
    TrackerKindInfo=type("TrackerKindInfo", (), {}),
    create_tracker_adapter=lambda *a, **k: None,
    default_active_states_for_kind=lambda k: ["open"],
    default_terminal_states_for_kind=lambda k: ["closed"],
    normalize_tracker_kind=lambda k: k or "local",
    repository_clone_url_for_tracker=lambda c: None,
    tracker_kind_info=lambda k: SimpleNamespace(),
    validate_tracker_config=lambda c: None,
)

# title_prefix_filter (needed by local_tracker.adapter, which git_sync lazy-imports)
_reg(
    "extensions.orchestrator.title_prefix_filter",
    normalize_title_prefixes=lambda p: tuple(s.strip() for s in (p or []) if isinstance(s, str) and s.strip()),
    normalize_title_prefix_match=lambda v: "any",
    matches_title_prefixes=lambda t, p, m: True,
)

# --- Real imports (git_sync, tracker, workspace migrated) ---

from extensions.orchestrator.config.schema import (  # noqa: E402
    AgentConfig,
    HooksConfig,
    PrTemplateConfig,
    WorkflowConfig,
)
from extensions.orchestrator.git_sync import (  # noqa: E402
    GitSyncPostCommitError,
    GitSyncService,
    HookFailedError,
    VerificationFailed,
)
from extensions.orchestrator.issue import Issue  # noqa: E402
from extensions.orchestrator.tracker import Intent, PullRequestRef, TrackerAdapter  # noqa: E402
from extensions.orchestrator.workspace import Workspace, WorkspaceConfig, WorkspaceManager  # noqa: E402


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def _git_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return result.stdout.strip()


class _Comment:
    def __init__(self, id: str, body: str) -> None:
        self.id = id
        self.body = body


class _Tracker(TrackerAdapter):
    def __init__(self) -> None:
        self.comments: list[tuple[str, str]] = []
        self.updated_comments: list[tuple[str, str, str]] = []
        self.pr_requests: list[tuple[str, str, str, str]] = []
        self.pr_updates: list[tuple[PullRequestRef, str | None, str | None]] = []

    async def fetch_candidate_issues(self) -> list[Issue]:
        return []

    async def fetch_issue_states_by_ids(self, issue_ids: list[str]) -> dict[str, Issue]:
        return {}

    async def create_comment(self, issue_id: str, body: str) -> _Comment:
        self.comments.append((issue_id, body))
        return _Comment(str(len(self.comments)), body)

    async def update_comment(self, issue_id: str, comment_id: str, body: str) -> _Comment | None:
        self.updated_comments.append((issue_id, comment_id, body))
        return _Comment(comment_id, body)

    async def update_issue_state(self, issue_id: str, state: str) -> None:
        return None

    async def create_clarification_comment(
        self, issue_id: str, body: str, mentions: list[str] | None = None
    ) -> _Comment | None:
        return None

    async def extract_intent_from_labels(self, labels: list[str] | None) -> Intent:
        return Intent.NONE

    async def close_pull_request(self, pull_request: PullRequestRef) -> bool:
        return False

    async def fetch_pull_request_mergeable(self, pull_request: PullRequestRef) -> None:
        return None

    async def find_pull_request(self, *, head_branch: str, base_branch: str) -> PullRequestRef | None:
        return None

    async def ensure_pull_request(
        self, *, issue: Issue, head_branch: str, base_branch: str, title: str, body: str
    ) -> PullRequestRef | None:
        self.pr_requests.append((head_branch, base_branch, title, body))
        return PullRequestRef(number="9", url="https://example.test/pr/9", title=title)

    async def update_pull_request(
        self, *, pull_request: PullRequestRef, title: str | None = None, body: str | None = None
    ) -> PullRequestRef | None:
        self.pr_updates.append((pull_request, title, body))
        return PullRequestRef(number=pull_request.number, url=pull_request.url, title=title or pull_request.title)


class _FindPRTracker(_Tracker):
    async def find_pull_request(self, *, head_branch: str, base_branch: str) -> PullRequestRef | None:
        return PullRequestRef(number="44", url="https://example.test/pr/44", title=f"{head_branch}->{base_branch}")


class _Session:
    def __init__(self, issue: Issue, workspace: Workspace) -> None:
        self.issue = issue
        self.workspace = workspace
        self.status = "completed"
        self.run_id = "run-01-20260601T000000Z"
        self.summary_comment_id = "summary-1"
        self.turn_count = 1
        self.tool_count = 1
        self.verification_status = None
        self.verification_output = None
        self.output_text = "done"


def _build_origin_repo(base: Path) -> Path:
    origin = base / "origin.git"
    seed = base / "seed"
    seed.mkdir(parents=True)
    _git(["init", "--bare", str(origin)], base)
    _git(["init"], seed)
    _git(["config", "user.email", "test@example.com"], seed)
    _git(["config", "user.name", "Test User"], seed)
    (seed / "README.md").write_text("main branch\n", encoding="utf-8")
    _git(["add", "README.md"], seed)
    _git(["commit", "-m", "initial"], seed)
    _git(["branch", "-M", "main"], seed)
    _git(["remote", "add", "origin", str(origin)], seed)
    _git(["push", "-u", "origin", "main"], seed)
    _git(["symbolic-ref", "HEAD", "refs/heads/main"], origin)
    return origin


class TestGitSyncService(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._home_dir = tempfile.TemporaryDirectory()
        self._home_patch = mock.patch.object(Path, "home", return_value=Path(self._home_dir.name))
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._home_dir.cleanup()

    def test_pr_template_renders_fixed_and_generated_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_path = Path(tmp)
            (workspace_path / "changes_summary.md").write_text("- Added configurable PR bodies.", encoding="utf-8")
            (workspace_path / "implementation_notes.md").write_text(
                "Templates use data-only substitutions.", encoding="utf-8"
            )
            (workspace_path / "verification_report.md").write_text("pytest: 12 passed", encoding="utf-8")
            issue = Issue(
                id="77", identifier="ISSUE-77", title="Configurable PR template", url="https://example.test/issues/77"
            )
            session = _Session(issue, Workspace(workspace_path, "ISSUE-77", "77"))
            session.verification_status = "passed"
            service = GitSyncService(
                _Tracker(),
                pr_template=PrTemplateConfig(
                    title="PR: {{ issue.identifier }} / {{ issue.title }}",
                    body=(
                        "## Summary\n{{ changes_summary }}\n\n## Notes\n{{ implementation_notes }}\n\n"
                        "## Checks\n{{ verification_status }}: {{ verification_summary }}\n\n{{ issue.url }}"
                    ),
                ),
            )
            self.assertEqual(service._build_pr_title(issue), "PR: ISSUE-77 / Configurable PR template")
            self.assertEqual(
                service._build_pr_body(
                    issue, "abc123", "clawcodex/issue-77", "main", session=session, pull_request=None
                ),
                "## Summary\n- Added configurable PR bodies.\n\n## Notes\nTemplates use data-only substitutions.\n\n"
                "## Checks\npassed: pytest: 12 passed\n\nhttps://example.test/issues/77",
            )

    def test_pr_template_keeps_unknown_variables_empty(self) -> None:
        service = GitSyncService(
            _Tracker(), pr_template=PrTemplateConfig(body="Known={{ issue.identifier }}, unknown={{ no_such_value }}")
        )
        issue = Issue(id="1", identifier="ISSUE-1", title="Test")
        self.assertEqual(
            service._build_pr_body(issue, None, "branch", "main", session=object(), pull_request=None),
            "Known=ISSUE-1, unknown=",
        )

    def test_workflow_config_parses_pr_template(self) -> None:
        workflow = WorkflowConfig.from_dict(
            {"pr_template": {"title": "{{ issue.title }}", "body": "## Fixed\n{{ changes_summary }}"}}
        )
        self.assertEqual(workflow.pr_template.title, "{{ issue.title }}")
        self.assertEqual(workflow.pr_template.body, "## Fixed\n{{ changes_summary }}")

    def test_merge_pr_ref_preserves_existing_number_and_url(self) -> None:
        service = GitSyncService(_Tracker())
        result = service._merge_pr_ref(
            PullRequestRef(title="Updated title"),
            PullRequestRef(number="9", url="https://example.test/pr/9", title="Old title"),
        )
        self.assertEqual(result, PullRequestRef(number="9", url="https://example.test/pr/9", title="Updated title"))

    async def test_find_pr_fallback_uses_tracker_find_pull_request(self) -> None:
        service = GitSyncService(_FindPRTracker())
        result = await service._find_pr_fallback(
            PullRequestRef(title="pending"), head_branch="feature/issue-44", base_branch="main"
        )
        self.assertEqual(
            result, PullRequestRef(number="44", url="https://example.test/pr/44", title="feature/issue-44->main")
        )

    async def test_sync_commits_pushes_and_creates_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(
                id="77", identifier="ISSUE-77", title="Automate git sync", url="https://example.test/issues/77"
            )
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("changed\n", encoding="utf-8")
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(_Session(issue, workspace))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.committed)
            self.assertTrue(result.pushed)
            self.assertEqual(result.base_branch, "main")
            self.assertTrue(result.branch_name.startswith("clawcodex/issue-77"))
            self.assertEqual(_git_output(["rev-parse", "--abbrev-ref", "HEAD"], workspace.path), result.branch_name)
            self.assertEqual(
                _git_output(["ls-remote", "--heads", "origin", result.branch_name], workspace.path) != "", True
            )
            self.assertEqual(len(tracker.pr_requests), 1)
            self.assertEqual(tracker.pr_requests[0][0], result.branch_name)
            self.assertEqual(tracker.pr_requests[0][1], "main")
            self.assertEqual(tracker.comments, [])
            self.assertEqual(len(tracker.updated_comments), 1)
            self.assertIn("Pull request: https://example.test/pr/9", tracker.updated_comments[0][2])
            self.assertEqual(len(tracker.pr_updates), 1)
            assert tracker.pr_updates[0][2] is not None
            self.assertIn("Verification: `skipped_no_tests`", tracker.pr_updates[0][2])
            self.assertIn("Report: `", tracker.pr_updates[0][2])

    async def test_followup_sync_reuses_existing_pr_and_uses_fix_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Automate git sync", branch_name="clawcodex/issue-77")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("follow-up\n", encoding="utf-8")
            session = _Session(issue, workspace)
            session.pull_request = PullRequestRef(number="9", url="https://example.test/pr/9")
            session.base_branch = "main"
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(session)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.committed)
            self.assertTrue(result.pushed)
            self.assertEqual(result.pull_request.number, session.pull_request.number)
            self.assertEqual(result.pull_request.url, session.pull_request.url)
            self.assertEqual(tracker.pr_requests, [])
            self.assertEqual(
                _git_output(["log", "-1", "--pretty=%s"], workspace.path), "fix: ISSUE-77 Automate git sync"
            )
            self.assertIn("Pull request: https://example.test/pr/9", tracker.updated_comments[0][2])

    async def test_followup_sync_does_not_overwrite_pr_body(self) -> None:
        """Follow-up sync must not rewrite the PR body/title."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(
                id="78", identifier="ISSUE-78", title="Do not clobber PR body", branch_name="clawcodex/issue-78"
            )
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("review fix\n", encoding="utf-8")
            session = _Session(issue, workspace)
            session.pull_request = PullRequestRef(number="12", url="https://example.test/pr/12")
            session.base_branch = "main"
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(session)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.committed)
            self.assertTrue(result.pushed)
            self.assertEqual(tracker.pr_updates, [])

    async def test_sync_push_non_fast_forward_recovers(self) -> None:
        """Non-fast-forward push triggers rebase and returns conflict state."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            work_a = base / "work_a"
            work_b = base / "work_b"
            _git(["clone", str(origin), str(work_a)], base)
            _git(["clone", str(origin), str(work_b)], base)
            _git(["config", "user.email", "a@example.com"], work_a)
            _git(["config", "user.name", "A"], work_a)
            _git(["config", "user.email", "b@example.com"], work_b)
            _git(["config", "user.name", "B"], work_b)
            (work_a / "file.txt").write_text("from A\n")
            _git(["add", "file.txt"], work_a)
            _git(["commit", "-m", "from A"], work_a)
            _git(["push", "-f", "origin", "main"], work_a)
            (work_b / "file.txt").write_text("from B\n")
            _git(["add", "file.txt"], work_b)
            _git(["commit", "-m", "from B"], work_b)
            issue = Issue(
                id="99",
                identifier="ISSUE-99",
                title="Conflict recovery test",
                url="https://example.test/issues/99",
                branch_name="main",
            )

            class _FakeWorkspace:
                def __init__(self, path: Path) -> None:
                    self.path = path

            class _FakeSession:
                def __init__(self, ws: _FakeWorkspace, iss: Issue) -> None:
                    self.workspace = ws
                    self.issue = iss

            session = _FakeSession(_FakeWorkspace(work_b), issue)
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(session)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertFalse(result.pushed)
            self.assertTrue(result.has_conflict)
            self.assertGreater(len(result.conflict_files), 0)

    async def test_pre_push_verification_failure_prevents_push(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Verify before push")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("changed\n", encoding="utf-8")
            service = GitSyncService(
                _Tracker(), agent_config=AgentConfig(test_command="python -c 'raise SystemExit(7)'")
            )
            with self.assertRaises(GitSyncPostCommitError) as cm:
                await service.sync(_Session(issue, workspace))
            self.assertIsInstance(cm.exception.cause, VerificationFailed)
            self.assertFalse(cm.exception.result.committed)
            self.assertIsNotNone(cm.exception.result.commit_sha)
            self.assertNotEqual(_git_output(["rev-parse", "HEAD"], workspace.path), cm.exception.result.commit_sha)
            self.assertEqual(
                _git_output(
                    ["ls-remote", "--heads", "origin", "clawcodex/issue-77-verify-before-push"], workspace.path
                ),
                "",
            )

    async def test_pre_push_verification_failure_with_existing_commit_registers_head(self) -> None:
        """No-staged-changes path: existing HEAD surfaces via GitSyncPostCommitError."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(
                    root=base / "workspace",
                    repo_clone_url=str(origin),
                    strategy="sequential",
                    checkout_issue_branch=False,
                    base_branch="main",
                    integration_branch="integration/f40",
                )
            )
            issue = Issue(id="40", identifier="F-40", title="Existing implementation")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "progress_sink.py").write_text("# implementation\n", encoding="utf-8")
            _git(["add", "progress_sink.py"], workspace.path)
            _git(["commit", "-m", "refactor: pre-existing F-40 implementation"], workspace.path)
            head_sha = _git_output(["rev-parse", "HEAD"], workspace.path)
            self.assertEqual(_git_output(["diff", "--cached", "--name-only"], workspace.path), "")
            session = _Session(issue, workspace)
            session.workspace_strategy = "sequential"
            session.integration_branch = "integration/f40"
            session.start_commit_sha = head_sha
            service = GitSyncService(
                _Tracker(), agent_config=AgentConfig(test_command="python -c 'raise SystemExit(7)'")
            )
            with self.assertRaises(GitSyncPostCommitError) as cm:
                await service.sync(session)
            self.assertIsInstance(cm.exception.cause, VerificationFailed)
            self.assertEqual(cm.exception.result.commit_sha, head_sha)
            self.assertFalse(cm.exception.result.committed)
            self.assertEqual(cm.exception.result.branch_name, "integration/f40")

    async def test_pre_commit_hook_modifies_files_and_amends_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Format before commit")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("changed\n", encoding="utf-8")
            service = GitSyncService(
                _Tracker(),
                hooks_config=HooksConfig(
                    pre_commit=f"{sys.executable} -c \"from pathlib import Path; Path('formatted.txt').write_text('ok\\n')\""
                ),
            )
            await service.sync(_Session(issue, workspace))
            self.assertIn("formatted.txt", _git_output(["show", "--name-only", "--pretty="], workspace.path))

    async def test_pre_push_hook_cannot_modify_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Dirty pre push")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("changed\n", encoding="utf-8")
            service = GitSyncService(
                _Tracker(),
                hooks_config=HooksConfig(
                    pre_push=f"{sys.executable} -c \"from pathlib import Path; Path('dirty.txt').write_text('dirty\\n')\""
                ),
            )
            with self.assertRaises(GitSyncPostCommitError) as cm:
                await service.sync(_Session(issue, workspace))
            self.assertIsInstance(cm.exception.cause, HookFailedError)
            self.assertEqual(cm.exception.hook_name, "pre_push")
            self.assertFalse(cm.exception.result.committed)
            self.assertIsNotNone(cm.exception.result.commit_sha)
            self.assertIn("modified the workspace", str(cm.exception))

    async def test_sequential_sync_commits_without_push_or_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(
                    root=base / "workspace",
                    repo_clone_url=str(origin),
                    strategy="sequential",
                    checkout_issue_branch=False,
                    base_branch="main",
                    integration_branch="integration/f42",
                )
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Sequential commit")
            workspace = await manager.create_for_issue(issue)
            (workspace.path / "README.md").write_text("sequential\n", encoding="utf-8")
            session = _Session(issue, workspace)
            session.workspace_strategy = "sequential"
            session.integration_branch = "integration/f42"
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(session)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.committed)
            self.assertFalse(result.pushed)
            self.assertIsNone(result.pull_request)
            self.assertEqual(result.branch_name, "integration/f42")
            self.assertEqual(tracker.pr_requests, [])
            self.assertEqual(tracker.pr_updates, [])
            self.assertEqual(_git_output(["rev-parse", "--abbrev-ref", "HEAD"], workspace.path), "integration/f42")
            self.assertEqual(
                _git_output(["log", "-1", "--pretty=%s"], workspace.path), "feat: ISSUE-77 Sequential commit"
            )
            self.assertEqual(_git_output(["ls-remote", "--heads", "origin", "integration/f42"], workspace.path), "")
            await manager.cleanup(issue)

    async def test_sequential_agent_created_commit_enters_pending_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(
                    root=base / "workspace",
                    repo_clone_url=str(origin),
                    strategy="sequential",
                    checkout_issue_branch=False,
                    base_branch="main",
                    integration_branch="integration/f42",
                )
            )
            issue = Issue(id="77", identifier="ISSUE-77", title="Already committed")
            workspace = await manager.create_for_issue(issue)
            service = GitSyncService(_Tracker())
            service._sync_gitignore(str(workspace.path))
            self.assertFalse((workspace.path / ".gitignore").exists())
            self.assertIn(
                ".clawcodex_issue_registry.json",
                (workspace.path / ".git" / "info" / "exclude").read_text(encoding="utf-8"),
            )
            start_commit_sha = _git_output(["rev-parse", "HEAD"], workspace.path)
            (workspace.path / "README.md").write_text("agent committed\n", encoding="utf-8")
            _git(["add", "README.md"], workspace.path)
            _git(["commit", "-m", "feat: ISSUE-77 Already committed"], workspace.path)
            agent_commit_sha = _git_output(["rev-parse", "HEAD"], workspace.path)
            session = _Session(issue, workspace)
            session.workspace_strategy = "sequential"
            session.integration_branch = "integration/f42"
            session.start_commit_sha = start_commit_sha
            service = GitSyncService(_Tracker(), agent_config=AgentConfig(review_required=True))
            result = await service.sync(session)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.commit_sha, agent_commit_sha)
            self.assertTrue(result.committed)
            self.assertTrue(result.pending_review)
            self.assertFalse(result.pushed)
            await manager.cleanup(issue)

    async def test_sequential_second_commit_builds_on_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(
                    root=base / "workspace",
                    repo_clone_url=str(origin),
                    strategy="sequential",
                    checkout_issue_branch=False,
                    base_branch="main",
                    integration_branch="integration/f42",
                )
            )
            service = GitSyncService(_Tracker())
            issue_one = Issue(id="1", identifier="ISSUE-1", title="First")
            workspace = await manager.create_for_issue(issue_one)
            (workspace.path / "one.txt").write_text("one\n", encoding="utf-8")
            session_one = _Session(issue_one, workspace)
            session_one.workspace_strategy = "sequential"
            session_one.integration_branch = "integration/f42"
            result_one = await service.sync(session_one)
            await manager.cleanup(issue_one)
            issue_two = Issue(id="2", identifier="ISSUE-2", title="Second")
            workspace = await manager.create_for_issue(issue_two)
            (workspace.path / "two.txt").write_text("two\n", encoding="utf-8")
            session_two = _Session(issue_two, workspace)
            session_two.workspace_strategy = "sequential"
            session_two.integration_branch = "integration/f42"
            result_two = await service.sync(session_two)
            self.assertIsNotNone(result_one)
            self.assertIsNotNone(result_two)
            assert result_one is not None
            assert result_two is not None
            self.assertEqual(
                _git_output(["rev-parse", f"{result_two.commit_sha}^"], workspace.path), result_one.commit_sha
            )
            self.assertEqual(
                _git_output(["log", "--pretty=%s", "-2"], workspace.path).splitlines(),
                ["feat: ISSUE-2 Second", "feat: ISSUE-1 First"],
            )
            await manager.cleanup(issue_two)

    async def test_empty_branch_no_commits_skips_pr_creation(self) -> None:
        """When daemon triggers read-only loop termination, sync() must refuse PR creation."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            origin = _build_origin_repo(base)
            manager = WorkspaceManager(
                WorkspaceConfig(root=base / "workspaces", repo_clone_url=str(origin), checkout_issue_branch=True)
            )
            issue = Issue(
                id="99", identifier="ISSUE-99", title="Read-only spiral", url="https://example.test/issues/99"
            )
            workspace = await manager.create_for_issue(issue)
            tracker = _Tracker()
            service = GitSyncService(tracker)
            result = await service.sync(_Session(issue, workspace))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertFalse(result.committed)
            self.assertIsNone(result.commit_sha)
            self.assertIsNone(result.pull_request)
            self.assertEqual(result.session_end_reason, "empty_branch_no_commits")
            self.assertEqual(tracker.pr_requests, [])
            self.assertEqual(tracker.pr_updates, [])
            await manager.cleanup(issue)


if __name__ == "__main__":
    unittest.main()
