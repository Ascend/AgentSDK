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

"""Tests for GitSyncService repro recheck in pre-push verification.

Covers the repro-first gate integration with git_sync: when a repro
command is set on the session, it is re-run during pre-push verification.
A green (exit 0) repro passes; a red (exit non-zero) blocks the push.

Unmigrated deps are stubbed via sys.modules (same pattern as
test_orchestrator_git_sync.py).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

# --- Stubs for unmigrated dependencies ---

_SKIP = {"extensions", "extensions.orchestrator"}


def _reg(n: str, **kw: object) -> None:
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
    get_default_branch=lambda cwd=None: "main",
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


_reg(
    "extensions.orchestrator.config.schema",
    AgentConfig=_AgentConfig,
    HooksConfig=_HooksConfig,
    PrTemplateConfig=_PrTemplateConfig,
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

_reg(
    "extensions.orchestrator.title_prefix_filter",
    normalize_title_prefixes=lambda p: tuple(s.strip() for s in (p or []) if isinstance(s, str) and s.strip()),
    normalize_title_prefix_match=lambda v: "any",
    matches_title_prefixes=lambda t, p, m: True,
)

# --- Real imports ---

from extensions.orchestrator.git_sync import GitSyncService, VerificationFailed  # noqa: E402


def _exit_script(root: Path, code: int) -> str:
    script = root / ".orchestrator_control" / "repro" / "check.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"import sys\nsys.exit({code})\n", encoding="utf-8")
    return f'"{sys.executable}" {script.relative_to(root)}'


class _NullTracker:
    """Duck-typed tracker stub (GitSyncService only stores it here)."""


class TestGitSyncReproRecheck(unittest.IsolatedAsyncioTestCase):
    async def test_green_repro_command_passes_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _exit_script(root, 0)

            class _Session:
                repro_command = command
                verification_status = None
                verification_output = None

            service = GitSyncService(_NullTracker())
            session = _Session()
            await service._run_pre_push_verification(str(root), session)
            self.assertEqual(session.verification_status, "passed")
            assert session.verification_output is not None
            self.assertIn("## repro", session.verification_output)

    async def test_red_repro_command_blocks_push(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _exit_script(root, 1)

            class _Session:
                repro_command = command
                verification_status = None
                verification_output = None

            service = GitSyncService(_NullTracker())
            with self.assertRaises(VerificationFailed) as ctx:
                await service._run_pre_push_verification(str(root), _Session())
            self.assertIn("still exits non-zero", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
