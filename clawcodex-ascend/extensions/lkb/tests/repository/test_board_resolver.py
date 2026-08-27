#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for lkb.board_resolver — Board identity resolution (spec §5.2).

Test IDs:
  LKB-BOARD-001 — explicit_id tier 1 overrides all lower tiers
  LKB-BOARD-002 — CLAWCODEX_LKB_BOARD_ID env var (tier 2)
  LKB-BOARD-003 — project .claude/config.json lkb.board_id (tier 3)
  LKB-BOARD-004 — derive from workspace root + git identity (tier 4)
  LKB-BOARD-005 — session board fallback (tier 5)
  LKB-BOARD-006 — safe_board_id + path traversal rejection
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lkb.board_resolver import (
    BoardResolutionError,
    board_dir,
    board_file_paths,
    normalize_workspace_root,
    resolve_board,
    safe_board_id,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:resolve_board\\(\\.\\.\\., home=\\.\\.\\.\\) is deprecated:DeprecationWarning"
)


# ── LKB-BOARD-001: explicit_id tier 1 ────────────────────────────────


class TestExplicitIdTier1:
    """Tier 1: explicit board_id parameter wins over everything."""

    def test_explicit_id_provided(self, tmp_home: Path) -> None:
        board = resolve_board(explicit_id="my-project-board", home=tmp_home)
        assert board.board_id == "my-project-board"
        assert board.schema_version == 1
        assert board.store_revision == 0
        assert board.display_name == "my-project-board"

    def test_home_is_ignored_but_kept_for_source_compatibility(self, tmp_home: Path) -> None:
        expected = resolve_board(explicit_id="legacy-home").board_id

        with pytest.warns(DeprecationWarning, match="home does not affect board identity"):
            legacy = resolve_board(explicit_id="legacy-home", home=tmp_home)

        assert legacy.board_id == expected

    def test_explicit_id_overrides_env(self, tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWCODEX_LKB_BOARD_ID", "env-board")
        board = resolve_board(explicit_id="explicit-wins", home=tmp_home)
        assert board.board_id == "explicit-wins"

    def test_explicit_id_with_workspace_root(self, tmp_path: Path, tmp_home: Path) -> None:
        # Even with a workspace that would resolve differently,
        # explicit_id wins.
        board = resolve_board(
            tmp_path / "some-project",
            explicit_id="override-board",
            home=tmp_home,
        )
        assert board.board_id == "override-board"
        # project_uri should still reflect the workspace
        assert board.project_uri.startswith("project:")

    def test_explicit_id_rejects_traversal(self, tmp_home: Path) -> None:
        with pytest.raises(BoardResolutionError, match="forbidden"):
            resolve_board(explicit_id="../escape", home=tmp_home)

    def test_explicit_id_rejects_empty(self, tmp_home: Path) -> None:
        with pytest.raises(BoardResolutionError, match="empty"):
            resolve_board(explicit_id="", home=tmp_home)

    def test_explicit_id_rejects_dot_prefix(self, tmp_home: Path) -> None:
        with pytest.raises(BoardResolutionError, match="dot"):
            resolve_board(explicit_id=".hidden", home=tmp_home)


# ── LKB-BOARD-002: env var tier 2 ────────────────────────────────────


class TestEnvVarTier2:
    """Tier 2: CLAWCODEX_LKB_BOARD_ID environment variable."""

    def test_env_var_resolves(self, tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWCODEX_LKB_BOARD_ID", "env-board-123")
        board = resolve_board(home=tmp_home)
        assert board.board_id == "env-board-123"

    def test_env_var_overrides_workspace_config(
        self, tmp_path: Path, tmp_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Set up a project config with a board_id
        _write_project_config(tmp_path, "config-board")
        monkeypatch.setenv("CLAWCODEX_LKB_BOARD_ID", "env-wins")
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id == "env-wins"

    def test_empty_env_var_skipped(self, tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWCODEX_LKB_BOARD_ID", "")
        board = resolve_board(home=tmp_home)
        # Should fall through to session tier
        assert board.board_id.startswith("session-")

    def test_env_var_rejects_traversal(self, tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWCODEX_LKB_BOARD_ID", "../../etc/passwd")
        with pytest.raises(BoardResolutionError, match="forbidden"):
            resolve_board(home=tmp_home)


# ── LKB-BOARD-003: project config tier 3 ─────────────────────────────


class TestProjectConfigTier3:
    """Tier 3: .claude/config.json lkb.board_id."""

    def test_project_config_read(self, tmp_path: Path, tmp_home: Path) -> None:
        _write_project_config(tmp_path, "config-board-42")
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id == "config-board-42"

    def test_project_config_skipped_when_missing(self, tmp_path: Path, tmp_home: Path) -> None:
        # No .claude/config.json — should fall through to tier 4
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id.startswith("proj-")

    def test_project_config_invalid_json_falls_through(self, tmp_path: Path, tmp_home: Path) -> None:
        config_dir = tmp_path / ".claude"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("not json {{{", encoding="utf-8")
        board = resolve_board(tmp_path, home=tmp_home)
        # Should not raise; falls through to tier 4
        assert board.board_id.startswith("proj-")

    def test_project_config_no_lkb_key(self, tmp_path: Path, tmp_home: Path) -> None:
        config_dir = tmp_path / ".claude"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"other": "stuff"}), encoding="utf-8")
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id.startswith("proj-")

    def test_project_config_non_string_board_id(self, tmp_path: Path, tmp_home: Path) -> None:
        config_dir = tmp_path / ".claude"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"lkb": {"board_id": 123}}), encoding="utf-8")
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id.startswith("proj-")

    def test_project_uri_reflects_workspace(self, tmp_path: Path, tmp_home: Path) -> None:
        _write_project_config(tmp_path, "my-board")
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.project_uri.startswith("project:")


# ── LKB-BOARD-004: derive from workspace root tier 4 ─────────────────


class TestDerivedIdTier4:
    """Tier 4: derive stable ID from normalized workspace root + git identity."""

    def test_derived_id_starts_with_proj(self, tmp_path: Path, tmp_home: Path) -> None:
        board = resolve_board(tmp_path, home=tmp_home)
        assert board.board_id.startswith("proj-")
        assert len(board.board_id) == len("proj-") + 16  # prefix + 16 hex chars

    def test_derived_id_is_stable(self, tmp_path: Path, tmp_home: Path) -> None:
        board1 = resolve_board(tmp_path, home=tmp_home)
        board2 = resolve_board(tmp_path, home=tmp_home)
        assert board1.board_id == board2.board_id

    def test_different_roots_different_ids(self, tmp_path: Path, tmp_home: Path) -> None:
        board_a = resolve_board(tmp_path / "alpha", home=tmp_home)
        board_b = resolve_board(tmp_path / "beta", home=tmp_home)
        assert board_a.board_id != board_b.board_id

    def test_display_name_from_basename(self, tmp_path: Path, tmp_home: Path) -> None:
        board = resolve_board(tmp_path / "my-cool-project", home=tmp_home)
        assert "my-cool-project" in board.display_name.lower()

    def test_project_uri_format(self, tmp_path: Path, tmp_home: Path) -> None:
        board = resolve_board(tmp_path / "proj", home=tmp_home)
        assert board.project_uri.startswith("project:")
        # The URI should contain the normalized path
        assert len(board.project_uri) > len("project:")

    def test_git_author_identity_does_not_change_board_id(self, tmp_path: Path, tmp_home: Path) -> None:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "First User"], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "first@example.test"],
            check=True,
        )
        first = resolve_board(tmp_path, home=tmp_home).board_id
        subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Second User"], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "second@example.test"],
            check=True,
        )
        assert resolve_board(tmp_path, home=tmp_home).board_id == first

    def test_same_remote_in_two_clone_paths_has_same_board_id(self, tmp_path: Path, tmp_home: Path) -> None:
        clone_a = tmp_path / "clone-a"
        clone_b = tmp_path / "elsewhere" / "clone-b"
        clone_b.parent.mkdir()
        for clone in (clone_a, clone_b):
            subprocess.run(["git", "init", "-q", str(clone)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(clone),
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/OpenAI/example.git",
                ],
                check=True,
            )
        assert resolve_board(clone_a, home=tmp_home).board_id == resolve_board(clone_b, home=tmp_home).board_id

    @pytest.mark.parametrize(
        "remote",
        [
            "https://token@github.com/OpenAI/example.git",
            "ssh://git@github.com:22/OpenAI/example.git",
            "git@github.com:OpenAI/example.git",
        ],
    )
    def test_remote_transport_forms_share_identity(self, tmp_path: Path, tmp_home: Path, remote: str) -> None:
        baseline = tmp_path / "baseline"
        candidate = tmp_path / f"candidate-{abs(hash(remote))}"
        for repo, url in (
            (baseline, "https://github.com/OpenAI/example.git"),
            (candidate, remote),
        ):
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", url], check=True)
        assert resolve_board(baseline, home=tmp_home).board_id == resolve_board(candidate, home=tmp_home).board_id


# ── LKB-BOARD-005: session board tier 5 ──────────────────────────────


class TestSessionBoardTier5:
    """Tier 5: session-scoped board when no workspace is available."""

    def test_session_board_no_workspace(self, tmp_home: Path) -> None:
        board = resolve_board(home=tmp_home)
        assert board.board_id.startswith("session-")
        assert board.project_uri.startswith("session:")
        assert board.display_name == "Session Board"

    def test_session_board_stable_per_process(self, tmp_home: Path) -> None:
        board_a = resolve_board(home=tmp_home)
        board_b = resolve_board(home=tmp_home)
        assert board_a.board_id == board_b.board_id

    def test_host_session_id_is_stable_and_not_exposed(self, tmp_home: Path) -> None:
        board_a = resolve_board(home=tmp_home, session_id="secret-session")
        board_b = resolve_board(home=tmp_home, session_id="secret-session")
        board_c = resolve_board(home=tmp_home, session_id="another-session")
        assert board_a.board_id == board_b.board_id
        assert board_a.board_id != board_c.board_id
        assert "secret-session" not in board_a.board_id

    def test_session_board_has_default_policy(self, tmp_home: Path) -> None:
        board = resolve_board(home=tmp_home)
        assert board.policy.invalidation_mode == "cascade"
        assert board.store_revision == 0
        assert board.schema_version == 1


# ── LKB-BOARD-006: safe_board_id + path helpers ──────────────────────


class TestSafeBoardId:
    """safe_board_id, board_dir, and path-traversal rejection."""

    def test_safe_board_id_format(self) -> None:
        sid = safe_board_id("my-project")
        assert "_" in sid
        prefix, suffix = sid.rsplit("_", 1)
        assert len(suffix) == 16
        # Suffix should be hex
        int(suffix, 16)

    def test_safe_board_id_stable(self) -> None:
        assert safe_board_id("hello") == safe_board_id("hello")

    def test_safe_board_id_different_inputs_different(self) -> None:
        assert safe_board_id("alpha") != safe_board_id("beta")

    def test_safe_board_id_sanitizes_prefix(self) -> None:
        sid = safe_board_id("My Project!@#$%^&*()")
        # Prefix should be lowercase alphanumeric/dash only
        prefix = sid.split("_")[0]
        assert prefix.isalnum() or "-" in prefix or "_" in prefix
        assert prefix == prefix.lower()

    def test_safe_board_id_truncates_long_prefix(self) -> None:
        long_id = "a-very-long-board-id-that-exceeds-twelve-characters"
        sid = safe_board_id(long_id)
        prefix = sid.split("_")[0]
        assert len(prefix) <= 12

    def test_safe_board_id_handles_non_ascii(self) -> None:
        sid = safe_board_id("项目看板")
        # Non-ASCII gets stripped to empty prefix, falls back to "board"
        prefix = sid.split("_")[0]
        assert prefix == "board" or len(prefix) > 0

    def test_safe_board_id_rejects_traversal(self) -> None:
        with pytest.raises(BoardResolutionError, match="forbidden"):
            safe_board_id("../etc/passwd")

    def test_safe_board_id_rejects_backslash(self) -> None:
        with pytest.raises(BoardResolutionError, match="forbidden"):
            safe_board_id("foo\\bar")

    def test_safe_board_id_rejects_null_byte(self) -> None:
        with pytest.raises(BoardResolutionError, match="forbidden"):
            safe_board_id("bad\x00id")

    def test_board_dir_path(self, tmp_home: Path) -> None:
        d = board_dir("my-board", home=tmp_home)
        assert d.parent.parent.parent == tmp_home  # <home>/lkb/boards/<sid>
        assert d.name == safe_board_id("my-board")

    def test_board_dir_rejects_traversal(self, tmp_home: Path) -> None:
        with pytest.raises(BoardResolutionError):
            board_dir("../../escape", home=tmp_home)

    def test_board_file_paths_keys(self, tmp_home: Path) -> None:
        paths = board_file_paths("test-board", home=tmp_home)
        expected_keys = {
            "board_json",
            "board_json_bak",
            "lock_file",
            "lock_owner_json",
            "tmp_dir",
            "history_dir",
            "quarantine_dir",
        }
        assert set(paths.keys()) == expected_keys
        # All values should be Path objects inside the board dir
        d = board_dir("test-board", home=tmp_home)
        for p in paths.values():
            assert str(p).startswith(str(d))


# ── normalize_workspace_root tests (LKB-BOARD-004 detail) ────────────


class TestNormalizeWorkspaceRoot:
    """Path normalization for stable identity input."""

    def test_absolute_path(self, tmp_path: Path) -> None:
        result = normalize_workspace_root(tmp_path)
        assert os.path.isabs(result.replace("\\", "/"))

    def test_separators_normalized(self, tmp_path: Path) -> None:
        result = normalize_workspace_root(tmp_path)
        # Should use forward slashes
        assert "\\" not in result

    def test_symlink_resolved(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real_dir)
            result = normalize_workspace_root(link)
            # Should resolve to the real path
            assert "real" in result
        except OSError:
            # Symlinks may not work on Windows without privileges
            pytest.skip("symlink not supported")

    def test_no_trailing_slash(self, tmp_path: Path) -> None:
        result = normalize_workspace_root(tmp_path)
        assert not result.endswith("/")

    def test_stable_for_same_path(self, tmp_path: Path) -> None:
        a = normalize_workspace_root(tmp_path)
        b = normalize_workspace_root(tmp_path)
        assert a == b


# ── helpers ───────────────────────────────────────────────────────────


def _write_project_config(root: Path, board_id: str) -> None:
    """Write a minimal .claude/config.json with lkb.board_id."""
    config_dir = root / ".claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {"lkb": {"board_id": board_id}}
    (config_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
