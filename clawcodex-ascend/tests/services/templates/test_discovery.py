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

"""Unit tests for :mod:`src.services.templates.discovery`.

Exercises the path resolvers in isolation — no I/O, no registry
mutation. Each test passes a tmp dir or monkeypatches the relevant
env vars so the host environment cannot leak in.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.templates import (
    CLAWCODEX_CONFIG_DIR_ENV,
    CLAWCODEX_MANAGED_CONFIG_DIR_ENV,
    PROJECT_CONFIG_DIR,
    TEMPLATES_SUBDIR,
    get_managed_templates_dir,
    get_project_templates_dirs,
    get_user_templates_dir,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the CLAWCODEX_* env vars so default-resolution tests
    are not affected by ambient values from the host shell.
    """
    monkeypatch.delenv(CLAWCODEX_CONFIG_DIR_ENV, raising=False)
    monkeypatch.delenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, raising=False)


# ---------------------------------------------------------------------------
# Env-var fallback
# ---------------------------------------------------------------------------


def test_user_dir_uses_env_var_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, str(tmp_path))
    result = get_user_templates_dir()
    assert result == (tmp_path / TEMPLATES_SUBDIR).resolve()


def test_user_dir_ignores_empty_env(clean_env: None, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, "")
    # Empty env → falls back to ~/.clawcodex/templates/.
    result = get_user_templates_dir()
    assert result == (fake_home / ".clawcodex" / TEMPLATES_SUBDIR).resolve()


def test_user_dir_ignores_whitespace_env(clean_env: None, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLAWCODEX_CONFIG_DIR_ENV, "   ")
    result = get_user_templates_dir()
    assert result == (fake_home / ".clawcodex" / TEMPLATES_SUBDIR).resolve()


def test_managed_dir_uses_env_var_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CLAWCODEX_MANAGED_CONFIG_DIR_ENV, str(tmp_path))
    result = get_managed_templates_dir()
    assert result == (tmp_path / TEMPLATES_SUBDIR).resolve()


def test_managed_dir_falls_back_to_etc(clean_env: None) -> None:
    result = get_managed_templates_dir()
    assert result == Path("/etc/clawcodex") / TEMPLATES_SUBDIR


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    return home


# ---------------------------------------------------------------------------
# Project walker
# ---------------------------------------------------------------------------


def test_project_walks_up_to_git_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    (repo / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    (nested / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    # The outer (parent-of-repo) templates must NOT leak; everything
    # inside the repo (cwd's own .clawcodex/templates + the repo
    # root's) appears, outermost-first so the innermost wins via
    # overwrite=True.
    assert get_project_templates_dirs(cwd=nested) == [
        (repo / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).resolve(),
        (nested / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).resolve(),
    ]


def test_project_walks_to_fs_root_when_no_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When no ``.git`` exists, the walker climbs to the filesystem
    root and returns every matching ancestor dir.
    """
    a = tmp_path / "a"
    a.mkdir()
    b = a / "b"
    b.mkdir()
    c = b / "c"
    c.mkdir()
    (a / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    (b / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    (c / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).mkdir(parents=True)
    result = get_project_templates_dirs(cwd=c)
    # outermost-first ordering
    assert result == [
        (a / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).resolve(),
        (b / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).resolve(),
        (c / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR).resolve(),
    ]


def test_project_no_match_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    assert get_project_templates_dirs(cwd=proj) == []


def test_project_nonexistent_cwd_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cwd that does not exist returns an empty list, not a raise."""
    assert get_project_templates_dirs(cwd=tmp_path / "does-not-exist") == []


def test_project_default_uses_cwd_when_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``cwd=None`` resolves through :func:`os.getcwd`."""
    monkeypatch.chdir(tmp_path)
    assert get_project_templates_dirs() == []
