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

# AgentSDK publishes this standalone src-layout package across independent Parts; the complete
# ClawCodex source and focused tests validate imports and dynamic patterns during migration.
# The target hook also enables legacy default diagnostics beyond its declared high-value set.
# pylint: disable=W0621,relative-beyond-top-level
"""Shared fixtures for LKB Plan Graph tests.

Isolation rules (spec §11.3):
- Every test uses a temporary HOME redirected via CLAWCODEX_HOME.
- Deterministic clock and ID generators for reproducible snapshots.
- ANSI output is disabled so ASCII golden tests are stable.
- A failpoint registry is provided for Phase-2 write-path injection.
- Feature-gate state is reset before every test so flag toggles never leak
  across modules (the clawcodex FeatureRegistry is a process singleton;
  programmatic overrides set by one test otherwise bleed into the next).

Reusable helper classes live in ``tests/_support.py`` and are imported via
relative imports (``from ._support import ...``); do not use absolute
``tests.X`` imports - that collides with the repo-root ``tests/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from ._support import (
    DeterministicClock,
    DeterministicIdFactory,
    Failpoint,
)

__all__ = [
    "tmp_home",
    "tmp_lkb_root",
    "deterministic_clock",
    "deterministic_id",
    "ansi_off",
    "Failpoint",
    "failpoint",
]


# ---------------------------------------------------------------------------
# Feature-gate isolation (autouse) - prevents cross-test flag leakage
# ---------------------------------------------------------------------------

_LKB_FLAG_ENVS = ("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH",)


@pytest.fixture(autouse=True)
def _reset_lkb_feature_flags(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Reset LKB feature-gate state before each test.

    Clears both the env vars and any programmatic overrides on the clawcodex
    FeatureRegistry singleton, so a test that enables a flag cannot leak it
    into a later test. Tests that need a flag on set it themselves after this.
    """
    for var in _LKB_FLAG_ENVS:
        monkeypatch.delenv(var, raising=False)
    try:
        from clawcodex_ext.feature_gate import get_registry  # type: ignore[import]

        reg = get_registry()
        overrides = getattr(reg, "_overrides", None)
        if isinstance(overrides, dict):
            overrides.clear()
    except Exception:
        # Standalone lkb (no clawcodex_ext) or registry shape changed - env reset suffices.
        pass
    yield


# ---------------------------------------------------------------------------
# Isolation: temporary HOME / LKB root
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect HOME and CLAWCODEX_HOME to a temp directory.

    All ~/.clawcodex resolution follows either HOME or CLAWCODEX_HOME;
    both are patched so tests never touch the user's real config.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows compat
    monkeypatch.setenv("CLAWCODEX_HOME", str(home))
    yield home


@pytest.fixture
def tmp_lkb_root(tmp_home: Path) -> Path:
    """A dedicated temp directory for LKB board storage.

    Lives under tmp_home so it is covered by the HOME redirection.
    The directory is created but left empty - boards are created by
    individual tests as needed.
    """
    root = tmp_home / "lkb"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Deterministic clock
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_clock() -> DeterministicClock:
    """Yield a DeterministicClock - callable, with advance() and peek()."""
    return DeterministicClock()


# ---------------------------------------------------------------------------
# Deterministic ID generator
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_id() -> DeterministicIdFactory:
    """Yield a callable DeterministicIdFactory."""
    return DeterministicIdFactory()


# ---------------------------------------------------------------------------
# ANSI suppression
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ansi_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force no-ANSI / plain-text rendering for all tests.

    Rich/color libraries honour these env vars. Tests that compare ASCII
    golden output rely on this being on.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "0")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("TERMINAL_WIDTH", "80")


# ---------------------------------------------------------------------------
# Failpoint registry (spec §11.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def failpoint() -> Generator[Failpoint, None, None]:
    """Yield a fresh Failpoint registry, cleared after each test."""
    fp = Failpoint()
    yield fp
    fp.clear()
