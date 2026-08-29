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

"""Pre-publish wheel sanity checks.

NOT in CI — run locally before pushing a release tag:

    python -m build
    python -m pytest tests/release_smoke/ -v

These tests verify the *artifact* properties of the built wheel:

1. METADATA fields match pyproject [project] declarations.
2. Console-script entry points (``clawcodex-dev``, etc.) are bundled.
3. ``RELEASE_TAG`` env var correctly freezes ``__version__`` in the
   installed wheel (so a wheel built today + installed tomorrow still
   reports the tag's date).

CI's ``release-preflight.yml`` already runs ``twine check`` + a CLI
smoke; this directory is the *artifact-level* belt-and-suspenders layer
that runs on the maintainer's machine before tagging.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from email import message_from_string
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "dist"


# Skip the whole module if no wheel exists — pytest will report
# ``6 skipped`` rather than 6 errors when the maintainer hasn't built
# the wheel yet.  The intended workflow is:
#     python -m build
#     pytest tests/release_smoke/
@pytest.fixture(scope="module")
def wheel_path() -> Path:
    if not DIST_DIR.exists():
        pytest.skip("dist/ not found; run `python -m build` first")
    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not wheels:
        pytest.skip(f"no wheel in {DIST_DIR}; run `python -m build` first")
    return wheels[-1]


@pytest.fixture(scope="module")
def wheel_metadata(wheel_path: Path) -> dict[str, str]:
    """Parse the wheel's METADATA file into a dict."""
    with zipfile.ZipFile(wheel_path) as zf:
        # wheel METADATA naming: <normalized_name>-<version>.dist-info/METADATA
        metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert metadata_names, f"No METADATA found in {wheel_path}"
        raw = zf.read(metadata_names[0]).decode("utf-8")
    msg = message_from_string(raw)
    return {k: (v or "").strip() for k, v in msg.items()}


class TestWheelMetadata:
    """Wheel METADATA must match pyproject [project] declarations."""

    def test_metadata_has_name(self, wheel_metadata: dict[str, str]) -> None:
        """Name field is ``clawcodex-dev-mind`` (matches pyproject + publish target)."""
        assert wheel_metadata["Name"] == "clawcodex-dev-mind", (
            f"unexpected wheel Name: {wheel_metadata['Name']!r}; pyproject [project] name = 'clawcodex-dev-mind'"
        )

    def test_metadata_has_valid_version(self, wheel_metadata: dict[str, str]) -> None:
        """Version is CalVer ``YYYY.M.D`` (the only supported scheme)."""
        version = wheel_metadata["Version"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
            f"Version {version!r} is not CalVer YYYY.M.D; clawcodex uses CalVer + tag freeze exclusively"
        )

    def test_metadata_requires_python(self, wheel_metadata: dict[str, str]) -> None:
        """Requires-Python is ``>=3.10`` (matches pyproject [project])."""
        req = wheel_metadata.get("Requires-Python", "")
        # Accept ``>=3.10``, ``>=3.10,<4``, etc.
        assert re.search(r">=\s*3\.10", req), (
            f"Requires-Python {req!r} does not declare >=3.10; check pyproject.toml [project] requires-python"
        )


class TestWheelConsoleScripts:
    """Console-script entry points must be bundled inside the wheel."""

    def test_wheel_contains_clawcodex_dev_entry_point(self, wheel_path: Path) -> None:
        """``clawcodex-dev`` console script exists in the wheel RECORD.

        The console script is registered via pyproject
        ``[project.scripts]``; PEP 427 stores it as
        ``<name>.dist-info/entry_points.txt``. We assert that file
        mentions ``clawcodex-dev`` (or any other declared script),
        which catches a missed entry_points registration.
        """
        with zipfile.ZipFile(wheel_path) as zf:
            entry_point_files = [n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt")]
            assert entry_point_files, (
                f"No entry_points.txt in {wheel_path}; pyproject [project.scripts] registration is missing"
            )
            content = "\n".join(zf.read(n).decode("utf-8") for n in entry_point_files)
            assert "clawcodex-dev" in content, (
                f"clawcodex-dev console script not declared in entry_points.txt; found: {content[:500]!r}"
            )

    def test_wheel_record_includes_console_script_directory(self, wheel_path: Path) -> None:
        """The RECORD file lists all bundled files (catch empty wheels).

        A wheel that bundles only metadata but no actual code would
        pass the METADATA check above but fail at pip install time.
        Verify the RECORD has at least 10 entries (a real clawcodex
        wheel has hundreds of modules + a venv-installed console
        script wrapper).
        """
        with zipfile.ZipFile(wheel_path) as zf:
            record_files = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
            assert record_files, f"No RECORD in {wheel_path}"
            record_content = zf.read(record_files[0]).decode("utf-8")
            entry_count = len([line for line in record_content.splitlines() if line.strip()])
            assert entry_count >= 10, (
                f"Wheel RECORD has only {entry_count} entries; "
                f"expected at least 10 (a real clawcodex wheel has hundreds)"
            )


class TestInstalledSopConvert:
    """The installed console script must work without the repository on sys.path."""

    def test_sop_convert_from_external_cwd(self, wheel_path: Path) -> None:
        import tempfile

        fixture = REPO_ROOT / "tests" / "fixtures" / "fixture_arc_project"
        with tempfile_venv() as venv_python, tempfile.TemporaryDirectory(prefix="cx-sop-convert-") as temp_dir:
            _pip_install(venv_python, str(wheel_path))
            work_dir = Path(temp_dir)
            command = _console_script(venv_python, "clawcodex-dev")
            result = subprocess.run(
                [
                    str(command),
                    "sop",
                    "convert",
                    str(fixture),
                    "--strategy",
                    "keyword",
                    "--out",
                    str(work_dir / "bundle"),
                    "--skills",
                    str(work_dir / "skills"),
                ],
                cwd=work_dir,
                env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}},
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, result.stdout + result.stderr
            assert (work_dir / "bundle").is_dir()


class TestReleaseTagFreeze:
    """``RELEASE_TAG`` env var must correctly freeze ``__version__``.

    When CI publishes a tagged release, ``RELEASE_TAG=v2026.6.24`` is
    exported into the build environment, and the wheel's
    ``__version__`` is pinned to ``2026.6.24`` even if rebuilt from the
    same commit a week later. We assert that contract end-to-end via
    subprocess so the test exercises the same import path users will.
    """

    def test_release_tag_freezes_version(self, wheel_path: Path) -> None:
        """With ``RELEASE_TAG=v2099.1.1``, a fresh venv reports 2099.1.1."""
        # Use a sentinel date far in the future so the test doesn't
        # silently pass on a coincidentally matching CalVer.
        sentinel = "v2099.1.1"
        with tempfile_venv() as venv_python:
            _pip_install(venv_python, str(wheel_path))
            out = subprocess.check_output(
                [
                    str(venv_python),
                    "-c",
                    "from clawcodex_ext._version import __version__; print(__version__)",
                ],
                env={**os.environ, "RELEASE_TAG": sentinel},
                text=True,
            ).strip()
            assert out == "2099.1.1", (
                f"RELEASE_TAG={sentinel} did not freeze __version__; "
                f"got {out!r}. Check clawcodex_ext/_version.py:_version() "
                f"priority order."
            )

    def test_no_release_tag_uses_calver(self, wheel_path: Path) -> None:
        """Without ``RELEASE_TAG``, ``__version__`` is today's CalVer."""
        with tempfile_venv() as venv_python:
            _pip_install(venv_python, str(wheel_path))
            out = subprocess.check_output(
                [
                    str(venv_python),
                    "-c",
                    "from clawcodex_ext._version import __version__; print(__version__)",
                ],
                # Strip RELEASE_TAG defensively in case the test runner
                # exports it for some other reason.
                env={k: v for k, v in os.environ.items() if k != "RELEASE_TAG"},
                text=True,
            ).strip()
            assert re.fullmatch(r"\d+\.\d+\.\d+", out), (
                f"__version__ {out!r} is not CalVer YYYY.M.D; "
                f"clawcodex_ext/_version.py:_version() must fall back to "
                f"date.today() when RELEASE_TAG is unset"
            )


# ── helpers ────────────────────────────────────────────────────────


def tempfile_venv() -> "subprocess.Popen | any":
    """Create a throwaway venv, yield the python interpreter path.

    On Linux/macOS uses ``python -m venv``; cleans up on teardown via
    ``tmp_path_factory``. We deliberately do NOT use ``pytest``
    fixtures here because we need a real subprocess that mimics what
    pip + a user would see.
    """
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="cx-release-smoke-"))

    def cleanup() -> None:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # No try/finally — pytest fixtures handle cleanup. Return a
    # small context-manager-like object instead.
    class _Venv:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.bin = root / "bin"
            self.python = self.bin / "python"

        def __enter__(self) -> Path:
            subprocess.check_call(
                [sys.executable, "-m", "venv", str(self.root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.python

        def __exit__(self, *exc: object) -> None:
            cleanup()

    return _Venv(tmpdir)


def _pip_install(python: Path, *args: str) -> None:
    """Run ``python -m pip install`` with the given args."""
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "--quiet", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _console_script(python: Path, name: str) -> Path:
    candidates = (
        python.parent / name,
        python.parent / f"{name}.exe",
        python.parent / f"{name}-script.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(f"console script {name!r} not found beside {python}")
