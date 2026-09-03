#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Tests for per-bundle virtualenv runtime behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from extensions.sop_converter import bundle_venv
from extensions.sop_converter.runtime_paths import (
    windows_path_to_wsl_path,
    wsl_path_to_windows_path,
)
from extensions.sop_converter.core.sdk_dependency_resolver import SdkDependencySpec


def test_wsl_and_windows_path_converters() -> None:
    assert wsl_path_to_windows_path("/mnt/d/projects/clawcodex") == ("D:\\projects\\clawcodex")
    assert wsl_path_to_windows_path("/home/user/project") is None
    assert windows_path_to_wsl_path("D:\\projects\\clawcodex") == ("/mnt/d/projects/clawcodex")
    assert windows_path_to_wsl_path("/mnt/d/projects/clawcodex") is None


def test_bundle_venv_dir_normalizes_wsl_path_on_windows() -> None:
    if os.name != "nt":
        return

    path = bundle_venv.bundle_venv_dir("/mnt/d/projects/clawcodex/bundle")

    assert str(path) == "D:\\projects\\clawcodex\\bundle\\.venv"


def test_bundle_venv_dir_wsl_includes_bundle_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bundle_venv, "is_wsl_runtime", lambda: True)
    monkeypatch.setattr(bundle_venv.Path, "home", staticmethod(lambda: tmp_path))
    bundle_dir = Path("/mnt/d/projects/clawcodex/JiuwenAgent_v7.18")

    path = bundle_venv.bundle_venv_dir(bundle_dir)

    assert path.parent == tmp_path / ".cache" / "clawcodex" / "bundle-venvs"
    assert path.name.startswith("JiuwenAgent_v7.18-")
    assert len(path.name.rsplit("-", 1)[-1]) == 16


def test_bundle_venv_dir_wsl_reuses_legacy_hash_dir(tmp_path: Path, monkeypatch) -> None:
    from extensions.sop_converter.runtime_paths import normalize_runtime_path

    monkeypatch.setattr(bundle_venv, "is_wsl_runtime", lambda: True)
    monkeypatch.setattr(bundle_venv.Path, "home", staticmethod(lambda: tmp_path))
    bundle_dir = Path("/mnt/d/projects/clawcodex/MyBundle")
    _, legacy_hash = bundle_venv._wsl_bundle_venv_slug(normalize_runtime_path(bundle_dir))
    legacy = tmp_path / ".cache" / "clawcodex" / "bundle-venvs" / legacy_hash
    legacy.mkdir(parents=True)

    path = bundle_venv.bundle_venv_dir(bundle_dir)

    assert path == legacy


def test_bundle_venv_site_packages_uses_current_platform(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"

    paths = bundle_venv.bundle_venv_site_packages(bundle_dir)

    assert paths
    if os.name == "nt":
        assert paths[0] == bundle_dir / ".venv" / "Lib" / "site-packages"
    else:
        py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        assert paths[0] == bundle_dir / ".venv" / "lib" / py_tag / "site-packages"


def test_last_building_package_from_uv_and_pip_output() -> None:
    uv_tail = "Resolved 48 packages in 3.53s\n   Building pyarrow==25.0.0\n"
    assert bundle_venv._last_building_package(uv_tail) == ("pyarrow", "25.0.0")

    pip_tail = "Building wheel for grpcio (pyproject.toml)\n"
    assert bundle_venv._last_building_package(pip_tail) == ("grpcio", None)

    assert bundle_venv._last_building_package("Downloading ray\n") is None


def test_activate_bundle_venv_imports_moves_site_packages_to_front(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_venv._reset_bundle_venv_import_state()
    bundle_dir = tmp_path / "bundle"
    site_packages = bundle_venv.bundle_venv_site_packages(bundle_dir)[0]
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["host-env", str(site_packages), "tail"])

    added = bundle_venv.activate_bundle_venv_imports(bundle_dir)

    assert added == (str(site_packages),)
    assert sys.path[0] == str(site_packages)
    assert sys.path.count(str(site_packages)) == 1
    bundle_venv._reset_bundle_venv_import_state()


def test_activate_bundle_venv_imports_warns_on_second_bundle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    bundle_venv._reset_bundle_venv_import_state()
    first = tmp_path / "bundle-a"
    second = tmp_path / "bundle-b"
    first_site = bundle_venv.bundle_venv_site_packages(first)[0]
    second_site = bundle_venv.bundle_venv_site_packages(second)[0]
    first_site.mkdir(parents=True)
    second_site.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["host-env"])

    bundle_venv.activate_bundle_venv_imports(first)
    bundle_venv.activate_bundle_venv_imports(second)

    captured = capsys.readouterr()
    assert "activating SDK dependencies for a second bundle" in captured.err
    bundle_venv._reset_bundle_venv_import_state()


def test_ensure_bundle_venv_and_reexec_activates_imports_in_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_ensure(bundle_dir, deps, *, force=False):
        del force
        calls.append(("ensure", str(bundle_dir)))
        return tmp_path / "venv" / "bin" / "python"

    def fake_activate(bundle_dir):
        calls.append(("activate", str(bundle_dir)))
        return ()

    def fail_execv(*_args, **_kwargs):
        raise AssertionError("in-process bundle setup attempted os.execv")

    monkeypatch.setattr(bundle_venv, "ensure_bundle_venv", fake_ensure)
    monkeypatch.setattr(bundle_venv, "activate_bundle_venv_imports", fake_activate)
    monkeypatch.setattr(bundle_venv.os, "execv", fail_execv)

    deps = SdkDependencySpec(
        requirements=("demo-dep>=1",),
        source="test",
        raw_path="",
    )
    with bundle_venv.in_process_bundle_venv_reexec():
        bundle_venv.ensure_bundle_venv_and_reexec(tmp_path / "bundle", deps)

    assert calls == [
        ("ensure", str((tmp_path / "bundle").resolve())),
        ("activate", str((tmp_path / "bundle").resolve())),
    ]


def test_ensure_bundle_venv_resets_wrong_platform_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    venv_dir = bundle_dir / ".venv"
    wrong_python = venv_dir / "bin" / "python" if os.name == "nt" else venv_dir / "Scripts" / "python.exe"
    wrong_python.parent.mkdir(parents=True)
    wrong_python.write_text("wrong platform", encoding="utf-8")

    def fake_create_venv(path: Path) -> None:
        python_path = bundle_venv.bundle_venv_python(path.parent)
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.write_text("current platform", encoding="utf-8")

    monkeypatch.setattr(bundle_venv, "_create_venv", fake_create_venv)
    monkeypatch.setattr(bundle_venv, "_install_requirements", lambda *_args, **_kwargs: None)

    python_path = bundle_venv.ensure_bundle_venv(
        bundle_dir,
        SdkDependencySpec(
            requirements=("openai>=1",),
            source="test",
            raw_path="",
        ),
    )

    assert python_path == bundle_venv.bundle_venv_python(bundle_dir)
    assert python_path.is_file()
    assert not wrong_python.exists()
    assert (venv_dir / ".bundle-venv-ready").is_file()


def test_install_wheel_preference_flags_uv_vs_pip() -> None:
    assert bundle_venv._install_wheel_preference_flags(use_uv=True) == []
    assert bundle_venv._install_wheel_preference_flags(use_uv=False) == ["--prefer-binary"]


def test_pypi_extra_index_flags_adds_pypi_for_mirror() -> None:
    flags = bundle_venv._pypi_extra_index_flags(
        use_uv=True,
        index_url="https://pypi.tuna.tsinghua.edu.cn/simple",
    )
    assert flags == ["--extra-index-url", "https://pypi.org/simple"]


def test_pypi_extra_index_flags_skips_when_primary_is_pypi() -> None:
    assert (
        bundle_venv._pypi_extra_index_flags(
            use_uv=True,
            index_url="https://pypi.org/simple",
        )
        == []
    )


def test_subprocess_env_sets_uv_http_timeout_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("UV_HTTP_TIMEOUT", raising=False)
    monkeypatch.setenv("CLAWCODEX_BUNDLE_VENV_UV_HTTP_TIMEOUT", "180")

    env = bundle_venv._subprocess_env(["uv", "pip", "install"])

    assert env is not None
    assert env["UV_HTTP_TIMEOUT"] == "180"


def test_subprocess_env_preserves_existing_uv_http_timeout(monkeypatch) -> None:
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "90")
    monkeypatch.setenv("CLAWCODEX_BUNDLE_VENV_UV_HTTP_TIMEOUT", "180")

    env = bundle_venv._subprocess_env(["/root/.local/bin/uv", "venv", "/tmp/x"])

    assert env is not None
    assert env["UV_HTTP_TIMEOUT"] == "90"


def test_subprocess_env_not_used_for_non_uv_commands() -> None:
    assert bundle_venv._subprocess_env([sys.executable, "-c", "pass"]) is None


def test_run_command_streamed_writes_child_output_to_stderr(capsys) -> None:
    returncode, tail = bundle_venv._run_command_streamed(
        [sys.executable, "-c", "print('bundle output')"],
        "Testing streamed command",
    )

    captured = capsys.readouterr()
    assert returncode == 0
    assert "bundle output" in tail
    assert "bundle output" in captured.err


def test_subprocess_env_sets_uv_concurrent_downloads_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("UV_CONCURRENT_DOWNLOADS", raising=False)
    monkeypatch.setenv("CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_DOWNLOADS", "32")

    env = bundle_venv._subprocess_env(["uv", "pip", "install"])

    assert env is not None
    assert env["UV_CONCURRENT_DOWNLOADS"] == "32"


def test_subprocess_env_preserves_existing_uv_concurrent_downloads(monkeypatch) -> None:
    monkeypatch.setenv("UV_CONCURRENT_DOWNLOADS", "8")
    monkeypatch.setenv("CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_DOWNLOADS", "32")

    env = bundle_venv._subprocess_env(["/root/.local/bin/uv", "pip", "install"])

    assert env is not None
    assert env["UV_CONCURRENT_DOWNLOADS"] == "8"


def test_subprocess_env_sets_uv_concurrent_builds_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("UV_CONCURRENT_BUILDS", raising=False)
    monkeypatch.setenv("CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_BUILDS", "16")

    env = bundle_venv._subprocess_env(["uv", "pip", "install"])

    assert env is not None
    assert env["UV_CONCURRENT_BUILDS"] == "16"


def test_uv_concurrent_downloads_default() -> None:
    assert bundle_venv._uv_concurrent_downloads() == 16


def test_uv_concurrent_builds_default() -> None:
    assert bundle_venv._uv_concurrent_builds() == 8
