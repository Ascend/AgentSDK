#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Repair SDK imports that ``sop convert`` did not install into the bundle venv."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .bundle_venv import install_bundle_packages
from .core.bundle_manifest import read_bundle_manifest, write_bundle_manifest
from .core.runtime_paths import normalize_runtime_path

_MISSING_MODULE_RE = re.compile(r"(?:ModuleNotFoundError:\s*)?No module named ['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]")
_IMPORT_TO_PIP = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "opentelemetry.sdk": "opentelemetry-sdk",
}


def pip_name_for_missing_module(name: str) -> str:
    """Map an import name to the pip distribution to install."""

    name = (name or "").strip()
    if not name:
        return name
    if name in _IMPORT_TO_PIP:
        return _IMPORT_TO_PIP[name]
    top = name.split(".", 1)[0]
    if top in _IMPORT_TO_PIP:
        return _IMPORT_TO_PIP[top]
    if "." in name:
        first, second, *_ = name.split(".")
        return f"{first}-{second}"
    return name


def missing_sdk_dependency_message(missing: str, *, python_exe: str) -> str:
    """Agent-facing error that includes a one-package pip Command."""

    pip_name = pip_name_for_missing_module(missing)
    return (
        "missing_sdk_dependency: No module named "
        f"{missing!r}. Install only {pip_name} into the bundle venv, then retry "
        f"the same tool with the same arguments. Command: {python_exe} -m pip install {pip_name}"
    )


def command_python_for_bundle(
    bundle_dir: str | Path | None = None,
    *,
    script_path: str | Path | None = None,
    module: object | None = None,
) -> str:
    """Python that pip-installs into the bundle venv, not the conversation interpreter."""

    if module is not None and not bundle_dir:
        bundle_dir = getattr(module, "_BUNDLE_DIR", None) or None
    if script_path is not None and not bundle_dir:
        bundle_dir = wrapper_bundle_dir(script_path)
    if bundle_dir:
        from .bundle_venv import bundle_venv_python

        return str(bundle_venv_python(bundle_dir))
    if module is not None:
        raw = getattr(module, "_BUNDLE_VENV_PYTHON", "") or ""
        if raw:
            return str(raw)
    if script_path is not None:
        venv = wrapper_bundle_venv_python(script_path)
        if venv is not None:
            return str(venv)
    return sys.executable


_SOURCE_FILE_RE = re.compile(r"_SOURCE_FILE\s*=\s*Path\(\s*_normalize_bootstrap_path\(\s*r(['\"])(.+?)\1\s*\)\s*\)")
_BUNDLE_DIR_RE = re.compile(r"_BUNDLE_DIR\s*=\s*_normalize_bootstrap_path\(\s*r(['\"])(.+?)\1\s*\)")
_BUNDLE_VENV_PYTHON_RE = re.compile(r"_BUNDLE_VENV_PYTHON\s*=\s*_normalize_bootstrap_path\(\s*r(['\"])(.+?)\1\s*\)")


@dataclass(frozen=True)
class RepairOutcome:
    installed: bool
    error_code: str | None = None
    message: str = ""


def parse_missing_module(text: str) -> str | None:
    """Return the top-level module name from a ``ModuleNotFoundError`` traceback."""

    if not text:
        return None
    match = _MISSING_MODULE_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def top_level_imported_modules(source_file: str | Path) -> frozenset[str]:
    """Return top-level imported package names from *source_file*."""

    path = Path(source_file)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root:
                    names.add(root)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".", 1)[0]
            if root:
                names.add(root)
    return frozenset(names)


def wrapper_source_file(script_path: str | Path) -> Path | None:
    """Read ``_SOURCE_FILE`` from a generated SOP wrapper script."""

    return _wrapper_bootstrap_path(script_path, _SOURCE_FILE_RE)


def wrapper_bundle_dir(script_path: str | Path) -> Path | None:
    """Read ``_BUNDLE_DIR`` from a generated SOP wrapper script."""

    return _wrapper_bootstrap_path(script_path, _BUNDLE_DIR_RE)


def wrapper_bundle_venv_python(script_path: str | Path) -> Path | None:
    """Read ``_BUNDLE_VENV_PYTHON`` from a generated SOP wrapper script."""

    return _wrapper_bootstrap_path(script_path, _BUNDLE_VENV_PYTHON_RE)


def _wrapper_bootstrap_path(script_path: str | Path, pattern: re.Pattern[str]) -> Path | None:
    path = Path(script_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = pattern.search(text)
    if match is None:
        return None
    raw = match.group(2)
    if not raw:
        return None
    try:
        return normalize_runtime_path(raw)
    except OSError:
        return Path(raw)


def append_sdk_requirement(bundle_dir: str | Path, package: str) -> None:
    """Record a runtime extra package without mutating convert-time ``sdk_requirements``."""

    bundle_path = Path(bundle_dir)
    manifest = read_bundle_manifest(bundle_path)
    if manifest is None:
        return
    package = package.strip()
    if not package or package in manifest.sdk_requirements or package in manifest.runtime_extra_requirements:
        return
    write_bundle_manifest(
        bundle_path,
        sdk_source_dir=manifest.sdk_source_dir,
        bundle_id=manifest.bundle_id,
        workflow_yaml=manifest.workflow_yaml,
        bridge_script=manifest.bridge_script,
        workflow_mode=manifest.workflow_mode,
        sdk_requirements=manifest.sdk_requirements,
        runtime_extra_requirements=(*manifest.runtime_extra_requirements, package),
        bundle_venv_dir=manifest.bundle_venv_dir,
    )


def _is_stdlib(name: str) -> bool:
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is not None and name in stdlib:
        return True
    return name in {"__future__", "builtins"}


def _is_install_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


class MissingDependencyRepair:
    """Install a missing import into the bundle venv, never the host interpreter."""

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir)

    def repair(
        self,
        missing: str,
        source_file: str | Path | None = None,
    ) -> RepairOutcome:
        full = (missing or "").strip()
        top = full.split(".", 1)[0]
        if not _is_install_name(top) or _is_stdlib(top):
            return RepairOutcome(
                installed=False,
                error_code="sdk_dependency_not_declared",
                message=f"{top!r} is not a third-party import that can be pip-installed",
            )
        _ = source_file
        pip_name = pip_name_for_missing_module(full)
        try:
            install_bundle_packages(self.bundle_dir, [pip_name])
        except Exception as exc:  # noqa: BLE001
            return RepairOutcome(
                installed=False,
                error_code="sdk_dependency_install_failed",
                message=f"could not pip-install {pip_name} into the bundle venv ({exc})",
            )
        append_sdk_requirement(self.bundle_dir, pip_name)
        return RepairOutcome(installed=True)
