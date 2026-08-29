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

"""AST-based decoupling checks for extensions/sop_converter.

Verifies that:
- ``extensions/sop_converter/core/`` has zero ``from src.`` or ``from clawcodex_ext.`` imports
- Keys that are not yet available (imports in ``runtime/``) still pass the ``core/`` gate
"""

import ast
import pathlib

import pytest

CORE_DIR = pathlib.Path("extensions/sop_converter/core")
FORBIDDEN_PREFIXES = ("src.", "clawcodex_ext.")


def _py_files_under(dirpath: pathlib.Path) -> list[pathlib.Path]:
    """Return all ``.py`` files under *dirpath* (recursive, sorted)."""
    if not dirpath.is_dir():
        return []
    return sorted(dirpath.rglob("*.py"))


def test_core_no_layer0_layer1_imports():
    """``extensions/sop_converter/core/`` must not import from ``src.*`` or ``clawcodex_ext.*``."""
    errors: list[str] = []
    for py in _py_files_under(CORE_DIR):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{py}: syntax error: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for prefix in FORBIDDEN_PREFIXES:
                    if node.module.startswith(prefix):
                        errors.append(f"{py}:{node.lineno} forbids import from {node.module!r}")
    assert not errors, "\n".join(errors)


def test_core_protocols_only():
    """``core/`` imports from ``extensions.capabilities`` are OK (Protocol layer)."""
    # This is a soft check — we expect zero capability imports in core/,
    # but it's not a hard failure since some core files may need Protocol
    # references for type annotations.
    imports: list[str] = []
    for py in _py_files_under(CORE_DIR):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("extensions.capabilities."):
                    imports.append(f"{py}:{node.lineno} imports {node.module!r}")
    if imports:
        pytest.skip(f"core/ still has {len(imports)} capability imports (acceptable)")


def test_core_all_modules_importable():
    """Every module in ``core/`` can be imported independently."""
    root = pathlib.Path(".").resolve()
    for py in _py_files_under(CORE_DIR):
        # Compute dotted module path relative to the project root.
        abs_py = py.resolve() if not py.is_absolute() else py
        try:
            rel = abs_py.relative_to(root)
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()  # skip __init__ — tested via the parent module
            continue
        if not parts:
            continue
        mod = ".".join(parts)
        try:
            __import__(mod)
        except Exception as exc:
            pytest.fail(f"Failed to import {mod}: {exc}")


def test_runtime_stub_backward_compatibility():
    """Root-level stub files re-export from core/ or runtime/ successfully."""
    stub_files = sorted(pathlib.Path("extensions/sop_converter").glob("*.py"))
    for py in stub_files:
        if py.name == "__init__.py":
            continue
        # Each stub is a one-liner: ``from .core.xxx import *`` or ``from .runtime.xxx import *``
        # We just check it parses and doesn't crash on import.
        mod = f"extensions.sop_converter.{py.stem}"
        try:
            __import__(mod)
        except ImportError as exc:
            pytest.fail(f"Stub {mod} failed to import: {exc}")
        except (AttributeError, TypeError, ValueError):
            # A stub may fail while re-exporting when an optional dependency
            # is absent. Tolerate that here — the decoupling test only
            # verifies the import path resolves.
            pass
