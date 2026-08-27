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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402
"""Filesystem containment helpers shared by LKB lifecycle operations."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .json_store import JsonBoardStore


__all__ = ["lkb_root_for_store", "safe_chain"]
_WINDOWS_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def lkb_root_for_store(store: JsonBoardStore) -> Path:
    """Return the managed LKB root for a public JsonBoardStore instance."""
    return store.board_dir.parent.parent


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _WINDOWS_REPARSE_ATTRIBUTE)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _contained(root: Path, path: Path) -> bool:
    root_abs = _lexical_absolute(root)
    path_abs = _lexical_absolute(path)
    try:
        return os.path.commonpath((root_abs, path_abs)) == os.fspath(root_abs)
    except ValueError:
        return False


def safe_chain(root: Path, target: Path, *, descendants: bool = False) -> bool:
    root_abs = _lexical_absolute(root)
    target_abs = _lexical_absolute(target)
    if not _contained(root_abs, target_abs):
        return False
    try:
        relative = target_abs.relative_to(root_abs)
    except ValueError:
        return False
    current = root_abs
    if _is_reparse(current):
        return False
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse(current):
            return False
    if descendants and target_abs.is_dir():
        try:
            for directory, dirs, files in os.walk(target_abs, followlinks=False):
                directory_path = Path(directory)
                for name in (*dirs, *files):
                    if _is_reparse(directory_path / name):
                        return False
        except OSError:
            return False
    return True


# Compatibility aliases for split migration modules that still import the
# original private spellings. New cross-module callers use the public API.
_lkb_root_for_store = lkb_root_for_store
_safe_chain = safe_chain
