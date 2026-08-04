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

"""Load portable text resource trees for bundled skills."""

from __future__ import annotations

from importlib.resources import files
from pathlib import PurePosixPath


_PACKAGE_ONLY_NAMES = frozenset({"__init__.py", "__pycache__"})


def load_bundled_text_resources(package: str) -> dict[str, str]:
    """Load a bundled skill's portable text resource tree.

    Python package markers and bytecode caches are packaging details rather
    than skill resources, so they are never exposed in the extracted tree.

    Args:
        package: Importable Python package containing only portable text
            resources and optional Python package markers.

    Returns:
        Resource contents keyed by relative POSIX path.

    Raises:
        FileNotFoundError: If the package contains no portable resources.
    """

    root = files(package)
    pending = [(root, PurePosixPath())]
    loaded: dict[str, str] = {}

    while pending:
        directory, prefix = pending.pop()
        for entry in sorted(directory.iterdir(), key=lambda item: item.name, reverse=True):
            if entry.name in _PACKAGE_ONLY_NAMES:
                continue
            relative_path = prefix / entry.name
            if entry.is_dir():
                pending.append((entry, relative_path))
                continue
            if entry.is_file():
                loaded[relative_path.as_posix()] = entry.read_text(encoding="utf-8")

    if not loaded:
        raise FileNotFoundError(f"bundled skill resource package is empty: {package}")
    return dict(sorted(loaded.items()))


__all__ = ["load_bundled_text_resources"]
