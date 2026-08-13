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

"""Shared file-IO and path helpers for the orchestrator subsystem.

Centralises the UTF-8 text read/write, JSON read, and rmtree patterns that
were duplicated across ``workspace``, ``workspace_verify``,
``workspace_locator``, ``report_writer``, and ``rules_learner``.  These
helpers are consumed by other modules in the package via direct import and
are intentionally public within the subsystem.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

# X-01: shared constant for the per-workspace orchestrator metadata directory.
ORCHESTRATOR_WORKSPACE_DIRNAME = ".orchestrator_workspace"


def write_text_utf8(path: Path, content: str) -> None:
    """Write *content* to *path* using UTF-8 encoding."""
    path.write_text(content, encoding="utf-8")


def read_text_utf8(path: Path) -> str:
    """Read text from *path* using UTF-8 encoding."""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read and parse a JSON file using UTF-8 encoding."""
    return json.loads(read_text_utf8(path))


def safe_rmtree(path: Path, ignore_errors: bool = False) -> None:
    """Remove a directory tree with consistent error-handling semantics."""
    shutil.rmtree(path, ignore_errors=ignore_errors)


def orchestrator_metadata_dir(workspace_path: Path) -> Path:
    """Return the ``.orchestrator_workspace`` path inside *workspace_path*."""
    return workspace_path / ORCHESTRATOR_WORKSPACE_DIRNAME
