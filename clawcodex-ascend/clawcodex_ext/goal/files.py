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
"""Materialize oversized goal objectives under Codex attachments."""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

# pylint: disable-next=no-name-in-module
from clawcodex_ext.goal.store import clawcodex_home

logger = logging.getLogger(__name__)

MAX_THREAD_GOAL_OBJECTIVE_CHARS = 4000
GOAL_ATTACHMENT_DIR = "attachments"
GOAL_FILE_NAME = "goal-objective.md"
GOAL_FILE_PREFIX = "Read the Codex goal objective file at "
GOAL_FILE_SUFFIX = " before continuing."


@dataclass(frozen=True)
class MaterializedGoalObjective:
    objective: str
    attachment_dir: Path | None = None


def materialize_goal_objective(
    objective: str,
    *,
    codex_home: Path | str | None = None,
) -> MaterializedGoalObjective:
    """Write long objectives to an attachment file and return a short reference."""
    if len(objective) <= MAX_THREAD_GOAL_OBJECTIVE_CHARS:
        return MaterializedGoalObjective(objective=objective)

    home = _codex_home(codex_home)
    attachment_dir = home / GOAL_ATTACHMENT_DIR / str(uuid.uuid4())
    goal_file = attachment_dir / GOAL_FILE_NAME
    try:
        attachment_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        attachment_dir.chmod(0o700)
        reference = objective_file_reference(goal_file)
        if len(reference) > MAX_THREAD_GOAL_OBJECTIVE_CHARS:
            raise ValueError(
                "Goal objective file reference is too long: "
                f"{len(reference)} characters. Limit: {MAX_THREAD_GOAL_OBJECTIVE_CHARS}."
            )
        goal_file.touch(mode=0o600, exist_ok=False)
        goal_file.write_text(objective, encoding="utf-8")
        goal_file.chmod(0o600)
    except (OSError, ValueError):
        shutil.rmtree(attachment_dir, ignore_errors=True)
        raise
    return MaterializedGoalObjective(objective=reference, attachment_dir=attachment_dir)


def objective_text_for_edit(
    objective: str,
    *,
    codex_home: Path | str | None = None,
) -> str:
    path = objective_file_path(objective, codex_home=codex_home)
    if path is None:
        return objective
    return path.read_text(encoding="utf-8")


def objective_file_reference(path: Path | str) -> str:
    return f"{GOAL_FILE_PREFIX}{Path(path)}{GOAL_FILE_SUFFIX}"


def objective_file_path(
    objective: str,
    *,
    codex_home: Path | str | None = None,
) -> Path | None:
    path_text = objective.removeprefix(GOAL_FILE_PREFIX)
    if path_text == objective:
        return None
    path_text = path_text.removesuffix(GOAL_FILE_SUFFIX)
    if objective_file_reference(path_text) != objective:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        return None

    home = _codex_home(codex_home)
    try:
        relative = path.relative_to(home / GOAL_ATTACHMENT_DIR)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 2 or parts[1] != GOAL_FILE_NAME:
        return None
    try:
        uuid.UUID(parts[0])
    except ValueError:
        return None
    attachments_root = home / GOAL_ATTACHMENT_DIR
    if attachments_root.is_symlink() or path.parent.is_symlink() or path.is_symlink():
        return None
    if not path.is_file():
        return None
    return path


def remove_materialized_goal_objective(
    objective: str,
    *,
    codex_home: Path | str | None = None,
) -> bool:
    """Remove an owned goal attachment without following symbolic links."""

    path = objective_file_path(objective, codex_home=codex_home)
    if path is None:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    try:
        path.parent.rmdir()
    except OSError:
        logger.warning("Removed goal objective file but could not remove attachment directory: %s", path.parent)
    return True


def _codex_home(codex_home: Path | str | None) -> Path:
    return Path(codex_home).expanduser() if codex_home is not None else clawcodex_home()


__all__ = [
    "GOAL_ATTACHMENT_DIR",
    "GOAL_FILE_NAME",
    "GOAL_FILE_PREFIX",
    "GOAL_FILE_SUFFIX",
    "MAX_THREAD_GOAL_OBJECTIVE_CHARS",
    "MaterializedGoalObjective",
    "materialize_goal_objective",
    "objective_file_path",
    "objective_file_reference",
    "objective_text_for_edit",
    "remove_materialized_goal_objective",
]
