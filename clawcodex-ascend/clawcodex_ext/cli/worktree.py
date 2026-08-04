#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Session-worktree helpers shared by CLI dispatch and compatibility wrappers."""

from __future__ import annotations

import sys


# Distinguishes an absent ``--worktree`` option from a requested worktree that
# could not be created.  Keep this object module-global: callers and the
# ``src.cli`` compatibility facade intentionally compare it by identity.
WORKTREE_FAILED = object()


def maybe_create_worktree(args):
    """Create or resume the worktree requested by ``args.worktree``."""
    option = getattr(args, "worktree", None)
    if option is None:
        return None

    from src.utils.worktree_session import (
        WorktreeError,
        create_session_from_cli_option,
    )

    try:
        session = create_session_from_cli_option(option)
    except WorktreeError as exc:
        print(f"Error creating worktree: {exc}", file=sys.stderr)
        return WORKTREE_FAILED

    print(
        f"Using worktree {session.worktree_name} at {session.worktree_path} (branch {session.worktree_branch})",
        file=sys.stderr,
    )
    return session


def print_worktree_keep_note(session) -> None:
    """Tell headless/local-UI callers where the deliberately kept tree lives."""
    print(
        f"Worktree kept at {session.worktree_path} (branch {session.worktree_branch})",
        file=sys.stderr,
    )
