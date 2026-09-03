#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Portable path handling for POSIX Unix-domain sockets."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

# macOS exposes a 104-byte ``sockaddr_un.sun_path`` including the trailing
# NUL; Linux allows 108 bytes. Keep a small portability margin so the same
# configured path works on both platforms.
MAX_UNIX_SOCKET_PATH_BYTES = 100
_PRIVATE_DIRECTORY_MODE = 0o700


def normalize_unix_socket_path(socket_path: str | Path) -> Path:
    """Return a deterministic UDS path that fits portable POSIX limits.

    Valid paths are preserved exactly. Overlong paths are represented by a
    hash inside a private, per-user directory under ``/tmp`` so independently
    started servers and clients resolve to the same socket.
    """

    candidate = Path(socket_path).expanduser()
    if os.name == "nt" or len(os.fsencode(candidate)) <= MAX_UNIX_SOCKET_PATH_BYTES:
        return candidate

    path_key = os.fsencode(candidate.resolve(strict=False))
    digest = hashlib.sha256(path_key).hexdigest()[:24]
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    # macOS requires the deliberately short system path; the directory is
    # created owner-only below before a socket can be bound inside it.
    socket_dir = Path("/tmp") / f"clawcodex-ipc-{uid}"  # nosec B108
    _ensure_private_socket_directory(socket_dir)
    return socket_dir / f"{digest}.sock"


def _ensure_private_socket_directory(path: Path) -> None:
    """Create or validate the predictable per-user directory in ``/tmp``."""

    try:
        path.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
    except FileExistsError:
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"IPC runtime path is not a directory: {path}")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError(f"IPC runtime directory has an unexpected owner: {path}")
    path.chmod(_PRIVATE_DIRECTORY_MODE)


__all__ = ["MAX_UNIX_SOCKET_PATH_BYTES", "normalize_unix_socket_path"]
