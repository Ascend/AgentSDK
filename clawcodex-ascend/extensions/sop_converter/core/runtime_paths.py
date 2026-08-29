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

"""Runtime path normalization for SOP-converted bundles."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_WSL_MOUNT_RE = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")


def is_wsl_runtime() -> bool:
    """Return True when the current Python is running inside WSL."""

    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = os.uname().release.lower()
    except AttributeError:
        return False
    return "microsoft" in release or "wsl" in release


def wsl_path_to_windows_path(path: str) -> str | None:
    """Convert ``/mnt/c/...`` to ``C:\\...`` when *path* has WSL shape."""

    match = _WSL_MOUNT_RE.match(path)
    if not match:
        return None
    drive = match.group(1).upper()
    rest = (match.group(2) or "").replace("/", "\\")
    return f"{drive}:\\" + rest if rest else f"{drive}:\\"


def windows_path_to_wsl_path(path: str) -> str | None:
    """Convert ``C:\\...`` or ``C:/...`` to ``/mnt/c/...``."""

    match = _WINDOWS_DRIVE_RE.match(path)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = (match.group(2) or "").replace("\\", "/").strip("/")
    return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"


def normalize_runtime_path(value: str | os.PathLike[str]) -> Path:
    """Normalize a persisted bundle path for the current runtime platform."""

    path = os.path.expanduser(os.fspath(value))
    converted: str | None = None
    if os.name == "nt":
        converted = wsl_path_to_windows_path(path)
    elif is_wsl_runtime():
        converted = windows_path_to_wsl_path(path)
    if converted:
        path = converted
    return Path(path).resolve()
