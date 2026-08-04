#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Platform-specific Computer Use backends.

The real Linux backend is the only shipped implementation. macOS and Windows
stubs are intentionally absent; ``build_provider_suite`` falls back to the
null suite on unsupported platforms.
"""

from __future__ import annotations

from ..exceptions import ComputerUseError
from .linux import (
    ALLOW_ENV_VAR,
    LinuxBackend,
    LinuxClipboardManager,
    LinuxInputSimulator,
    LinuxScreenshotProvider,
    LinuxWindowManager,
    build_linux_suite,
    default_linux_backend,
)
from .null import (
    NullClipboardManager,
    NullInputSimulator,
    NullScreenshotProvider,
    NullWindowManager,
    build_null_suite,
)

__all__ = [
    "ALLOW_ENV_VAR",
    "LinuxBackend",
    "LinuxClipboardManager",
    "LinuxInputSimulator",
    "LinuxScreenshotProvider",
    "LinuxWindowManager",
    "NullClipboardManager",
    "NullInputSimulator",
    "NullScreenshotProvider",
    "NullWindowManager",
    "build_linux_suite",
    "build_null_suite",
    "default_linux_backend",
]


_UNSUPPORTED_PLATFORMS: frozenset[str] = frozenset()


def _current_platform() -> str:
    import sys

    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in {"win32", "cygwin"}:
        return "windows"
    return sys.platform or "unknown"


def build_provider_suite(
    platform: str | None = None,
    *,
    backend: LinuxBackend | None = None,
    recorder=None,
) -> dict[str, object]:
    """Return a dict of provider instances for the requested platform.

    The default platform is the runtime platform. Unsupported platforms return
    the null suite so callers can still wire Computer Use tools without a
    hard crash.
    """
    name = (platform or _current_platform()).lower()
    if name == "linux":
        return build_linux_suite(backend=backend, recorder=recorder)
    if name in _UNSUPPORTED_PLATFORMS:
        raise ComputerUseError(f"platform {name!r} is explicitly disabled")
    return build_null_suite()
