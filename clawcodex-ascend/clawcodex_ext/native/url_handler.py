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

# pylint: disable=logging-too-few-args

"""Operating-system URL scheme registration and browser launching."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["UrlHandlerModule"]

_logger = logging.getLogger("clawcodex_ext.native.url_handler")


@NativeModuleRegistry.register("url_handler")
class UrlHandlerModule:
    """Register URL schemes and open external URLs."""

    name = "url_handler"

    # -- NativeModule protocol --------------------------------------------

    def is_available(self) -> bool:
        # ``webbrowser`` is always available; registration needs platform tools.
        return True

    def get_version(self) -> str:
        return f"python-webbrowser/{sys.platform}"

    # -- URL scheme registration -----------------------------------------

    def register_protocol(
        self,
        protocol: str = "clawcodex",
        executable: str = "clawcodex",
    ) -> bool:
        """Register a URL scheme for the current operating system."""
        if sys.platform.startswith("linux"):
            return self._register_linux(protocol, executable)
        if sys.platform == "darwin":
            return self._register_macos(protocol, executable)
        if sys.platform.startswith("win"):
            return self._register_windows(protocol, executable)
        _logger.warning("url_handler: unsupported platform %r", sys.platform)
        return False

    def _register_linux(self, protocol: str, executable: str) -> bool:
        apps_dir = Path.home() / ".local/share/applications"
        try:
            apps_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _logger.warning("url_handler: cannot create %s: %s", apps_dir, exc)
            return False
        desktop_file = apps_dir / f"{protocol}-handler.desktop"
        content = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=ClawCodex\n"
            f"Exec={executable} %u\n"
            f"MimeType=x-scheme-handler/{protocol};\n"
            "NoDisplay=true\n"
        )
        try:
            desktop_file.write_text(content, encoding="utf-8")
            os.chmod(desktop_file, 0o755)
        except OSError as exc:
            _logger.warning("url_handler: cannot write %s: %s", desktop_file, exc)
            return False
        xdg_mime = shutil.which("xdg-mime")
        if not xdg_mime:
            _logger.warning("url_handler: xdg-mime not found; desktop file written only")
            return True  # The file remains available for manual association.
        try:
            subprocess.run(
                [
                    xdg_mime,
                    "default",
                    f"{protocol}-handler.desktop",
                    f"x-scheme-handler/{protocol}",
                ],
                check=False,
                capture_output=True,
            )
            return True
        except (subprocess.SubprocessError, OSError) as exc:
            _logger.warning("url_handler: xg-mime default failed: %s", exc)
            return False

    def _register_macos(self, protocol: str, executable: str) -> bool:
        # macOS requires an application bundle and lsregister. A CLI-only
        # process cannot register reliably, so callers receive ``False``.
        _logger.info(
            "url_handler: macOS protocol registration requires .app bundle; "
            "use `open %s://...` after manual registration"
        )
        return False

    def _register_windows(self, protocol: str, executable: str) -> bool:
        # Write HKCU with reg.exe without requiring administrator access.
        reg = shutil.which("reg")
        if not reg:
            _logger.warning("url_handler: reg.exe not found")
            return False
        key = f"HKCU\\Software\\Classes\\{protocol}"
        cmd_key = f"{key}\\shell\\open\\command"
        try:
            subprocess.run(
                [reg, "add", key, "/ve", "/d", "URL:ClawCodex Protocol", "/f"],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                [reg, "add", key, "/v", "URL Protocol", "/d", "", "/f"],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                [reg, "add", cmd_key, "/ve", "/d", f'"{executable}" "%1"', "/f"],
                check=False,
                capture_output=True,
            )
            return True
        except (subprocess.SubprocessError, OSError) as exc:
            _logger.warning("url_handler: reg add failed: %s", exc)
            return False

    # -- Opening URLs -----------------------------------------------------

    def open_url(self, url: str) -> bool:
        """Open a URL in the default browser."""
        try:
            return bool(webbrowser.open(url))
        except webbrowser.Error as exc:
            _logger.warning("url_handler: webbrowser.open failed: %s", exc)
            return False

    def open_clawcodex(self, path: str) -> bool:
        """Open a clawcodex URL for the supplied path."""
        path = path.lstrip("/")
        return self.open_url(f"clawcodex://{path}")

    # -- fallback --------------------------------------------------
    # This standard-library implementation needs no fallback.
