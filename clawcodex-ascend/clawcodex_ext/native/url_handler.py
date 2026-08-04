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

# pylint: disable=logging-too-few-args

"""F-81.4: OS URL Scheme 注册与浏览器唤起模块.

对标 CCB ``url-handler-napi``，用 Python 标准库 ``webbrowser`` + 平台
特定机制注册 ``clawcodex://`` URL Scheme:

* **Linux** — 写 ``~/.local/share/applications/<protocol>-handler.desktop``
  并调用 ``xdg-mime default`` 关联 MimeType.
* **macOS** — 写 ``~/Library/LaunchAgents`` 旁的 ``.plist`` 风格说明
  （简化版：仅提示用 ``Swift`` / ``lsregister``，本模块返回 ``False``
  且不抛异常，调用方可继续用 ``open_url``）.
* **Windows** — 通过 ``reg`` 命令写 ``HKCU\\Software\\Classes\\<protocol>``.

``webbrowser.open`` 在所有平台可用，故 ``is_available`` 恒 ``True``.
"""

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
    """注册 ``clawcodex://`` URL Scheme 并打开外部 URL."""

    name = "url_handler"

    # -- NativeModule protocol --------------------------------------------

    def is_available(self) -> bool:
        # webbrowser 是标准库，永远可用；register_protocol 取决于平台工具.
        return True

    def get_version(self) -> str:
        return f"python-webbrowser/{sys.platform}"

    # -- URL Scheme 注册 --------------------------------------------------

    def register_protocol(
        self,
        protocol: str = "clawcodex",
        executable: str = "clawcodex",
    ) -> bool:
        """注册 ``<protocol>://`` URL Scheme（按 OS 平台）.

        Args:
            protocol: 协议名，默认 ``"clawcodex"``.
            executable: 接收 URL 参数的可执行命令，默认 ``"clawcodex"``.

        Returns:
            ``True`` 注册成功；``False`` 平台不支持或注册失败（不抛异常）.
        """
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
            return True  # 文件已写，用户可手动关联
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
        # macOS 注册 URL Scheme 需要打包 .app bundle 并 lsregister，
        # 纯 CLI 场景无法可靠注册。返回 False 让调用方走 fallback。
        _logger.info(
            "url_handler: macOS protocol registration requires .app bundle; "
            "use `open %s://...` after manual registration"
        )
        return False

    def _register_windows(self, protocol: str, executable: str) -> bool:
        # 用 reg.exe 写 HKCU（无需管理员权限）
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

    # -- 打开 URL ---------------------------------------------------------

    def open_url(self, url: str) -> bool:
        """用默认浏览器打开 URL，返回 ``True`` 表示成功唤起."""
        try:
            return bool(webbrowser.open(url))
        except webbrowser.Error as exc:
            _logger.warning("url_handler: webbrowser.open failed: %s", exc)
            return False

    def open_clawcodex(self, path: str) -> bool:
        """便捷包装：打开 ``clawcodex://<path>``."""
        path = path.lstrip("/")
        return self.open_url(f"clawcodex://{path}")

    # -- F-81.6 fallback --------------------------------------------------
    # 该模块基于标准库，永远可用，无需 fallback；load_or_fallback 直接返回本类实例.
