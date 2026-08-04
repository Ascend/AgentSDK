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

"""F-81.4: URL Handler 模块单元测试."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from clawcodex_ext.native import load, load_or_fallback
from clawcodex_ext.native.url_handler import UrlHandlerModule


def test_url_handler_registered():
    from clawcodex_ext.native import NativeModuleRegistry

    assert NativeModuleRegistry.is_registered("url_handler")


def test_url_handler_always_available():
    mod = UrlHandlerModule()
    assert mod.is_available() is True
    assert "python-webbrowser" in mod.get_version()


def test_url_handler_load_returns_instance():
    inst = load("url_handler")
    assert inst is not None
    assert isinstance(inst, UrlHandlerModule)


def test_url_handler_open_url_delegates_to_webbrowser():
    mod = UrlHandlerModule()
    with mock.patch("webbrowser.open", return_value=True) as m:
        assert mod.open_url("https://example.com") is True
        m.assert_called_once_with("https://example.com")


def test_url_handler_open_clawcodex_prefix():
    mod = UrlHandlerModule()
    with mock.patch("webbrowser.open", return_value=True) as m:
        assert mod.open_clawcodex("session/123") is True
        m.assert_called_once_with("clawcodex://session/123")


def test_url_handler_open_url_handles_error():
    import webbrowser

    mod = UrlHandlerModule()
    with mock.patch("webbrowser.open", side_effect=webbrowser.Error("no browser")):
        assert mod.open_url("https://example.com") is False


def test_url_handler_register_linux_writes_desktop_file(monkeypatch, tmp_path):
    """Linux 路径应写 .desktop 文件并调用 xdg-mime."""
    mod = UrlHandlerModule()
    monkeypatch.setattr(sys, "platform", "linux-x86_64")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/xdg-mime")
    with mock.patch("subprocess.run") as run_mock:
        result = mod.register_protocol("clawcodex", executable="clawcodex-dev")
    assert result is True
    desktop = tmp_path / ".local/share/applications/clawcodex-handler.desktop"
    assert desktop.exists()
    content = desktop.read_text(encoding="utf-8")
    assert "Exec=clawcodex-dev %u" in content
    assert "x-scheme-handler/clawcodex" in content
    assert run_mock.called


def test_url_handler_register_unsupported_platform(monkeypatch):
    mod = UrlHandlerModule()
    monkeypatch.setattr(sys, "platform", "freebsd13")
    assert mod.register_protocol("clawcodex") is False


def test_url_handler_load_or_fallback():
    """url_handler 永远可用，load_or_fallback 返回主实例."""
    inst = load_or_fallback("url_handler")
    assert isinstance(inst, UrlHandlerModule)
