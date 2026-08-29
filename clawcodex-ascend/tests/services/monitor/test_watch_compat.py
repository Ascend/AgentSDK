#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for watch compatibility conversion."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import platform


from clawcodex_ext.services.monitor.watch_compat import normalize_watch_command


class TestNormalizeWatchCommand:
    def test_posix_unchanged(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert normalize_watch_command("watch -n 5 git status") == "watch -n 5 git status"
        assert normalize_watch_command("tail -f /var/log/syslog") == "tail -f /var/log/syslog"

    def test_macos_unchanged(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert normalize_watch_command("watch -n 5 git status") == "watch -n 5 git status"

    def test_windows_watch_converted(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        result = normalize_watch_command("watch -n 5 git status")
        assert result.startswith("powershell -c")
        assert "while(1){" in result
        assert "git status" in result
        assert "Start-Sleep 5" in result

    def test_windows_non_watch_unchanged(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert normalize_watch_command("tail -f /var/log/syslog") == "tail -f /var/log/syslog"

    def test_windows_quotes_safely_escaped(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        result = normalize_watch_command('watch -n 1 echo "hello $world"')
        # The inner command should be escaped so it does not break the loop.
        assert "powershell -c" in result
        assert "echo" in result
        assert "Start-Sleep 1" in result

    def test_windows_invalid_interval_unchanged(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert normalize_watch_command("watch -n abc git status") == "watch -n abc git status"

    def test_windows_negative_interval_unchanged(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert normalize_watch_command("watch -n -1 git status") == "watch -n -1 git status"
