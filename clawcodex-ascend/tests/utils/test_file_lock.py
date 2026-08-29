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

from __future__ import annotations

import os

from clawcodex_ext.utils import file_lock


def test_exclusive_file_lock_uses_windows_byte_lock(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd: int, mode: int, length: int) -> None:
            calls.append((mode, length, os.lseek(fd, 0, os.SEEK_CUR)))

    monkeypatch.setattr(file_lock, "_msvcrt", FakeMsvcrt)

    with file_lock.exclusive_file_lock(tmp_path / "catalog.lock") as fd:
        assert fd >= 0

    assert calls == [(FakeMsvcrt.LK_LOCK, 1, 0), (FakeMsvcrt.LK_UNLCK, 1, 0)]
