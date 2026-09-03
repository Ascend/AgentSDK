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

"""Regression tests for side-effect-free installer help options."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("--help", "USAGE"),
        ("-h", "USAGE"),
        ("--help-zh", "用法"),
    ],
)
def test_help_option_is_available_before_install(option: str, expected: str) -> None:
    result = subprocess.run(
        ["bash", str(INSTALLER), option],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "Installer crash" not in result.stderr
