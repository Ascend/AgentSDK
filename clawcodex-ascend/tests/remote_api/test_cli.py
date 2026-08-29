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

"""Tests for the Remote API CLI handler without editing the shared registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from clawcodex_ext.cli import subcommand_registry
from extensions.remote_api.cli import register_api_subcommand, run_api
from extensions.remote_api.core import RemoteAPIConfig


def test_api_subcommand_registration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_api_subcommand()
    first = subcommand_registry._SUBCOMMANDS["api"]  # pylint: disable=protected-access
    register_api_subcommand()

    monkeypatch.setattr(subcommand_registry, "load_builtin_subcommands", lambda: None)
    second = subcommand_registry.get_subcommand("api")

    assert first is not None
    assert second is not None
    assert second.__name__ == "_api_handler"


def test_api_serve_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        run_api(["serve", "--help"])

    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert "clawcodex api serve" in output
    assert "--host" in output
    assert "--port" in output
    assert "--workspace" in output
    assert "--state-limit" in output
    assert "--permission-mode" in output


def test_api_serve_uses_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[RemoteAPIConfig] = []
    monkeypatch.setattr(
        "extensions.remote_api.stdlib_server.serve",
        observed.append,
    )

    assert run_api(["serve", "--workspace", str(tmp_path)]) == 0

    assert observed[0].workspace == tmp_path.resolve()
