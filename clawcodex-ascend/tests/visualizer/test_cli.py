#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""CLI contracts for the local Visualizer entry point."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

from extensions.visualizer.cli import register_viz_subcommand, run_viz


def test_default_host_is_loopback_and_arguments_reach_uvicorn(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}
    server = ModuleType("extensions.visualizer.server")

    def create_app(**kwargs):
        calls["create_app"] = kwargs
        return SimpleNamespace(name="app")

    server.create_app = create_app
    monkeypatch.setitem(sys.modules, "extensions.visualizer.server", server)

    uvicorn = ModuleType("uvicorn")

    def run(app, **kwargs) -> None:
        calls["uvicorn_app"] = app
        calls["uvicorn"] = kwargs

    uvicorn.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    result = run_viz(["--no-open", "--port", "9876"])

    assert result == 0
    assert calls["create_app"]["host"] == "127.0.0.1"
    assert calls["uvicorn"]["host"] == "127.0.0.1"
    assert calls["uvicorn"]["port"] == 9876
    assert "http://localhost:9876" in capsys.readouterr().out


def test_missing_uvicorn_returns_actionable_error(monkeypatch, capsys) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("missing uvicorn", name="uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert run_viz(["--no-open"]) == 1
    assert "approved AgentSDK runtime" in capsys.readouterr().err


def test_missing_web_dependency_returns_clean_error(monkeypatch, capsys) -> None:
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setitem(sys.modules, "extensions.visualizer.server", None)

    assert run_viz(["--no-open"]) == 1
    assert "web dependency is unavailable" in capsys.readouterr().err


def test_registration_uses_viz_name(monkeypatch) -> None:
    registered: dict[str, object] = {}
    registry = ModuleType("clawcodex_ext.cli.subcommand_registry")

    def register(name: str):
        def decorator(handler):
            registered[name] = handler
            return handler

        return decorator

    registry.register = register
    monkeypatch.setitem(sys.modules, "clawcodex_ext.cli.subcommand_registry", registry)

    register_viz_subcommand()

    assert set(registered) == {"viz"}
    assert callable(registered["viz"])
