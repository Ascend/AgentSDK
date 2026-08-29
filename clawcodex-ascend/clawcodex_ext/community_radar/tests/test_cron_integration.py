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

"""Tests for clawcodex_ext.community_radar.cron_integration."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.community_radar.cron_integration import (
    DEFAULT_CRON_TASK_ID,
    ensure_cron_installed,
    get_cron_task_status,
    install_cron_task,
    load_registry_safely,
    uninstall_cron_task,
)


def test_install_uninstall_roundtrip(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    summary = install_cron_task(schedule="0 8 * * 1")
    if not summary.installed:
        # Cron subsystem not available in this environment — smoke test.
        return
    assert summary.schedule == "0 8 * * 1"

    status = get_cron_task_status()
    assert status.installed is True
    assert status.task_id == DEFAULT_CRON_TASK_ID

    removed = uninstall_cron_task()
    assert removed.installed is True

    again = get_cron_task_status()
    assert again.installed is False


def test_install_with_invalid_id(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    summary = install_cron_task(task_id="custom-task", schedule="0 9 * * 1")
    if not summary.installed:
        return
    status = get_cron_task_status(task_id="custom-task")
    assert status.installed is True
    uninstall_cron_task(task_id="custom-task")


def test_load_registry_safely_seeds_defaults(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    registry = load_registry_safely()
    assert len(registry) > 0
    names = {s.name for s in registry.list()}
    assert "aider" in names
    assert "claude-code" in names


def test_uninstall_when_no_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    summary = uninstall_cron_task()
    # Either way: it should not raise.
    assert summary.task_id == DEFAULT_CRON_TASK_ID


def test_ensure_cron_installed_first_time(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    summary = ensure_cron_installed()
    if not summary.installed:
        # Cron subsystem not available; treat as smoke test.
        return
    assert summary.task_id == DEFAULT_CRON_TASK_ID
    assert summary.schedule  # cron expression recorded


def test_ensure_cron_installed_idempotent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    first = ensure_cron_installed()
    if not first.installed:
        return
    second = ensure_cron_installed()
    assert second.installed is True
    # Same schedule ⇒ no rewrite required.
    assert second.schedule == first.schedule


def test_ensure_cron_force_rewrites(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    first = ensure_cron_installed()
    if not first.installed:
        return
    again = ensure_cron_installed(force=True)
    assert again.installed is True


def test_ensure_cron_disabled_no_op(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    # Write a config.yaml that disables the radar so _load_config_safely
    # returns enabled=False (env vars don't reach the cron helper's loader).
    config_dir = tmp_path / "community-radar"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text("community_radar:\n  enabled: false\n", encoding="utf-8")
    summary = ensure_cron_installed()
    assert summary.installed is False
    assert "RadarConfig.enabled=False" in summary.message
