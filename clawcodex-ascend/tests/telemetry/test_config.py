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

"""Tests for TelemetryConfig + load_config()."""

from __future__ import annotations

from pathlib import Path

import pytest

from telemetry.config import (
    ReportingConfig,
    TelemetryConfig,
    load_config,
)
from telemetry.redaction import RedactionConfig


def test_default_config_is_disabled():
    cfg = TelemetryConfig()
    assert cfg.enabled is True  # dev-default
    assert cfg.reporting.reporting_enabled is True  # dev-default
    assert cfg.reporting.kind == "local_file"
    assert cfg.retention_days == 30
    assert isinstance(cfg.redaction, RedactionConfig)
    assert cfg.redaction.include_prompts is False
    assert cfg.redaction.include_outputs is False


def test_load_config_defaults_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("CLAW_TELEMETRY_REPORTING_ENABLED", raising=False)
    monkeypatch.delenv("CLAW_TELEMETRY_STORAGE_DIR", raising=False)
    cfg = load_config()
    assert cfg.enabled is True  # dev-default
    assert cfg.reporting.reporting_enabled is True  # dev-default


def test_env_override_enables_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAW_TELEMETRY_ENABLED", "1")
    cfg = load_config()
    assert cfg.enabled is True


def test_env_override_enables_reporting(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAW_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("CLAW_TELEMETRY_REPORTING_ENABLED", "1")
    cfg = load_config()
    assert cfg.enabled is True
    assert cfg.reporting.reporting_enabled is True


def test_env_storage_dir_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "telemetry-store"
    monkeypatch.setenv("CLAW_TELEMETRY_STORAGE_DIR", str(target))
    cfg = load_config()
    assert cfg.storage_dir == target.expanduser()


def test_reporting_kind_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAW_TELEMETRY_ENABLED", "1")
    cfg = load_config()
    assert cfg.reporting.kind == "local_file"


def test_config_dataclass_is_frozen():
    cfg = TelemetryConfig()
    with pytest.raises(Exception):
        cfg.enabled = True  # type: ignore[misc]


def test_reporting_config_defaults():
    rc = ReportingConfig()
    assert rc.reporting_enabled is True  # dev-default
    assert rc.kind == "local_file"
    assert rc.platform == "gitcode"
    assert rc.mode == "create_daily"
    assert rc.interval_hours == 24


def test_load_config_parses_issue_reporting_section(monkeypatch, tmp_path):
    token = "ghp_12345678901234567890"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAW_TELEMETRY_REPORTING_TOKEN", token)
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("CLAW_TELEMETRY_REPORTING_ENABLED", raising=False)
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {
            "telemetry": {
                "enabled": True,
                "storage_dir": str(tmp_path / "store"),
                "reporting": {
                    "enabled": True,
                    "kind": "issue",
                    "platform": "gitee",
                    "owner": "acme",
                    "repo": "widget",
                    "endpoint": "https://example.test/api/",
                    "issue_title": "Telemetry Inbox",
                    "mode": "create_daily",
                    "interval_hours": "0",
                    "token_env": "CLAW_TELEMETRY_REPORTING_TOKEN",
                },
            }
        },
    )

    cfg = load_config()

    assert cfg.enabled is True
    assert cfg.reporting.reporting_enabled is True
    assert cfg.reporting.kind == "issue"
    assert cfg.reporting.platform == "gitee"
    assert cfg.reporting.owner == "acme"
    assert cfg.reporting.repo == "widget"
    assert cfg.reporting.endpoint == "https://example.test/api"
    assert cfg.reporting.issue_title == "Telemetry Inbox"
    assert cfg.reporting.mode == "create_daily"
    assert cfg.reporting.interval_hours == 1
    assert cfg.reporting.token_env == "CLAW_TELEMETRY_REPORTING_TOKEN"
    assert cfg.reporting.api_key == token


def test_reporting_enabled_alias_preserves_reporting_enabled_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_REPORTING_ENABLED", raising=False)
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {"telemetry": {"reporting": {"reporting_enabled": True}}},
    )

    cfg = load_config()

    assert cfg.reporting.reporting_enabled is True


def test_invalid_interval_hours_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "src.config.load_config",
        lambda: {"telemetry": {"reporting": {"interval_hours": "not-an-int"}}},
    )

    cfg = load_config()

    assert cfg.reporting.interval_hours == 24


def test_load_config_passes_cwd_when_supported(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_load_config(*, cwd=None):
        seen["cwd"] = cwd
        return {"telemetry": {"enabled": True}}

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("src.config.load_config", fake_load_config)

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True
    assert seen["cwd"] == tmp_path


def test_load_config_cwd_falls_back_for_legacy_loader(monkeypatch, tmp_path):
    calls = 0

    def fake_load_config():
        nonlocal calls
        calls += 1
        return {"telemetry": {"enabled": True}}

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("src.config.load_config", fake_load_config)

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True
    assert calls == 1


# ---------------------------------------------------------------------------
# K: TOML config loader
# ---------------------------------------------------------------------------


def _write_pyproject(cwd: Path, table: str) -> Path:
    """Write a minimal pyproject.toml with one telemetry table at cwd."""
    p = cwd / "pyproject.toml"
    p.write_text(
        "[project]\n"
        'name = "demo"\n'
        'version = "0.0.1"\n\n'
        f"[tool.clawcodex.{table}]\n"
        "enabled = true\n"
        'storage_dir = "/tmp/tel-toml-test"\n',
        encoding="utf-8",
    )
    return p


def _write_telemetry_toml(cwd: Path) -> Path:
    p = cwd / "telemetry.toml"
    p.write_text(
        "[telemetry]\nenabled = true\nretention_days = 7\n",
        encoding="utf-8",
    )
    return p


def test_load_config_reads_pyproject_toml_section(monkeypatch, tmp_path):
    """pyproject.toml [tool.clawcodex.telemetry] is discovered
    by walking upward from cwd.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setattr("src.config.load_config", lambda *a, **k: {})
    _write_pyproject(tmp_path, "telemetry")

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True
    assert str(cfg.storage_dir) == "/tmp/tel-toml-test"


def test_load_config_reads_telemetry_toml(monkeypatch, tmp_path):
    """standalone <cwd>/telemetry.toml is read directly."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setattr("src.config.load_config", lambda *a, **k: {})
    _write_telemetry_toml(tmp_path)

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True
    assert cfg.retention_days == 7


def test_load_config_project_toml_overrides_global_json(monkeypatch, tmp_path):
    """Project-scoped TOML wins over the legacy global JSON section."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    _write_telemetry_toml(tmp_path)  # TOML: enabled = true
    monkeypatch.setattr(
        "src.config.load_config",
        lambda *a, **k: {"telemetry": {"enabled": False}},
    )

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True


def test_load_config_toml_invalid_falls_back_to_defaults(monkeypatch, tmp_path):
    """malformed TOML must not break load_config — silently
    treat it as 'no TOML section' and use dataclass defaults.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setattr("src.config.load_config", lambda *a, **k: {})
    (tmp_path / "telemetry.toml").write_text(
        "this is not valid = toml = = =",
        encoding="utf-8",
    )

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True  # dev-default
    assert cfg.retention_days == 30


def test_load_config_pyproject_walks_upward_from_cwd(monkeypatch, tmp_path):
    """the TOML loader walks upward from cwd looking for
    pyproject.toml, so a nested invocation still finds the table.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setattr("src.config.load_config", lambda *a, **k: {})
    _write_pyproject(tmp_path, "telemetry")
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)

    cfg = load_config(cwd=nested)

    assert cfg.enabled is True
    assert str(cfg.storage_dir) == "/tmp/tel-toml-test"


def test_load_config_no_toml_files_unchanged_from_json_only(monkeypatch, tmp_path):
    """when no TOML files are present, behavior is identical to
    the JSON-only path.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAW_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setattr(
        "src.config.load_config",
        lambda *a, **k: {"telemetry": {"enabled": True, "retention_days": 14}},
    )

    cfg = load_config(cwd=tmp_path)

    assert cfg.enabled is True
    assert cfg.retention_days == 14
