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

"""Unit tests for F-114 ContractValidator.

Covers all 7 built-in validators, custom validator registration, and context injection (workspace_dir / llm_client).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from extensions.orchestrator.workflow_engine.validators import (
    ContractValidator,
    ValidationResult,
)


# ── ValidationResult ──────────────────────────────────────────────────


def test_validation_result_defaults() -> None:
    r = ValidationResult(passed=True, validator_type="file_exists")
    assert r.passed is True
    assert r.validator_type == "file_exists"
    assert r.message == ""
    assert r.score is None
    assert r.detail == {}
    assert r.details == {}


# ── file_exists ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_exists_pass(tmp_path: Path) -> None:
    file = tmp_path / "report.md"
    file.write_text("hello")

    validator = ContractValidator()
    result = await validator.validate({"type": "file_exists", "path": str(file)})

    assert result.passed is True
    assert "exists" in result.message


@pytest.mark.asyncio
async def test_file_exists_fail() -> None:
    validator = ContractValidator()
    result = await validator.validate({"type": "file_exists", "path": "/nonexistent/file.txt"})

    assert result.passed is False
    assert "not found" in result.message


# ── file_size ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_size_within_range(tmp_path: Path) -> None:
    file = tmp_path / "data.bin"
    file.write_bytes(b"12345")

    validator = ContractValidator()
    result = await validator.validate({"type": "file_size", "path": str(file), "min_bytes": 3, "max_bytes": 10})

    assert result.passed is True
    assert result.details["size"] == 5


@pytest.mark.asyncio
async def test_file_size_too_small(tmp_path: Path) -> None:
    file = tmp_path / "tiny.txt"
    file.write_text("hi")

    validator = ContractValidator()
    result = await validator.validate({"type": "file_size", "path": str(file), "min_bytes": 100})

    assert result.passed is False
    assert result.details["size"] == 2


@pytest.mark.asyncio
async def test_file_size_missing_file() -> None:
    validator = ContractValidator()
    result = await validator.validate({"type": "file_size", "path": "/no/such/file", "min_bytes": 1})

    assert result.passed is False
    assert "not found" in result.message


# ── regex ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regex_pass(tmp_path: Path) -> None:
    file = tmp_path / "out.txt"
    file.write_text("foo bar baz foo")

    validator = ContractValidator()
    result = await validator.validate({"type": "regex", "path": str(file), "pattern": r"foo", "min_matches": 2})

    assert result.passed is True
    assert result.details["match_count"] == 2


@pytest.mark.asyncio
async def test_regex_not_enough_matches(tmp_path: Path) -> None:
    file = tmp_path / "out.txt"
    file.write_text("foo bar")

    validator = ContractValidator()
    result = await validator.validate({"type": "regex", "path": str(file), "pattern": r"foo", "min_matches": 2})

    assert result.passed is False
    assert result.details["match_count"] == 1


@pytest.mark.asyncio
async def test_regex_invalid_pattern(tmp_path: Path) -> None:
    file = tmp_path / "out.txt"
    file.write_text("content")

    validator = ContractValidator()
    result = await validator.validate({"type": "regex", "path": str(file), "pattern": r"[invalid"})

    assert result.passed is False
    assert "Invalid pattern" in result.message


# ── line_count ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_line_count_pass(tmp_path: Path) -> None:
    file = tmp_path / "lines.txt"
    file.write_text("a\nb\nc\n")

    validator = ContractValidator()
    result = await validator.validate({"type": "line_count", "path": str(file), "min_lines": 2, "max_lines": 10})

    assert result.passed is True
    assert result.details["line_count"] == 3


@pytest.mark.asyncio
async def test_line_count_too_many(tmp_path: Path) -> None:
    file = tmp_path / "lines.txt"
    file.write_text("a\nb\nc\n")

    validator = ContractValidator()
    result = await validator.validate({"type": "line_count", "path": str(file), "max_lines": 2})

    assert result.passed is False
    assert result.details["line_count"] == 3


# ── json_schema ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_json_schema_valid(tmp_path: Path) -> None:
    file = tmp_path / "config.json"
    file.write_text('{"version": "1.0", "enabled": true}')

    validator = ContractValidator()
    result = await validator.validate(
        {
            "type": "json_schema",
            "path": str(file),
            "schema": {
                "type": "object",
                "required": ["version"],
                "properties": {"version": {"type": "string"}},
            },
        }
    )

    assert result.passed is True
    assert "valid" in result.message


@pytest.mark.asyncio
async def test_json_schema_violation(tmp_path: Path) -> None:
    file = tmp_path / "config.json"
    file.write_text('{"version": 42}')

    validator = ContractValidator()
    result = await validator.validate(
        {
            "type": "json_schema",
            "path": str(file),
            "schema": {
                "type": "object",
                "properties": {"version": {"type": "string"}},
            },
        }
    )

    assert result.passed is False
    assert "Schema violation" in result.message


@pytest.mark.asyncio
async def test_json_schema_missing_dependency(tmp_path: Path) -> None:
    file = tmp_path / "config.json"
    file.write_text('{"version": "1.0"}')

    validator = ContractValidator()
    with patch.dict(sys.modules, {"jsonschema": None}):
        result = await validator.validate(
            {
                "type": "json_schema",
                "path": str(file),
                "schema": {"type": "object"},
            }
        )

    assert result.passed is False
    assert "jsonschema library not installed" in result.message


@pytest.mark.asyncio
async def test_custom_command_success() -> None:
    validator = ContractValidator()
    result = await validator.validate(
        {
            "type": "custom",
            "command": f'{sys.executable} -c print(1)',
            "shell": False,
        }
    )

    assert result.passed is True
    assert result.detail["exit_code"] == 0


@pytest.mark.asyncio
async def test_custom_command_failure() -> None:
    validator = ContractValidator()
    result = await validator.validate(
        {
            "type": "custom",
            "command": f'{sys.executable} -c print(1/0)',
            "shell": False,
        }
    )

    assert result.passed is False
    assert result.detail["exit_code"] == 1


@pytest.mark.asyncio
async def test_custom_command_uses_workspace_dir(tmp_path: Path) -> None:
    subdir = tmp_path / "workspace"
    subdir.mkdir()
    marker = subdir / "marker.txt"
    marker.write_text("ok")

    script = subdir / "check.py"
    script.write_text("from pathlib import Path\nassert Path('marker.txt').exists()\nprint('ok')\n")

    validator = ContractValidator(workspace_dir=str(subdir))
    result = await validator.validate(
        {
            "type": "custom",
            "command": f"{sys.executable} check.py",
            "cwd": ".",
            "shell": False,
        }
    )

    assert result.passed is True
    assert result.detail["cwd"] == str(subdir)


@pytest.mark.asyncio
async def test_custom_command_with_env(tmp_path: Path) -> None:
    subdir = tmp_path / "workspace"
    subdir.mkdir()
    script = subdir / "env_check.py"
    script.write_text("import os\nprint(os.environ['MY_VAR'])\n")

    validator = ContractValidator()
    result = await validator.validate(
        {
            "type": "custom",
            "command": f"{sys.executable} env_check.py",
            "cwd": str(subdir),
            "env": {"MY_VAR": "42"},
            "shell": False,
        }
    )

    assert result.passed is True
    assert "42" in (result.detail.get("stdout_tail") or "")


# ── registration / extensibility ──────────────────────────────────────


@pytest.mark.asyncio
async def test_register_sync_validator() -> None:
    def always_pass(**kwargs: Any) -> ValidationResult:
        return ValidationResult(passed=True, validator_type="demo", message="demo ok")

    validator = ContractValidator()
    validator.register("demo", always_pass)
    result = await validator.validate({"type": "demo"})

    assert result.passed is True
    assert result.message == "demo ok"


@pytest.mark.asyncio
async def test_register_async_validator() -> None:
    async def async_pass(**kwargs: Any) -> ValidationResult:
        return ValidationResult(passed=True, validator_type="async_demo", message="async ok")

    validator = ContractValidator()
    validator.register("async_demo", async_pass, is_async=True)
    result = await validator.validate({"type": "async_demo"})

    assert result.passed is True
    assert result.message == "async ok"


@pytest.mark.asyncio
async def test_unknown_validator_type() -> None:
    validator = ContractValidator()
    result = await validator.validate({"type": "not_real"})

    assert result.passed is False
    assert "Unknown validator type" in result.message


# ── validate_sync ─────────────────────────────────────────────────────


def test_validate_sync_no_event_loop(tmp_path: Path) -> None:
    file = tmp_path / "x.txt"
    file.write_text("x")

    validator = ContractValidator()
    result = validator.validate_sync({"type": "file_exists", "path": str(file)})

    assert result.passed is True


@pytest.mark.asyncio
async def test_validate_sync_rejects_running_loop() -> None:
    validator = ContractValidator()
    with pytest.raises(RuntimeError, match="running event loop"):
        validator.validate_sync({"type": "file_exists", "path": "/tmp/x"})


# ── integration: all validators in one pipeline ───────────────────────


@pytest.mark.asyncio
async def test_validate_all_mixed_results(tmp_path: Path) -> None:
    file = tmp_path / "doc.md"
    file.write_text("# Summary\n\nContent.\n")

    validator = ContractValidator()
    results = await validator.validate_all(
        [
            {"type": "file_exists", "path": str(file)},
            {"type": "regex", "path": str(file), "pattern": r"^# Summary"},
            {"type": "file_exists", "path": str(tmp_path / "missing.md")},
        ]
    )

    assert len(results) == 3
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is False
