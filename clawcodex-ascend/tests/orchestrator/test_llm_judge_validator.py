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

"""Unit tests for the LLM-as-Judge contract validator.

Covers all 7 built-in validators, custom validator registration, and context injection (workspace_dir / llm_client).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.orchestrator.workflow_engine.validators import (
    ContractValidator,
)


# ── ValidationResult ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_judge_pass_with_mock_client(tmp_path: Path) -> None:
    file = tmp_path / "report.md"
    file.write_text(
        textwrap.dedent("""\
        # Summary
        This is a complete report.
        ```python
        print("ok")
        ```
    """)
    )

    client = AsyncMock()
    client.complete.return_value = '{"score": 0.85, "reasoning": "good"}'

    validator = ContractValidator(llm_client=client)
    result = await validator.validate({"type": "llm_judge", "path": str(file), "threshold": 0.7})

    assert result.passed is True
    assert result.score == pytest.approx(0.85)
    assert client.complete.called


@pytest.mark.asyncio
async def test_llm_judge_fail_below_threshold(tmp_path: Path) -> None:
    file = tmp_path / "report.md"
    file.write_text("short")

    chat = AsyncMock(return_value='{"score": 0.3, "reasoning": "too short"}')
    client = SimpleNamespace(chat=chat)

    validator = ContractValidator(llm_client=client)
    result = await validator.validate({"type": "llm_judge", "path": str(file), "threshold": 0.7})

    assert result.passed is False
    assert result.score == pytest.approx(0.3)
    chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_judge_fallback_without_client(tmp_path: Path) -> None:
    file = tmp_path / "report.md"
    file.write_text("# Report\n\nSome content here.\n")

    validator = ContractValidator()
    result = await validator.validate({"type": "llm_judge", "path": str(file)})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.passed == (result.score >= 0.7)


@pytest.mark.asyncio
async def test_llm_judge_missing_path() -> None:
    validator = ContractValidator()
    result = await validator.validate({"type": "llm_judge"})

    assert result.passed is False
    assert "no path specified" in result.message
