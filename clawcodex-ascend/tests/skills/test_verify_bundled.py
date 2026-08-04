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

"""Regression coverage for the bundled ``verify`` skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

from clawcodex_ext.command_system.aggregator import clear_commands_cache
from clawcodex_ext.agent.conversation import Conversation
from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.skills.bundled_skills import (
    clear_bundled_skills,
    get_bundled_skill_by_name,
)
from clawcodex_ext.skills.loader import (
    clear_dynamic_skills,
    clear_skill_caches,
    clear_skill_registry,
)
from clawcodex_ext.tool_system.defaults import build_default_registry
from src.skills.bundled import reset_bundled_skills_init_flag
from src.tool_system.context import ToolContext
from clawcodex_ext.types.messages import UserMessage
from src.tool_system.tools.skill import run_user_invoked_skill


@pytest.fixture(autouse=True)
def _reset_skill_runtime() -> None:
    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()
    yield
    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()
    reset_bundled_skills_init_flag()


def test_verify_registration_matches_bundled_contract() -> None:
    skill = get_bundled_skill_by_name("verify")

    assert skill is not None
    assert skill.description == "Verify a code change does what it should by running the app."
    assert skill.user_invocable is True
    assert skill.loaded_from == "bundled"
    assert skill.skill_root is not None
    assert not Path(skill.skill_root).exists()
    assert skill.context == "fork"
    assert skill.agent == "verification"


def test_verify_invocation_extracts_examples_and_renders_evidence_contract(
    tmp_path: Path,
) -> None:
    skill = get_bundled_skill_by_name("verify")
    assert skill is not None
    prompt = skill.get_prompt("")
    assert skill.skill_root is not None
    root = Path(skill.skill_root)
    assert prompt.startswith(f"Base directory for this skill: {root}\n\n")
    assert "Independently verify the implementation" in prompt
    assert "at least one relevant adversarial probe" in prompt
    assert "expected-versus-actual comparisons" in prompt
    assert "VERDICT: PASS" in prompt
    assert (root / "examples" / "cli.md").read_text(encoding="utf-8").startswith("# CLI verification example")
    assert (root / "examples" / "server.md").read_text(encoding="utf-8").startswith("# Server verification example")


def test_verify_user_request_is_appended_verbatim(tmp_path: Path) -> None:
    request = "Verify issue #42, including malformed Unicode input."
    skill = get_bundled_skill_by_name("verify")
    assert skill is not None
    prompt = skill.get_prompt(request)
    assert "## User Request" in prompt
    assert request in prompt


def test_verify_prompt_keeps_skill_resources_distinct_from_project_workspace(
    tmp_path: Path,
) -> None:
    skill = get_bundled_skill_by_name("verify")
    assert skill is not None
    context = ToolContext(workspace_root=tmp_path, cwd=tmp_path)

    assert skill.get_prompt_for_command is not None
    prompt = skill.get_prompt_for_command("Run the focused tests.", context)

    assert "contains reference material only" in prompt
    assert "not the project under verification" in prompt
    assert f"Project workspace root: `{tmp_path}`" in prompt
    assert "run those commands first" in prompt
    assert "report the verdict" in prompt
    assert "immediately instead of starting optional inspection" in prompt
    assert "never end on a tool" in prompt


def test_verify_and_verify_content_remain_distinct(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    verify = get_bundled_skill_by_name("verify")
    verify_content = run_user_invoked_skill("verify-content", "", ctx)

    assert verify is not None
    assert verify_content.output["success"] is True
    assert verify.agent == "verification"
    assert "Verification Assignment" in verify.get_prompt("")
    assert "Verify Recent Edits Match Intent" in verify_content.output["prompt"]


class _VerificationProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="verification-test-model")
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return ChatResponse(
            content="runtime evidence\nVERDICT: PASS",
            model="verification-test-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
        )

    def chat_stream(self, *_args: Any, **_kwargs: Any) -> Generator[str, None, None]:
        if False:  # pragma: no cover - makes this a generator
            yield ""

    def get_available_models(self) -> list[str]:
        return ["verification-test-model"]


def test_verify_runs_real_foreground_agent_with_inherited_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _VerificationProvider()
    registry = build_default_registry(provider=provider)
    context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        tool_registry=registry,
        messages=[UserMessage(content="Original task: add strict Unicode validation.")],
    )
    context._active_provider = provider
    monkeypatch.setattr(
        "src.agent.transcript.get_agent_transcript_path",
        lambda *_args, **_kwargs: str(tmp_path / "verification-agent.jsonl"),
    )

    result = run_user_invoked_skill(
        "verify",
        "Confirm malformed Unicode is rejected.",
        context,
    )

    assert result.is_error is False
    assert result.output["success"] is True
    assert result.output["status"] == "fork"
    assert result.output["result"].endswith("VERDICT: PASS")
    assert len(provider.calls) == 1
    call = provider.calls[0]
    rendered_messages = repr(call["messages"])
    assert "Original task: add strict Unicode validation." in rendered_messages
    assert "Confirm malformed Unicode is rejected." in rendered_messages
    tool_names = {schema["name"] for schema in call["tools"] or []}
    assert {"Read", "Bash"}.issubset(tool_names)
    assert {"Agent", "Edit", "Write", "NotebookEdit"}.isdisjoint(tool_names)


@pytest.mark.asyncio
async def test_real_slash_command_returns_verification_agent_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clawcodex_ext.command_system.engine import CommandEngine
    from clawcodex_ext.command_system.registry import CommandRegistry
    from clawcodex_ext.command_system.skills_integration import load_and_register_skills
    from clawcodex_ext.command_system.types import CommandContext

    ctx = ToolContext(workspace_root=tmp_path, cwd=tmp_path)
    from clawcodex_ext.tool_system.protocol import ToolResult

    def fake_run_user_invoked_skill(name: str, args: str, _context: object) -> ToolResult:
        assert name == "verify"
        assert args == "confirm malformed input returns a non-zero exit code"
        return ToolResult(
            name="Skill",
            output={
                "success": True,
                "status": "fork",
                "result": "runtime evidence\nVERDICT: PASS",
            },
        )

    monkeypatch.setattr("clawcodex_ext.tool_system.tools.skill.run_user_invoked_skill", fake_run_user_invoked_skill)
    registry = CommandRegistry()
    load_and_register_skills(project_root=tmp_path, registry=registry)
    command_context = CommandContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        conversation=Conversation(),
        tool_context=ctx,
    )

    result = await CommandEngine(
        registry=registry,
        workspace_root=tmp_path,
        context=command_context,
    ).execute("/verify confirm malformed input returns a non-zero exit code")

    assert result.success is True
    assert result.result_type == "text"
    assert result.display == "assistant"
    assert result.should_query is False
    assert result.prompt_is_meta is True
    assert "runtime evidence" in result.text
    assert result.text.endswith("VERDICT: PASS")
    messages = command_context.conversation.messages
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert "VERDICT: PASS" in repr(messages[0].content)
