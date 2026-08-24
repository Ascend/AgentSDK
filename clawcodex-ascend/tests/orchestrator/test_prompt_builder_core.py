# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Focused tests for the PromptBuilder core migrated in Part A.2."""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace


def _install_prerequisite_stubs() -> None:
    context = ModuleType("extensions.orchestrator.prompt_context")
    context.USER_MESSAGE_MARKER = "<!-- === USER MESSAGE === -->"  # type: ignore[attr-defined]
    context._build_sequential_workspace_context = lambda session: "sequential"  # type: ignore[attr-defined]
    context._expand_agent_mentions_in_prompt = lambda system, user, **kwargs: (  # type: ignore[attr-defined]
        system,
        user,
    )
    context._get_git_log_summary = lambda session: ""  # type: ignore[attr-defined]
    context._get_operator_hints = lambda path: None  # type: ignore[attr-defined]
    context._get_workspace_diff = lambda path: None  # type: ignore[attr-defined]
    context._resolve_workspace_path = lambda session: None  # type: ignore[attr-defined]
    context._to_jinja_value = lambda value: value  # type: ignore[attr-defined]
    context.resolve_python_executable = lambda **kwargs: ""  # type: ignore[attr-defined]
    sys.modules.setdefault(context.__name__, context)

    store = ModuleType("extensions.orchestrator.workflow_store")
    store.get_workflow_store = lambda: SimpleNamespace(current=lambda: None)  # type: ignore[attr-defined]
    sys.modules.setdefault(store.__name__, store)


_install_prerequisite_stubs()

from extensions.orchestrator.prompt_builder import PromptBuilder  # noqa: E402


class _Issue:
    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": "F-2",
            "title": "Repair runtime",
            "description": "Keep the session alive",
            "priority": "high",
            "state": "open",
        }


def test_render_uses_default_template_and_issue_fields() -> None:
    prompt = PromptBuilder.render(_Issue())

    assert "F-2 - Repair runtime" in prompt
    assert "Keep the session alive" in prompt
    assert "clawcodex-dev" in prompt


def test_render_parts_splits_system_and_user_message(monkeypatch) -> None:
    template = "Stable system rules\n<!-- === USER MESSAGE === -->\nIssue {{ issue.identifier }}: {{ issue.title }}"
    monkeypatch.setattr(
        "extensions.orchestrator.prompt_builder.get_workflow_store",
        lambda: SimpleNamespace(current=lambda: (SimpleNamespace(), template)),
    )

    system, user = PromptBuilder.render_parts(_Issue())

    assert system == "Stable system rules"
    assert user == "Issue F-2: Repair runtime"


def test_build_clarification_context_formats_question_and_options() -> None:
    context = PromptBuilder.build_clarification_context(
        pending_question="Which API should remain compatible?",
        options=["v1", "v2"],
    )

    assert "Which API should remain compatible?" in context
    assert "v1, v2" in context


def test_build_clarification_context_returns_empty_without_input() -> None:
    assert PromptBuilder.build_clarification_context() == ""


def test_render_warns_when_premise_check_fails(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "extensions.orchestrator.prompt_builder._resolve_workspace_path",
        lambda _session: tmp_path,
    )

    def _raise_premise_error(*_args) -> list[str]:
        raise RuntimeError("premise backend unavailable")

    monkeypatch.setattr(
        "extensions.orchestrator.prompt_builder.check_issue_premise",
        _raise_premise_error,
    )

    with caplog.at_level(logging.WARNING, logger="extensions.orchestrator.prompt_builder"):
        prompt = PromptBuilder.render(_Issue(), session=SimpleNamespace())

    assert "Repair runtime" in prompt
    assert "premise check failed" in caplog.text
