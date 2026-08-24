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
#
"""from_issue — Issue-tracker context provider.

Registers a ``register_section`` builder that reads ``runtime_ctx["issue_info"]``
and renders a structured "Current Issue Context" block at ``order=55`` (after
skills, before output_style).

Usage
-----
Importing this module triggers registration at module-load time::

    from extensions.context_providers import from_issue  # noqa: F401

The builder returns ``None`` (no-op) when ``runtime_ctx`` does not contain
``issue_info``, so the provider is safe to import unconditionally.

Tags
----
``workflow``, ``issue-tracker``
"""

from __future__ import annotations

from clawcodex_ext.context_system.section_registry import (
    SectionScope,
    register_section,
)

__all__: list[str] = []


def _issue_context_builder(runtime_ctx: dict) -> str | None:
    """Build the issue-context section block.

    Returns ``None`` (skip section) when no issue info is available,
    or a markdown-formatted block otherwise.
    """
    issue = runtime_ctx.get("issue_info")
    if not issue:
        return None

    title = issue.get("title", "").strip()
    description = issue.get("description", "").strip()
    labels = issue.get("labels", [])

    lines = ["## Current Issue Context"]
    if title:
        lines.append(f"- Title: {title}")
    if description:
        # Truncate very long descriptions to avoid bloating the prompt.
        desc_short = (description[:500] + "…") if len(description) > 500 else description
        lines.append(f"- Description: {desc_short}")
    if labels:
        lines.append(f"- Labels: {', '.join(str(label) for label in labels)}")

    phase = runtime_ctx.get("workflow_phase")
    if phase:
        lines.append(f"- Phase: {phase}")

    return "\n".join(lines) if len(lines) > 1 else None


register_section(
    "issue-context",
    builder=_issue_context_builder,
    order=55,
    cache_scope=SectionScope.REQUEST,
    tags=["workflow", "issue-tracker"],
)
