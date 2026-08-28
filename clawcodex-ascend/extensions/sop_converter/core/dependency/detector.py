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

"""Detect tool lifecycle patterns and build a :class:`ToolDependencyGraph`.

Feed a ``list[SourceComponent]`` through:

1. :func:`pair_build_invoke` — locate ``build_*`` / ``create_*`` ↔
   ``run_*`` / ``invoke_*`` candidate pairs.
2. Apply known hidden-step templates (e.g. ``persist_agent_catalog``)
   when the build op's lifecycle is "create" and the invoke op
   consumes the returned id.
3. Group tools that share a stem (``agent`` / ``team_session``) into
   intent groups, with the build op as the primary entry.
4. Generate priority routes from a small keyword allowlist.

The detector never raises: malformed ops are skipped with a warning,
so a partially-parseable SDK still produces a usable graph.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from extensions.sop_converter.core.source_parser import SourceComponent, SourceOperation
from extensions.sop_converter.core.dependency.heuristics import _op_key, extract_shared_params, pair_build_invoke
from extensions.sop_converter.core.dependency.models import (
    HiddenStep,
    IntentGroup,
    PriorityRoute,
    ToolDependency,
    ToolDependencyGraph,
)

logger = logging.getLogger(__name__)


# Hidden step templates by lifecycle tag.  Keys are lifecycle names
# (e.g. ``create → invoke``); values are the steps the runtime
# performs between the two visible tool calls.
_HIDDEN_STEP_TEMPLATES: dict[str, list[HiddenStep]] = {
    "create → invoke": [
        HiddenStep(
            action="persist_agent_catalog",
            description=("保存 agent_id → DSL/config/model/provider 映射,跨进程可恢复"),
        ),
        HiddenStep(
            action="materialize_on_invoke",
            description="调用时从 catalog 恢复并 create_llm_agent(DSL/config)",
        ),
        HiddenStep(
            action="invoke_same_runtime",
            description="在同一工具调用运行时执行 invoke / run",
        ),
    ],
    "prepare → execute": [
        HiddenStep(
            action="load_spec",
            description="读取并解析外部 spec / config / yaml",
        ),
        HiddenStep(
            action="materialize_session",
            description="在 runtime 中实例化 session / executor",
        ),
    ],
}


# Default priority route keywords per intent group.  Both English and
# Chinese are supported so the agent's ToolSearch bias works
# regardless of which language the user prompt is written in.
_DEFAULT_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "agent_lifecycle": [
        "create agent",
        "build agent",
        "new agent",
        "创建 agent",
        "调用 agent",
        "invoke agent",
        "run agent",
        "call agent",
    ],
    "session_lifecycle": [
        "create team session",
        "start team session",
        "run team",
        "创建会话",
        "运行团队",
    ],
    "spec_lifecycle": [
        "load spec",
        "load config",
        "加载配置",
        "加载 spec",
    ],
}


# Map of build-stem → intent group name.  Stems come from stripping
# the verb prefix; e.g. ``build_agent`` → ``agent``.
_STEM_TO_INTENT_GROUP: dict[str, str] = {
    "agent": "agent_lifecycle",
    "team": "session_lifecycle",
    "team_session": "session_lifecycle",
    "session": "session_lifecycle",
    "spec": "spec_lifecycle",
    "config": "spec_lifecycle",
    "pipeline": "spec_lifecycle",
}


def _lifecycle_label(stem: str) -> str:
    """Map a build stem to a lifecycle label (used for hidden steps)."""
    if stem in {"agent"}:
        return "create → invoke"
    if stem in {"team", "team_session", "session"}:
        return "create → invoke"
    if stem in {"spec", "config", "pipeline"}:
        return "prepare → execute"
    return ""


def _hidden_steps_for(lifecycle: str) -> list[HiddenStep]:
    return list(_HIDDEN_STEP_TEMPLATES.get(lifecycle, []))


def _stem_from_op_name(op_name: str) -> str:
    """Return the noun stem of an op name, e.g. ``build_agent`` → ``agent``.

    Strips any lifecycle verb (create or invoke) so both ends of a
    pair resolve to the same stem.
    """
    name = (op_name or "").lower()
    for prefix in (
        "build_",
        "create_",
        "init_",
        "register_",
        "ensure_",
        "load_",
        "invoke_",
        "run_",
        "call_",
        "send_",
    ):
        if name.startswith(prefix):
            stem = name[len(prefix) :]
            return stem
    return name


def _safe_op_pairs(
    components: Iterable[SourceComponent],
) -> list[SourceOperation]:
    ops: list[SourceOperation] = []
    for comp in components:
        try:
            ops.extend(comp.operations or [])
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Skipping component %s: %s", getattr(comp, "name", "?"), exc)
    return ops


def _build_intent_groups(
    pairs: list[tuple[SourceOperation, SourceOperation, list[str]]],
    components: list[SourceComponent],
) -> list[IntentGroup]:
    """Cluster paired ops by stem into intent groups.

    The group is keyed by the stem, not the op name, so a
    ``build_agent`` + ``invoke_existing_agent`` + ``run_agent`` all
    fall under the ``agent_lifecycle`` group.

    Component membership is tracked via ``op_to_comp`` so the op keys
    use the component's display name (matches the tool spec naming).
    """
    op_to_comp: dict[int, str] = {}
    for comp in components:
        for op in comp.operations or []:
            op_to_comp[id(op)] = comp.name

    groups: dict[str, IntentGroup] = {}
    for build_op, invoke_op, _ in pairs:
        for op in (build_op, invoke_op):
            stem = _stem_from_op_name(op.name)
            if stem not in _STEM_TO_INTENT_GROUP:
                continue
            intent_name = _STEM_TO_INTENT_GROUP[stem]
            comp_name = op_to_comp.get(id(op), "")
            key = _op_key(op, comp_name=comp_name)
            group = groups.get(intent_name)
            if group is None:
                group = IntentGroup(
                    name=intent_name,
                    description=_default_group_description(intent_name),
                    tools=[],
                    primary_entry=None,
                )
                groups[intent_name] = group
            if key not in group.tools:
                group.tools.append(key)
            # First build op encountered becomes primary entry
            if group.primary_entry is None and infer_is_build(op):
                group.primary_entry = key
    return list(groups.values())


def infer_is_build(op: SourceOperation) -> bool:
    name = (op.name or "").lower()
    return any(name.startswith(p) for p in ("build_", "create_", "init_", "register_", "ensure_", "load_"))


def _entry_op_name(primary_entry: str) -> str:
    """Extract the op-name part of a ``{comp}.{op_name}`` key.

    ``comp.build-agent`` → ``build-agent``; ``build-agent`` →
    ``build-agent``.  Used by :func:`_build_priority_routes` to
    decide if the primary entry is a create-style op.
    """
    if not primary_entry:
        return ""
    return primary_entry.rsplit(".", 1)[-1]


def _default_group_description(name: str) -> str:
    return {
        "agent_lifecycle": "Agent 完整生命周期(创建→持久化→恢复→调用)",
        "session_lifecycle": "团队 / 会话生命周期",
        "spec_lifecycle": "Spec / config 加载与执行链",
    }.get(name, name)


def _build_priority_routes(groups: list[IntentGroup]) -> list[PriorityRoute]:
    routes: list[PriorityRoute] = []
    for group in groups:
        kws = _DEFAULT_PRIORITY_KEYWORDS.get(group.name, [])
        if not kws:
            continue
        # Split into create-words vs invoke-words by inspecting the
        # group's primary entry.  The primary entry uses the
        # ``{comp}.{op}`` dotted format, so we extract the op-name
        # part before applying the build-prefix check.
        entry_op = _entry_op_name(group.primary_entry or "")
        is_create_entry = any(
            entry_op.startswith(p) for p in ("build-", "create-", "init-", "register-", "ensure-", "load-")
        )
        if is_create_entry:
            create_kws = [k for k in kws if re.search(r"create|build|new|创建|加载", k, re.IGNORECASE)]
            invoke_kws = [k for k in kws if re.search(r"invoke|run|call|调用|运行", k, re.IGNORECASE)]
            if create_kws:
                routes.append(
                    PriorityRoute(
                        keywords=create_kws,
                        intent_group=group.name,
                        entry_first=True,
                    )
                )
            if invoke_kws:
                routes.append(
                    PriorityRoute(
                        keywords=invoke_kws,
                        intent_group=group.name,
                        entry_first=False,
                    )
                )
        else:
            routes.append(
                PriorityRoute(
                    keywords=kws,
                    intent_group=group.name,
                    entry_first=True,
                )
            )
    return routes


def detect_lifecycle_patterns(
    components: list[SourceComponent],
) -> ToolDependencyGraph:
    """Main entry point — build a :class:`ToolDependencyGraph` from components.

    The detector is *idempotent*: feeding the same components list
    twice produces structurally identical graphs.  The output is
    deterministic (no random ids, no timestamps) so the YAML writer
    can be idempotent too.
    """
    if not components:
        return ToolDependencyGraph()
    try:
        all_ops = _safe_op_pairs(components)
        pairs = pair_build_invoke(all_ops)

        op_to_comp: dict[int, str] = {}
        for comp in components:
            for op in comp.operations or []:
                op_to_comp[id(op)] = comp.name

        deps: list[ToolDependency] = []
        seen: set[tuple[str, str]] = set()
        for build_op, invoke_op, shared in pairs:
            from_key = _op_key(build_op, comp_name=op_to_comp.get(id(build_op), ""))
            to_key = _op_key(invoke_op, comp_name=op_to_comp.get(id(invoke_op), ""))
            if (from_key, to_key) in seen:
                continue
            seen.add((from_key, to_key))
            stem = _stem_from_op_name(build_op.name)
            lifecycle = _lifecycle_label(stem)
            deps.append(
                ToolDependency(
                    from_tool=from_key,
                    to_tool=to_key,
                    shared_params=shared or extract_shared_params(build_op, invoke_op),
                    hidden_steps=_hidden_steps_for(lifecycle),
                    lifecycle=lifecycle,
                )
            )

        groups = _build_intent_groups(pairs, components)
        routes = _build_priority_routes(groups)
        return ToolDependencyGraph(
            dependencies=deps,
            intent_groups=groups,
            priority_routes=routes,
        )
    except Exception as exc:  # pragma: no cover - defensive, see module docstring
        logger.warning(
            "Lifecycle detection failed for %d components: %s",
            len(components),
            exc,
        )
        return ToolDependencyGraph()


__all__ = ["detect_lifecycle_patterns"]
