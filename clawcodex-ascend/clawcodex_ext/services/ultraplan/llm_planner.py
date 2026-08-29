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

"""LLM-backed generation of :class:`Plan` objects."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clawcodex_ext.providers.base import BaseProvider, ChatResponse  # pylint: disable=no-name-in-module

from .exceptions import PlannerFailedError, ProviderUnavailableError
from .models import CheckKind, Plan, PlanStatus, StepKind
from .templates import TemplateLibrary

_DANGEROUS_PATTERNS = (
    "rm -rf",
    "mkfs",
    "dd if=",
    "drop table",
    "kill -9",
)

_PLAN_KEYS = {
    "id",
    "title",
    "goal",
    "sub_plans",
    "status",
    "created_at",
    "updated_at",
    "metadata",
    "notes",
}
_SUB_PLAN_KEYS = {"id", "title", "description", "steps", "status", "notes"}
_STEP_KEYS = {
    "id",
    "title",
    "description",
    "kind",
    "criteria",
    "depends_on",
    "status",
    "started_at",
    "completed_at",
    "result",
    "error",
    "notes",
}
_CRITERIA_KEYS = {"id", "description", "kind", "target", "args", "required"}


@dataclass(frozen=True)
class PlannerContext:
    user_prompt: str
    cwd: str
    active_files: tuple[str, ...] = ()
    template: str | None = None
    model: str | None = None
    max_sub_plans: int = 5
    max_steps_per_sub_plan: int = 8
    existing_plan_id: str | None = None


@dataclass(frozen=True)
class PlannerResult:
    plan: Plan
    raw_response: str
    provider: str
    model: str
    latency_ms: int
    retry_count: int = 0


class LLMPlanner:
    def __init__(
        self,
        provider: BaseProvider | None,
        *,
        templates: TemplateLibrary | None = None,
        max_retries: int = 1,
    ) -> None:
        if provider is None:
            raise ProviderUnavailableError("ultraplan requires an active LLM provider")
        self.provider = provider
        self.templates = templates or TemplateLibrary()
        self.max_retries = max(0, max_retries)

    async def generate_plan(self, context: PlannerContext) -> PlannerResult:
        prompt = self._build_prompt(context)
        started = time.perf_counter()
        last_error: Exception | None = None
        raw = ""
        for attempt in range(self.max_retries + 1):
            messages = [
                {
                    "role": "system",
                    "content": self._system_prompt(context),
                },
                {"role": "user", "content": prompt},
            ]
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid. Return only strict "
                            "JSON matching the requested Plan shape."
                        ),
                    }
                )
            try:
                response: ChatResponse = await self.provider.chat_async(
                    messages,
                    model=context.model or getattr(self.provider, "model", None),
                    temperature=0,
                )
                raw = response.content
                plan = self.parse_plan(raw, context)
                latency = int((time.perf_counter() - started) * 1000)
                return PlannerResult(
                    plan=plan,
                    raw_response=raw,
                    provider=type(self.provider).__name__,
                    model=response.model or context.model or getattr(self.provider, "model", "") or "",
                    latency_ms=latency,
                    retry_count=attempt,
                )
            except Exception as exc:
                last_error = exc
        raise PlannerFailedError(f"could not generate a valid ultraplan: {last_error}") from last_error

    def parse_plan(self, raw_response: str, context: PlannerContext) -> Plan:
        data = _extract_json_object(raw_response)
        _validate_plan_shape(data, context)
        data.setdefault("id", _make_plan_id())
        data.setdefault("goal", context.user_prompt)
        data.setdefault("status", PlanStatus.DRAFT.value)
        metadata = dict(data.get("metadata") or {})
        metadata.update(
            {
                "planner": "llm",
                "cwd": str(Path(context.cwd)),
                "template": context.template,
            }
        )
        data["metadata"] = metadata
        plan = Plan.from_dict(data)
        _reject_dangerous_acceptance(plan)
        return plan

    def _build_prompt(self, context: PlannerContext) -> str:
        user_prompt = context.user_prompt.strip()
        if not user_prompt:
            raise PlannerFailedError("planning prompt must not be empty")
        if context.template:
            user_prompt = self.templates.apply(context.template, user_prompt)
        files = "\n".join(f"- {item}" for item in context.active_files) or "- none"
        return (
            f"Goal:\n{user_prompt}\n\n"
            f"Working directory: {context.cwd}\n"
            f"Active files:\n{files}\n"
            f"Limits: at most {context.max_sub_plans} sub_plans, "
            f"at most {context.max_steps_per_sub_plan} steps per sub_plan."
        )

    @staticmethod
    def _system_prompt(context: PlannerContext) -> str:
        return (
            "You create executable project plans as strict JSON. Return no markdown. "
            "Shape: {id?, title, goal, sub_plans:[{id,title,description,steps:"
            "[{id,title,description,kind,depends_on?,criteria:[{id,description,kind,target,args?,required?}]}]}]}. "
            "Allowed step kind values: research, implement, verify, review, other. "
            "Allowed check kind values: file_exists, file_contains, python_predicate, shell_command, custom. "
            "Use only safe shell_command checks and never destructive commands. "
            f"Respect limits: {context.max_sub_plans} sub plans, "
            f"{context.max_steps_per_sub_plan} steps each."
        )


def _make_plan_id() -> str:
    return f"up-{uuid.uuid4().hex[:12]}"


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerFailedError(f"planner returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlannerFailedError("planner response must be a JSON object")
    return data


def _validate_plan_shape(data: dict[str, Any], context: PlannerContext) -> None:
    _reject_unknown_keys(data, _PLAN_KEYS, "plan")
    _require_string(data, "title", "plan")
    if "id" in data:
        _require_string(data, "id", "plan")
    if "goal" in data:
        _require_string(data, "goal", "plan")
    if "metadata" in data and not isinstance(data["metadata"], dict):
        raise PlannerFailedError("plan.metadata must be an object")
    sub_plans = data.get("sub_plans")
    if not isinstance(sub_plans, list) or not sub_plans:
        raise PlannerFailedError("planner response must include non-empty sub_plans")
    if len(sub_plans) > context.max_sub_plans:
        raise PlannerFailedError("planner response exceeds sub_plan limit")
    seen_sub_plans: set[str] = set()
    seen_steps: set[str] = set()
    for sp_index, sp in enumerate(sub_plans):
        if not isinstance(sp, dict):
            raise PlannerFailedError("sub_plan entries must be objects")
        where = f"sub_plans[{sp_index}]"
        _reject_unknown_keys(sp, _SUB_PLAN_KEYS, where)
        _require_string(sp, "id", where)
        _require_string(sp, "title", where)
        _require_string(sp, "description", where)
        if sp["id"] in seen_sub_plans:
            raise PlannerFailedError(f"duplicate sub_plan id: {sp['id']}")
        seen_sub_plans.add(sp["id"])
        steps = sp.get("steps")
        if not isinstance(steps, list) or not steps:
            raise PlannerFailedError("each sub_plan must include non-empty steps")
        if len(steps) > context.max_steps_per_sub_plan:
            raise PlannerFailedError("planner response exceeds step limit")
        local_steps: set[str] = set()
        pending_dep_checks: list[tuple[str, list[str]]] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise PlannerFailedError(f"{where}.steps[{step_index}] must be an object")
            step_where = f"{where}.steps[{step_index}]"
            _reject_unknown_keys(step, _STEP_KEYS, step_where)
            _require_string(step, "id", step_where)
            _require_string(step, "title", step_where)
            _require_string(step, "description", step_where)
            if step["id"] in seen_steps:
                raise PlannerFailedError(f"duplicate step id: {step['id']}")
            seen_steps.add(step["id"])
            local_steps.add(step["id"])
            kind = step.get("kind", StepKind.OTHER.value)
            if kind not in {item.value for item in StepKind}:
                raise PlannerFailedError(f"{step_where}.kind is invalid: {kind}")
            depends_on = step.get("depends_on", [])
            if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
                raise PlannerFailedError(f"{step_where}.depends_on must be a string list")
            pending_dep_checks.append((step_where, depends_on))
            criteria = step.get("criteria", [])
            if not isinstance(criteria, list):
                raise PlannerFailedError(f"{step_where}.criteria must be a list")
            for criterion_index, criterion in enumerate(criteria):
                if not isinstance(criterion, dict):
                    raise PlannerFailedError(f"{step_where}.criteria[{criterion_index}] must be an object")
                criterion_where = f"{step_where}.criteria[{criterion_index}]"
                _reject_unknown_keys(criterion, _CRITERIA_KEYS, criterion_where)
                _require_string(criterion, "id", criterion_where)
                _require_string(criterion, "description", criterion_where)
                _require_string(criterion, "kind", criterion_where)
                _require_string(criterion, "target", criterion_where)
                if criterion["kind"] not in {item.value for item in CheckKind}:
                    raise PlannerFailedError(f"{criterion_where}.kind is invalid: {criterion['kind']}")
                if "args" in criterion and not isinstance(criterion["args"], dict):
                    raise PlannerFailedError(f"{criterion_where}.args must be an object")
                if "required" in criterion and not isinstance(criterion["required"], bool):
                    raise PlannerFailedError(f"{criterion_where}.required must be a bool")
        for step_where, deps in pending_dep_checks:
            for dep in deps:
                if dep not in local_steps:
                    raise PlannerFailedError(f"{step_where}.depends_on references unknown step {dep!r}")


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise PlannerFailedError(f"{where} contains unknown fields: {', '.join(unknown)}")


def _require_string(data: dict[str, Any], key: str, where: str) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlannerFailedError(f"{where}.{key} must be a non-empty string")


def _reject_dangerous_acceptance(plan: Plan) -> None:
    payload = json.dumps(plan.to_dict(), ensure_ascii=False).lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in payload:
            raise PlannerFailedError(f"planner produced unsafe acceptance content: {pattern}")
