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

"""AcceptanceVerifier tests: file/shell/predicate/custom check kinds."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from src.services.ultraplan import (
    AcceptanceCriteria,
    AcceptanceVerifier,
    CheckKind,
    CheckResult,
    Plan,
    Step,
    SubPlan,
    VerificationCheckFailedError,
)


def _plan_with_step(step: Step) -> Plan:
    sp = SubPlan(id="sp1", title="A", description="d", steps=[step])
    return Plan(id="p1", title="My plan", goal="Goal", sub_plans=[sp])


# ---------------------------------------------------------------------------
# FILE_EXISTS
# ---------------------------------------------------------------------------


def test_file_exists_passes(tmp_path: Path) -> None:
    f = tmp_path / "real.txt"
    f.write_text("hello", encoding="utf-8")
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_EXISTS,
                    target=str(f),
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_file_exists_fails_on_missing(tmp_path: Path) -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_EXISTS,
                    target=str(tmp_path / "missing.txt"),
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


def test_file_exists_rejects_relative_path() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_EXISTS,
                    target="relative/path.txt",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False
    assert "not absolute" in result.details


# ---------------------------------------------------------------------------
# FILE_CONTAINS
# ---------------------------------------------------------------------------


def test_file_contains_passes(tmp_path: Path) -> None:
    f = tmp_path / "real.txt"
    f.write_text("hello world", encoding="utf-8")
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_CONTAINS,
                    target=str(f),
                    args={"substring": "world"},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_file_contains_fails_on_missing_substring(tmp_path: Path) -> None:
    f = tmp_path / "real.txt"
    f.write_text("hello", encoding="utf-8")
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_CONTAINS,
                    target=str(f),
                    args={"substring": "missing"},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


def test_file_contains_requires_substring_arg() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.FILE_CONTAINS,
                    target="/tmp/x",
                    args={},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


# ---------------------------------------------------------------------------
# SHELL_COMMAND
# ---------------------------------------------------------------------------


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX-only test")
def test_shell_command_passes_on_exit_0() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.SHELL_COMMAND,
                    target="true",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan, default_timeout=5.0)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX-only test")
def test_shell_command_fails_on_nonzero() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.SHELL_COMMAND,
                    target="false",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan, default_timeout=5.0)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False
    assert "exited 1" in result.details


def test_shell_command_rejects_unsafe_timeout() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.SHELL_COMMAND,
                    target="true",
                    args={"timeout": -1},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan, default_timeout=5.0)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False
    assert "unsafe timeout" in result.details


def test_shell_command_tokenizes_via_shlex() -> None:
    """A command like ``echo hi ; rm -rf /`` is tokenized and run safely."""
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.SHELL_COMMAND,
                    target=f"{sys.executable} -c print(1)",  # safe tokenized
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan, default_timeout=5.0)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


# ---------------------------------------------------------------------------
# PYTHON_PREDICATE
# ---------------------------------------------------------------------------


def test_python_predicate_passes() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="1 + 1 == 2",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_python_predicate_uses_context() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="len(values) > 2",
                    args={"context": {"values": [1, 2, 3]}},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan, predicate_context={"values": [1]})  # ignored
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_python_predicate_fails_on_falsy() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="1 == 2",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


def test_python_predicate_rejects_imports() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="__import__('os').system('echo hi')",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False
    assert "predicate raised" in result.details or "not valid" in result.details


def test_python_predicate_rejects_syntax_error() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="this is not python",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


def test_python_predicate_rejects_underscore_method_call() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.PYTHON_PREDICATE,
                    target="x._private()",
                    args={"context": {"x": object()}},
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False


# ---------------------------------------------------------------------------
# CUSTOM
# ---------------------------------------------------------------------------


def test_custom_check_passes() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.CUSTOM,
                    target="is_positive",
                    args={"value": 5},
                )
            ],
        )
    )

    def is_positive(target: str, args: dict) -> CheckResult:
        return CheckResult(args["value"] > 0, details=f"value={args['value']}")

    v = AcceptanceVerifier(plan, custom_checks={"is_positive": is_positive})
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_custom_check_unknown_returns_failed() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.CUSTOM,
                    target="missing",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    # The public ``verify_criterion`` is the safe API: check-dispatch
    # errors are returned as a failed ``CheckResult`` rather than
    # raised, so callers can iterate without try/except.
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is False
    assert "missing" in result.details


def test_register_custom_check() -> None:
    plan = _plan_with_step(
        Step(
            id="s1",
            title="T",
            description="D",
            criteria=[
                AcceptanceCriteria(
                    id="c1",
                    description="d",
                    kind=CheckKind.CUSTOM,
                    target="my_check",
                )
            ],
        )
    )
    v = AcceptanceVerifier(plan)
    v.register_custom_check("my_check", lambda t, a: CheckResult(True, "ok"))
    result = v.verify_criterion(plan.sub_plans[0].steps[0].criteria[0])
    assert result.passed is True


def test_register_custom_check_rejects_empty_name() -> None:
    plan = Plan(id="p1", title="x", goal="x")
    v = AcceptanceVerifier(plan)
    with pytest.raises(ValueError):
        v.register_custom_check("", lambda t, a: CheckResult(True))


def test_register_custom_check_rejects_non_callable() -> None:
    plan = Plan(id="p1", title="x", goal="x")
    v = AcceptanceVerifier(plan)
    with pytest.raises(TypeError):
        v.register_custom_check("x", "not-callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plan-level verification
# ---------------------------------------------------------------------------


def test_verify_step_returns_per_criterion_results() -> None:
    step = Step(
        id="s1",
        title="T",
        description="D",
        criteria=[
            AcceptanceCriteria(id="c1", description="d", kind=CheckKind.PYTHON_PREDICATE, target="True"),
            AcceptanceCriteria(id="c2", description="d", kind=CheckKind.PYTHON_PREDICATE, target="False"),
        ],
    )
    plan = _plan_with_step(step)
    v = AcceptanceVerifier(plan)
    results = v.verify_step("s1")
    assert results["c1"].passed is True
    assert results["c2"].passed is False


def test_verify_sub_plan_unknown_raises() -> None:
    plan = Plan(id="p1", title="x", goal="x")
    v = AcceptanceVerifier(plan)
    with pytest.raises(ValueError):
        v.verify_sub_plan("missing")


def test_verify_step_unknown_raises() -> None:
    plan = Plan(id="p1", title="x", goal="x")
    v = AcceptanceVerifier(plan)
    with pytest.raises(ValueError):
        v.verify_step("missing")


def test_is_plan_passing_true_when_all_required_pass() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(
                id="s1",
                title="T",
                description="D",
                criteria=[
                    AcceptanceCriteria(
                        id="c1",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="True",
                    )
                ],
            )
        ],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    v = AcceptanceVerifier(plan)
    assert v.is_plan_passing() is True


def test_is_plan_passing_false_when_required_fails() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(
                id="s1",
                title="T",
                description="D",
                criteria=[
                    AcceptanceCriteria(
                        id="c1",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="False",
                    )
                ],
            )
        ],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    v = AcceptanceVerifier(plan)
    assert v.is_plan_passing() is False


def test_is_plan_passing_ignores_non_required_failures() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(
                id="s1",
                title="T",
                description="D",
                criteria=[
                    AcceptanceCriteria(
                        id="c1",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="True",
                    ),
                    AcceptanceCriteria(
                        id="c2",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="False",
                        required=False,
                    ),
                ],
            )
        ],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    v = AcceptanceVerifier(plan)
    assert v.is_plan_passing() is True


def test_assert_all_required_passing_raises() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(
                id="s1",
                title="T",
                description="D",
                criteria=[
                    AcceptanceCriteria(
                        id="c1",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="False",
                    )
                ],
            )
        ],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    v = AcceptanceVerifier(plan)
    with pytest.raises(VerificationCheckFailedError):
        v.assert_all_required_passing()


def test_assert_all_required_passing_succeeds() -> None:
    sp = SubPlan(
        id="sp1",
        title="A",
        description="d",
        steps=[
            Step(
                id="s1",
                title="T",
                description="D",
                criteria=[
                    AcceptanceCriteria(
                        id="c1",
                        description="d",
                        kind=CheckKind.PYTHON_PREDICATE,
                        target="True",
                    )
                ],
            )
        ],
    )
    plan = Plan(id="p1", title="x", goal="x", sub_plans=[sp])
    v = AcceptanceVerifier(plan)
    v.assert_all_required_passing()  # must not raise


def test_verifier_requires_plan_instance() -> None:
    with pytest.raises(TypeError):
        AcceptanceVerifier({"id": "p1", "title": "x", "goal": "x"})  # type: ignore[arg-type]


def test_verifier_step_with_unknown_check_kind_returns_failed() -> None:
    """If a step somehow holds an unknown kind, the verifier should
    surface the failure for that criterion rather than crashing.
    """
    # Build the criterion directly and stuff it into the step; bypass
    # the model validator by mutating a constructed one.
    c = AcceptanceCriteria(
        id="c1",
        description="d",
        kind=CheckKind.PYTHON_PREDICATE,
        target="True",
    )
    # Replace the kind with an invalid enum value.
    object.__setattr__(c, "kind", "not-a-kind")
    step = Step(id="s1", title="T", description="D", criteria=[c])
    plan = _plan_with_step(step)
    v = AcceptanceVerifier(plan)
    # Patch the registry to remove PYTHON_PREDICATE so it's truly unknown.
    v._registry.pop(CheckKind.PYTHON_PREDICATE, None)  # noqa: SLF001
    result = v.verify_criterion(c)
    assert result.passed is False
