#!/usr/bin/env python3
# coding=utf-8
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

from typing import Any

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


SUPPORTED_ACTOR_STRATEGIES = frozenset({"fsdp", "fsdp2", "megatron"})
ONE_STEP_OFF_MODE = "one_step_off"


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_power_of_two(value: Any) -> bool:
    return _is_positive_integer(value) and (value & (value - 1)) == 0


class VerlValueRule(ValidationRule):
    """Base class for required verl and Ascend configuration constraints."""

    scopes = frozenset({"verl"})

    def __init__(self, rule_id: str, path: str, expected: str, work_modes: frozenset[str] | None = None):
        self.rule_id = rule_id
        self.path = path
        self.expected = expected
        self.work_modes = work_modes

    def is_enabled(self, context: ValidationContext) -> bool:
        return self.work_modes is None or context.work_mode in self.work_modes

    def _read(self, context: ValidationContext) -> Any:
        return get_config_value(context.train_config, self.path)

    def _issue(self, context: ValidationContext, value: Any) -> ValidationIssue:
        current_value = "<missing>" if value is MISSING else value
        return ValidationIssue(
            rule_id=self.rule_id,
            severity=Severity.ERROR,
            path=self.path,
            value=current_value,
            expected=self.expected,
            source=context.source_for(self.path),
            message=f"Parameter {self.path} violates an Aura verl/Ascend requirement.",
            suggestion=f"Set {self.path} to {self.expected}.",
        )


class StringChoiceRule(VerlValueRule):
    """Require a string value from a supported set."""

    def __init__(self, rule_id: str, path: str, choices: frozenset[str]):
        expected = "one of {" + ", ".join(sorted(choices)) + "}"
        super().__init__(rule_id, path, expected)
        self.choices = choices

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        return [] if isinstance(value, str) and value in self.choices else [self._issue(context, value)]


class ExactValueRule(VerlValueRule):
    """Require an exact value and exact Python type."""

    def __init__(
        self,
        rule_id: str,
        path: str,
        expected_value: Any,
        work_modes: frozenset[str] | None = None,
    ):
        super().__init__(rule_id, path, repr(expected_value))
        self.expected_value = expected_value
        self.work_modes = work_modes

    def is_enabled(self, context: ValidationContext) -> bool:
        return self.work_modes is None or context.work_mode in self.work_modes

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        valid = type(value) is type(self.expected_value) and value == self.expected_value
        return [] if valid else [self._issue(context, value)]


class PositiveIntegerRule(VerlValueRule):
    """Require a non-boolean positive integer."""

    def __init__(self, rule_id: str, path: str, work_modes: frozenset[str] | None = None):
        super().__init__(rule_id, path, "integer > 0", work_modes)

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        return [] if _is_positive_integer(value) else [self._issue(context, value)]


class PowerOfTwoRule(VerlValueRule):
    """Require a positive integer power of two."""

    def __init__(self, rule_id: str, path: str):
        super().__init__(rule_id, path, "positive integer power of two")

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = self._read(context)
        return [] if _is_power_of_two(value) else [self._issue(context, value)]


class TopologyRule(ValidationRule):
    """Require TP times DP to equal the total configured NPU count."""

    rule_id = "verl.topology.tp_dp"
    scopes = frozenset({"verl"})
    _paths = (
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.data_parallel_size",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    )

    def is_enabled(self, context: ValidationContext) -> bool:
        # Hybrid rollout replicas derive their topology inside VERL.
        return context.work_mode in {ONE_STEP_OFF_MODE, "fully_async"}

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        tp, dp, n_gpus_per_node, nnodes = (
            get_config_value(context.train_config, path) for path in self._paths
        )
        valid_inputs = (
            _is_power_of_two(tp)
            and _is_positive_integer(dp)
            and _is_positive_integer(n_gpus_per_node)
            and _is_positive_integer(nnodes)
        )
        if not valid_inputs:
            return []

        parallel_size = tp * dp
        device_count = n_gpus_per_node * nnodes
        if parallel_size == device_count:
            return []

        expected = (
            f"TP({tp}) * DP({dp}) == n_gpus_per_node({n_gpus_per_node}) * nnodes({nnodes}); "
            f"evaluated as {parallel_size} == {device_count}"
        )
        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                path="actor_rollout_ref.rollout.tensor_model_parallel_size",
                value={
                    "tensor_model_parallel_size": tp,
                    "data_parallel_size": dp,
                    "n_gpus_per_node": n_gpus_per_node,
                    "nnodes": nnodes,
                },
                expected=expected,
                source=_combined_source(context, self._paths),
                message="The model parallel topology does not match the configured NPU topology.",
                suggestion="Adjust TP, DP, n_gpus_per_node, or nnodes so both products are equal.",
            )
        ]


class BatchDivisibilityRule(ValidationRule):
    """Require the sampled training batch to divide evenly across NPUs."""

    rule_id = "verl.topology.train_batch_divisibility"
    scopes = frozenset({"verl"})
    _paths = (
        "data.train_batch_size",
        "actor_rollout_ref.rollout.n",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    )

    def is_enabled(self, context: ValidationContext) -> bool:
        # Streaming batches use ppo_mini_batch_size * require_batches prompt
        # groups, not data.train_batch_size; FullyAsyncSamplingRule checks them.
        return context.work_mode != "fully_async"

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        train_batch_size, rollout_n, n_gpus_per_node, nnodes = (
            get_config_value(context.train_config, path) for path in self._paths
        )
        values = (train_batch_size, rollout_n, n_gpus_per_node, nnodes)
        if not all(_is_positive_integer(value) for value in values):
            return []

        sampled_batch_size = train_batch_size * rollout_n
        device_count = n_gpus_per_node * nnodes
        remainder = sampled_batch_size % device_count
        if remainder == 0:
            return []

        expected = (
            f"(train_batch_size({train_batch_size}) * rollout.n({rollout_n})) "
            f"% total_npus({device_count}) == 0; remainder is {remainder}"
        )
        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                path="data.train_batch_size",
                value={
                    "train_batch_size": train_batch_size,
                    "rollout_n": rollout_n,
                    "total_npus": device_count,
                },
                expected=expected,
                source=_combined_source(context, self._paths),
                message="The sampled training batch cannot be divided evenly across all configured NPUs.",
                suggestion="Adjust train_batch_size, rollout.n, or the NPU topology to remove the remainder.",
            )
        ]


def _combined_source(context: ValidationContext, paths: tuple[str, ...]) -> str:
    sources = {context.source_for(path) for path in paths}
    return next(iter(sources)) if len(sources) == 1 else "mixed"


def build_verl_rules() -> tuple[ValidationRule, ...]:
    """Build Aura's verl and Ascend NPU compatibility rule set."""
    return (
        StringChoiceRule(
            "verl.actor.strategy",
            "actor_rollout_ref.actor.strategy",
            SUPPORTED_ACTOR_STRATEGIES,
        ),
        ExactValueRule(
            "verl.rollout.name",
            "actor_rollout_ref.rollout.name",
            "vllm",
        ),
        ExactValueRule(
            "verl.rollout.mode",
            "actor_rollout_ref.rollout.mode",
            "async",
            work_modes=frozenset({ONE_STEP_OFF_MODE, "fully_async"}),
        ),
        ExactValueRule(
            "verl.rollout.calculate_log_probs",
            "actor_rollout_ref.rollout.calculate_log_probs",
            True,
            work_modes=frozenset({ONE_STEP_OFF_MODE}),
        ),
        ExactValueRule(
            "verl.actor.use_rollout_log_probs",
            "actor_rollout_ref.actor.use_rollout_log_probs",
            True,
            work_modes=frozenset({ONE_STEP_OFF_MODE}),
        ),
        ExactValueRule(
            "verl.model.use_remove_padding",
            "actor_rollout_ref.model.use_remove_padding",
            True,
            work_modes=frozenset({ONE_STEP_OFF_MODE}),
        ),
        ExactValueRule(
            "verl.trainer.device",
            "trainer.device",
            "npu",
        ),
        PowerOfTwoRule(
            "verl.rollout.tensor_model_parallel_size",
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
        ),
        PositiveIntegerRule(
            "verl.rollout.data_parallel_size",
            "actor_rollout_ref.rollout.data_parallel_size",
            work_modes=frozenset({ONE_STEP_OFF_MODE, "fully_async"}),
        ),
        PositiveIntegerRule(
            "verl.rollout.n",
            "actor_rollout_ref.rollout.n",
        ),
        PositiveIntegerRule(
            "verl.trainer.n_gpus_per_node",
            "trainer.n_gpus_per_node",
        ),
        PositiveIntegerRule(
            "verl.trainer.nnodes",
            "trainer.nnodes",
        ),
        PositiveIntegerRule(
            "verl.data.train_batch_size",
            "data.train_batch_size",
        ),
        TopologyRule(),
        BatchDivisibilityRule(),
    )
