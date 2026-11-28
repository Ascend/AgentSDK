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

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


TRAIN_CONFIG = "train_config"
ROLLOUT_CONFIG = "rollout_config"


def _display_value(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if OmegaConf.is_config(value):
        try:
            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            return OmegaConf.to_container(value, resolve=False)
    return value


def _values_match(expected: Any, actual: Any) -> bool:
    if expected is MISSING or actual is MISSING:
        return False
    expected = _display_value(expected)
    actual = _display_value(actual)
    return type(expected) is type(actual) and expected == actual


class ConfigEqualityRule(ValidationRule):
    """Require one duplicated configuration value to equal its source of truth."""

    scopes = frozenset({"consistency"})

    def __init__(
        self,
        rule_id: str,
        expected_namespace: str,
        expected_path: str,
        actual_namespace: str,
        actual_path: str,
    ):
        self.rule_id = rule_id
        self.expected_namespace = expected_namespace
        self.expected_path = expected_path
        self.actual_namespace = actual_namespace
        self.actual_path = actual_path

    def is_enabled(self, context: ValidationContext) -> bool:
        namespaces = {self.expected_namespace, self.actual_namespace}
        return ROLLOUT_CONFIG not in namespaces or context.rollout_config is not None

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        expected = self._read(context, self.expected_namespace, self.expected_path)
        actual = self._read(context, self.actual_namespace, self.actual_path)
        if _values_match(expected, actual):
            return []

        expected_label = f"{self.expected_namespace}.{self.expected_path}"
        actual_label = f"{self.actual_namespace}.{self.actual_path}"
        expected_source = context.source_for(self.expected_path, self.expected_namespace)
        actual_source = context.source_for(self.actual_path, self.actual_namespace)
        combined_source = expected_source if expected_source == actual_source else "mixed"

        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                path=actual_label,
                value={
                    expected_label: {
                        "value": _display_value(expected),
                        "source": expected_source,
                    },
                    actual_label: {
                        "value": _display_value(actual),
                        "source": actual_source,
                    },
                },
                expected=f"exactly equal {expected_label}",
                source=combined_source,
                message=f"Duplicated parameters {expected_label} and {actual_label} are inconsistent.",
                suggestion=f"Set {actual_label} to the exact value of {expected_label}.",
            )
        ]

    @staticmethod
    def _read(context: ValidationContext, namespace: str, path: str) -> Any:
        config = context.train_config if namespace == TRAIN_CONFIG else context.rollout_config
        return get_config_value(config, path)


def build_consistency_rules() -> tuple[ValidationRule, ...]:
    """Build train-internal and train-to-rollout consistency rules."""
    return (
        ConfigEqualityRule(
            "consistency.train.rollout_n",
            TRAIN_CONFIG,
            "actor_rollout_ref.rollout.n",
            TRAIN_CONFIG,
            "rollout.n",
        ),
        ConfigEqualityRule(
            "consistency.train.rollout_nnodes",
            TRAIN_CONFIG,
            "trainer.nnodes",
            TRAIN_CONFIG,
            "rollout.nnodes",
        ),
        ConfigEqualityRule(
            "consistency.train.rollout_n_gpus_per_node",
            TRAIN_CONFIG,
            "trainer.n_gpus_per_node",
            TRAIN_CONFIG,
            "rollout.n_gpus_per_node",
        ),
        ConfigEqualityRule(
            "consistency.rollout.n_samples_per_prompt",
            TRAIN_CONFIG,
            "actor_rollout_ref.rollout.n",
            ROLLOUT_CONFIG,
            "n_samples_per_prompt",
        ),
        ConfigEqualityRule(
            "consistency.rollout.tokenizer_name_or_path",
            TRAIN_CONFIG,
            "actor_rollout_ref.model.path",
            ROLLOUT_CONFIG,
            "tokenizer_name_or_path",
        ),
        ConfigEqualityRule(
            "consistency.rollout.dataset_additional_keys",
            TRAIN_CONFIG,
            "extras.data_loader.dataset_additional_keys",
            ROLLOUT_CONFIG,
            "dataset_additional_keys",
        ),
        ConfigEqualityRule(
            "consistency.rollout.global_batch_size",
            TRAIN_CONFIG,
            "extras.data_loader.global_batch_size",
            ROLLOUT_CONFIG,
            "global_batch_size",
        ),
        ConfigEqualityRule(
            "consistency.rollout.train_tensor_parallel_size",
            TRAIN_CONFIG,
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            ROLLOUT_CONFIG,
            "train_tensor_parallel_size",
        ),
    )
