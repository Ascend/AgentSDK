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

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule
from aura.trainer.config_validation.rules.step_budget import OneStepOffStepBudgetRule


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _plain_value(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if OmegaConf.is_config(value):
        try:
            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            return OmegaConf.to_container(value, resolve=False)
    return value


def _has_configured_value(value: Any) -> bool:
    if value is MISSING or value is None:
        return False
    value = _plain_value(value)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return True


def _combined_source(context: ValidationContext, paths: tuple[str, ...]) -> str:
    sources = {context.source_for(path) for path in paths}
    return next(iter(sources)) if len(sources) == 1 else "mixed"


class QueueCapacityRule(ValidationRule):
    """Require the Aura queue to hold at least one complete training batch."""

    rule_id = "linkage.queue_capacity"
    scopes = frozenset({"linkage"})
    _queue_path = "extras.max_queue_size"
    _batch_path = "data.train_batch_size"

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        max_queue_size = get_config_value(context.train_config, self._queue_path)
        train_batch_size = get_config_value(context.train_config, self._batch_path)
        if not _is_positive_integer(max_queue_size) or not _is_positive_integer(train_batch_size):
            return []
        if max_queue_size >= train_batch_size:
            return []

        paths = (self._queue_path, self._batch_path)
        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                path=self._queue_path,
                value={
                    self._queue_path: max_queue_size,
                    self._batch_path: train_batch_size,
                },
                expected=f"max_queue_size >= train_batch_size ({train_batch_size})",
                source=_combined_source(context, paths),
                message="Queue capacity is smaller than one training batch and may deadlock training.",
                suggestion=f"Set {self._queue_path} to at least {train_batch_size}.",
            )
        ]


class ValidationDataRequiredRule(ValidationRule):
    """Require validation data when one validation trigger is enabled."""

    scopes = frozenset({"linkage"})
    _val_data_path = "data.val_files"

    def __init__(
        self,
        rule_id: str,
        trigger_path: str,
        trigger_description: str,
        is_active: Callable[[Any], bool],
    ):
        self.rule_id = rule_id
        self.trigger_path = trigger_path
        self.trigger_description = trigger_description
        self.is_active = is_active

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        trigger_value = get_config_value(context.train_config, self.trigger_path)
        val_files = get_config_value(context.train_config, self._val_data_path)
        if not self.is_active(trigger_value) or _has_configured_value(val_files):
            return []

        paths = (self.trigger_path, self._val_data_path)
        return [
            ValidationIssue(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                path=self._val_data_path,
                value={
                    self.trigger_path: _plain_value(trigger_value),
                    self._val_data_path: _plain_value(val_files),
                },
                expected=f"a non-empty {self._val_data_path} when {self.trigger_description}",
                source=_combined_source(context, paths),
                message=f"Validation data is required because {self.trigger_description}.",
                suggestion=f"Configure {self._val_data_path} or disable {self.trigger_path}.",
            )
        ]


def build_linkage_rules() -> tuple[ValidationRule, ...]:
    """Build rules whose validity depends on multiple configuration fields."""
    return (
        OneStepOffStepBudgetRule(),
        QueueCapacityRule(),
        ValidationDataRequiredRule(
            "linkage.trainer_validate_requires_val_data",
            "async_training.use_trainer_do_validate",
            "async_training.use_trainer_do_validate is True",
            lambda value: value is True,
        ),
        ValidationDataRequiredRule(
            "linkage.test_freq_requires_val_data",
            "trainer.test_freq",
            "trainer.test_freq is a positive integer",
            _is_positive_integer,
        ),
    )
