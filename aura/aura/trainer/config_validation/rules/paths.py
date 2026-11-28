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

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.context import ValidationContext
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


TRAIN_CONFIG = "train_config"
ROLLOUT_CONFIG = "rollout_config"
ONE_STEP_OFF_MODE = "one_step_off"


def _is_non_empty(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return next(path.iterdir(), None) is not None
    return False


class FileSystemRule(ValidationRule):
    """Base rule for startup-only path and filesystem validation."""

    scopes = frozenset({"paths"})
    io_bound = True

    def __init__(self, rule_id: str, namespace: str, path: str, expected: str):
        self.rule_id = rule_id
        self.namespace = namespace
        self.path = path
        self.expected = expected

    def is_enabled(self, context: ValidationContext) -> bool:
        return self.namespace != ROLLOUT_CONFIG or context.rollout_config is not None

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        value = get_config_value(self._config(context), self.path)
        resolved_path, input_error = self._resolve(value)
        if input_error:
            return [self._issue(context, value, input_error)]

        try:
            problem = self._check_with_context(resolved_path, context)
        except (OSError, ValueError) as exc:
            problem = f"filesystem check failed: {exc}"
        return [] if problem is None else [self._issue(context, value, problem)]

    def _config(self, context: ValidationContext) -> Any:
        return context.train_config if self.namespace == TRAIN_CONFIG else context.rollout_config

    @staticmethod
    def _resolve(value: Any) -> tuple[Path | None, str | None]:
        if value is MISSING:
            return None, "configuration value is missing"
        if not isinstance(value, (str, os.PathLike)):
            return None, "path must be a string or os.PathLike value"
        raw_path = os.fspath(value)
        if not raw_path.strip():
            return None, "path must not be empty"
        expanded_path = os.path.expandvars(os.path.expanduser(raw_path))
        return Path(expanded_path), None

    def _check(self, path: Path) -> str | None:
        raise NotImplementedError

    def _check_with_context(self, path: Path, context: ValidationContext) -> str | None:
        return self._check(path)

    def _issue(self, context: ValidationContext, value: Any, problem: str) -> ValidationIssue:
        if value is MISSING:
            current_value = "<missing>"
        elif isinstance(value, (str, os.PathLike)):
            current_value = os.fspath(value)
        else:
            current_value = value
        full_path = f"{self.namespace}.{self.path}"
        return ValidationIssue(
            rule_id=self.rule_id,
            severity=Severity.ERROR,
            path=full_path,
            value=current_value,
            expected=self.expected,
            source=context.source_for(self.path, self.namespace),
            message=f"Filesystem requirement failed for {full_path}: {problem}.",
            suggestion=f"Set {full_path} to {self.expected}.",
        )


class RequiredFileInDirectoryRule(FileSystemRule):
    """Require a readable directory containing one named readable file."""

    def __init__(self, rule_id: str, namespace: str, path: str, required_file: str):
        expected = f"an existing readable directory containing readable {required_file}"
        super().__init__(rule_id, namespace, path, expected)
        self.required_file = required_file

    def _check(self, path: Path) -> str | None:
        if not path.exists():
            return "path does not exist"
        if not path.is_dir():
            return "path is not a directory"
        if not os.access(path, os.R_OK):
            return "directory is not readable"

        required_path = path / self.required_file
        if not required_path.is_file():
            return f"required file {self.required_file} does not exist"
        if not os.access(required_path, os.R_OK):
            return f"required file {self.required_file} is not readable"
        return None


class NonEmptyReadablePathRule(FileSystemRule):
    """Require an existing, readable, non-empty file or directory."""

    def _check(self, path: Path) -> str | None:
        if not path.exists():
            return "path does not exist"
        if not os.access(path, os.R_OK):
            return "path is not readable"
        if not _is_non_empty(path):
            return "path is empty"
        return None


class OneStepOffDataPathRule(NonEmptyReadablePathRule):
    """Accept readable paths or the packed dataset prefix used by PromptDataset."""

    def is_enabled(self, context: ValidationContext) -> bool:
        return context.work_mode in {ONE_STEP_OFF_MODE, "fully_async"} and super().is_enabled(context)

    def _check_with_context(self, path: Path, context: ValidationContext) -> str | None:
        if path.exists():
            return self._check(path)
        keys = get_config_value(
            context.train_config, "extras.data_loader.dataset_additional_keys", default=[]
        )
        if OmegaConf.is_config(keys):
            keys = OmegaConf.to_container(keys, resolve=True)
        if keys is None:
            keys = []
        if not isinstance(keys, (list, tuple)) or any(
            not isinstance(key, str) or not key or any(char in key for char in "/\\\0")
            for key in keys
        ):
            return "dataset_additional_keys must be a list of non-empty field names"
        for key in dict.fromkeys(["input_ids", "attention_mask", *keys]):
            for suffix in ("bin", "idx"):
                member = path.parent / f"{path.name}_packed_{key}_document.{suffix}"
                if not member.is_file():
                    return f"dataset prefix requires file {member.name}, which does not exist or is not a file"
                if not os.access(member, os.R_OK):
                    return f"dataset prefix file {member.name} is not readable"
                if member.stat().st_size == 0:
                    return f"dataset prefix file {member.name} is empty"
        return None


class OneStepOffTestDataPathRule(OneStepOffDataPathRule):
    """Validate explicitly supplied test data; an unset path disables its loader."""

    def is_enabled(self, context: ValidationContext) -> bool:
        if not super().is_enabled(context):
            return False
        value = get_config_value(context.train_config, self.path)
        return value is not MISSING and value is not None and value != ""


def build_path_rules() -> tuple[ValidationRule, ...]:
    """Build startup-only path rules; the validator executes them concurrently."""
    return (
        RequiredFileInDirectoryRule(
            "paths.model",
            TRAIN_CONFIG,
            "actor_rollout_ref.model.path",
            "config.json",
        ),
        OneStepOffDataPathRule(
            "paths.train_data",
            TRAIN_CONFIG,
            "extras.data_loader.train_data_path",
            "an existing readable non-empty file/directory or a prefix with complete readable non-empty packed bin/idx files",
        ),
        OneStepOffTestDataPathRule(
            "paths.test_data",
            TRAIN_CONFIG,
            "extras.data_loader.test_data_path",
            "a readable non-empty file/directory or complete packed bin/idx prefix when test_data_path is supplied",
        ),
        RequiredFileInDirectoryRule(
            "paths.llm_tokenizer",
            ROLLOUT_CONFIG,
            "llm_tokenizer_path",
            "tokenizer_config.json",
        ),
    )
