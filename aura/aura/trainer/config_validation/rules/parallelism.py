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

"""Startup-only external inference parallelism; never import accelerator packages."""
import json
import os
from pathlib import Path

from aura.trainer.config_validation.access import MISSING, get_config_value
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


def positive(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class RolloutParallelValuesRule(ValidationRule):
    rule_id = "consistency.rollout.parallel_values"
    scopes = frozenset({"consistency"})

    def is_enabled(self, context):
        return context.work_mode in {"one_step_off", "fully_async"} and context.rollout_config is not None

    def issue(self, context, path, value, expected, message, severity=Severity.ERROR):
        return ValidationIssue(
            rule_id=self.rule_id, severity=severity, path="rollout_config." + path,
            value="<missing>" if value is MISSING else value, expected=expected,
            source=context.source_for(path, "rollout_config"), message=message,
            suggestion=expected,
        )

    def validate(self, context):
        fields = ("train_tensor_parallel_size", "infer_tensor_parallel_size", "infer_expert_parallel_size")
        values = {key: get_config_value(context.rollout_config, key) for key in fields}
        issues = [self.issue(context, key, value, "a positive non-boolean integer",
                             "The consumed rollout parallel size is invalid.")
                  for key, value in values.items() if not positive(value)]
        strategy = get_config_value(context.train_config, "actor_rollout_ref.actor.strategy")
        if not issues and strategy in ("fsdp", "fsdp2"):
            issues.append(self.issue(
                context, "infer_tensor_parallel_size", values,
                "Validate each topology separately; external inference TP need not equal training-rollout TP.",
                "FSDP sharding is not tensor parallelism; HF weight loading permits different TP configurations.",
                Severity.INFO,
            ))
        return issues


class Qwen3InferenceTPRule(RolloutParallelValuesRule):
    rule_id = "paths.qwen3_inference_tp"
    scopes = frozenset({"paths"})
    io_bound = True

    def is_enabled(self, context):
        return super().is_enabled(context) and positive(
            get_config_value(context.rollout_config, "infer_tensor_parallel_size")
        )

    def validate(self, context):
        tp = get_config_value(context.rollout_config, "infer_tensor_parallel_size")
        model = get_config_value(context.train_config, "actor_rollout_ref.model.path")
        if not isinstance(model, (str, os.PathLike)) or not os.fspath(model).strip():
            return []  # paths.model reports missing/invalid paths.
        path = Path(os.path.expandvars(os.path.expanduser(os.fspath(model)))) / "config.json"
        try:
            if not path.is_file():
                return []  # paths.model reports missing config.json.
            with path.open(encoding="utf-8") as stream:
                model_config = json.load(stream)
            if not isinstance(model_config, dict):
                raise ValueError("config.json must contain an object")
        except (OSError, ValueError) as error:
            return [self.issue(context, "infer_tensor_parallel_size", tp,
                               "a readable valid model config.json", str(error))]
        if model_config.get("model_type") != "qwen3":
            return [self.issue(context, "infer_tensor_parallel_size", tp,
                               "Check TP geometry using the actual model implementation.",
                               "This architecture-specific rule only covers dense Qwen3.", Severity.WARNING)]
        heads = model_config.get("num_attention_heads", 32)
        kv = model_config.get("num_key_value_heads", 32)
        if kv is None:  # Qwen3Config backward-compatible MHA default.
            kv = heads
        expected = "Qwen3 requires heads % TP == 0 and (KV % TP == 0 or TP % KV == 0)."
        value = {"infer_tp": tp, "num_attention_heads": heads, "num_key_value_heads": kv}
        if not positive(heads) or not positive(kv):
            return [self.issue(context, "infer_tensor_parallel_size", value,
                               "positive integer attention/KV head counts in config.json",
                               "The model head counts are invalid.")]
        if heads % tp or (kv % tp and tp % kv):
            return [self.issue(context, "infer_tensor_parallel_size", value, expected,
                               "External inference TP cannot partition or replicate Qwen3 attention heads.")]
        return [self.issue(context, "infer_tensor_parallel_size", value, expected,
                           "Qwen3 attention TP geometry passed; device capacity and runtime compatibility remain separate checks.",
                           Severity.INFO)]


def build_parallelism_rules():
    return (RolloutParallelValuesRule(), Qwen3InferenceTPRule())
