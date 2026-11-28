"""CPU-only TP checks with tiny synthetic config.json files, not model weights."""
import json
from dataclasses import replace

import pytest

from aura.trainer.config_validation import ValidationContext, ValidationOptions
from aura.trainer.config_validation.models import ConfigValidationError, Severity
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.parallelism import build_parallelism_rules
from aura.trainer.config_validation.models import ValidationReport


class ConfigValidator:
    def __init__(self, registry):
        self.registry = registry

    def validate(self, context, scopes=None):
        return ValidationReport(
            issues=tuple(issue for rule in self.registry.get_rules(scopes)
                         if (context.options.show_default_warnings or not rule.default_value_notice)
                         and rule.is_enabled(context) for issue in rule.validate(context)),
            sources=context.source_map,
        )

    def validate_or_raise(self, context, scopes=None):
        report = self.validate(context, scopes)
        if report.has_errors:
            from aura.trainer.config_validation.models import ConfigValidationError
            raise ConfigValidationError(report)
        return report


def build_default_registry():
    registry = RuleRegistry()
    registry.register_many(build_parallelism_rules())
    return registry

def context(tmp_path, train_tp=4, infer_tp=2, dp=1, heads=32, kv=8, strategy="fsdp2", model_type="qwen3"):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": model_type, "num_attention_heads": heads, "num_key_value_heads": kv,
        "hidden_size": 2560, "head_dim": 128,
    }), encoding="utf-8")
    return ValidationContext(train_config={
        "actor_rollout_ref": {"model": {"path": str(tmp_path)},
                             "actor": {"strategy": strategy},
                             "rollout": {"tensor_model_parallel_size": train_tp}},
    }, rollout_config={"train_tensor_parallel_size": train_tp,
                       "infer_tensor_parallel_size": infer_tp, "infer_expert_parallel_size": dp},
        work_mode="one_step_off", source_map={"rollout_config.infer_tensor_parallel_size": "user_explicit"})


def validator():
    registry = RuleRegistry()
    registry.register_many(build_parallelism_rules())
    return ConfigValidator(registry)


@pytest.mark.parametrize("train_tp,infer_tp", [(4, 2), (2, 4), (8, 8), (4, 1), (4, 16), (4, 32)])
def test_legal_independent_tp_and_kv_replication(tmp_path, train_tp, infer_tp):
    report = validator().validate(context(tmp_path, train_tp=train_tp, infer_tp=infer_tp))
    assert report.valid
    assert all(issue.severity == Severity.INFO for issue in report.issues)
    assert all(issue.source == "user_explicit" for issue in report.issues)


@pytest.mark.parametrize("field", ["train_tensor_parallel_size", "infer_tensor_parallel_size", "infer_expert_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, True, "2", 1.5, None])
def test_invalid_parallel_sizes(tmp_path, field, value):
    current = context(tmp_path)
    current.rollout_config[field] = value
    report = validator().validate(current)
    assert report.has_errors
    assert any(issue.path == "rollout_config." + field and issue.severity == Severity.ERROR
               for issue in report.issues)


@pytest.mark.parametrize("field", ["train_tensor_parallel_size", "infer_tensor_parallel_size", "infer_expert_parallel_size"])
def test_missing_parallel_sizes(tmp_path, field):
    current = context(tmp_path)
    del current.rollout_config[field]
    assert validator().validate(current).has_errors


@pytest.mark.parametrize("tp", [3, 6, 12, 64])
def test_attention_head_tp_conflicts(tmp_path, tp):
    report = validator().validate(context(tmp_path, infer_tp=tp))
    assert report.has_errors
    assert any(issue.rule_id == "paths.qwen3_inference_tp" for issue in report.issues)


def test_kv_heads_need_partition_or_replication(tmp_path):
    assert validator().validate(context(tmp_path, infer_tp=4, heads=24, kv=6)).has_errors


def test_null_kv_heads_follow_qwen3_mha_default(tmp_path):
    assert validator().validate(context(tmp_path, kv=None)).valid


def test_missing_kv_heads_use_qwen3_default_32(tmp_path):
    current = context(tmp_path, infer_tp=8, heads=24)
    path = tmp_path / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["num_key_value_heads"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert validator().validate(current).valid


@pytest.mark.parametrize("heads,kv", [(0, 8), (32, 0), (True, 8), (32, "8"), (None, 8)])
def test_invalid_model_head_counts(tmp_path, heads, kv):
    assert validator().validate(context(tmp_path, heads=heads, kv=kv)).has_errors


def test_bad_json_is_configuration_error_not_internal_error(tmp_path):
    current = context(tmp_path)
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    report = validator().validate(current)
    assert report.has_errors
    assert all(not issue.rule_id.endswith(".internal") for issue in report.issues)


def test_unknown_architecture_is_not_claimed_verified(tmp_path):
    report = validator().validate(context(tmp_path, model_type="other"))
    assert report.valid
    assert any(issue.severity == Severity.WARNING for issue in report.issues)


@pytest.mark.parametrize("mode", ["", "hybrid"])
def test_other_modes_are_not_changed(tmp_path, mode):
    report = validator().validate(replace(context(tmp_path, infer_tp=0), work_mode=mode))
    assert report.issues == ()


def test_remote_stage_without_rollout_skips(tmp_path):
    assert validator().validate(replace(context(tmp_path), rollout_config=None)).issues == ()


def test_rule_registration_and_io_classification():
    rules = {rule.rule_id: rule for rule in build_default_registry().get_rules()}
    assert not rules["consistency.rollout.parallel_values"].io_bound
    assert rules["paths.qwen3_inference_tp"].io_bound


def test_strict_startup_rejects_invalid_tp(tmp_path):
    with pytest.raises(ConfigValidationError):
        validator().validate_or_raise(context(tmp_path, infer_tp=3))
