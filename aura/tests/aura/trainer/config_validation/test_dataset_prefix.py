# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""Regression coverage for separated training's packed dataset prefixes."""
import os

import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.paths import build_path_rules
from aura.trainer.config_validation.rules.parallelism import build_parallelism_rules


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
    registry.register_many(build_path_rules())
    registry.register_many(build_parallelism_rules())
    return registry

def validate_prefix(tmp_path, *, test=False, omega=False):
    config = {
        "extras": {"data_loader": {
            "train_data_path": str(tmp_path / "rl"),
            "test_data_path": str(tmp_path / "rl") if test else None,
            "dataset_additional_keys": ["labels"],
        }},
        "rollout": {"test_freq": 1 if test else 0},
    }
    if omega:
        config = OmegaConf.create(config)
    report = ConfigValidator(build_default_registry()).validate(
        ValidationContext(train_config=config, work_mode="one_step_off"), scopes={"paths"}
    )
    return [issue for issue in report.issues if issue.rule_id in {"paths.train_data", "paths.test_data"}]


def make_prefix(tmp_path):
    for key in ("input_ids", "attention_mask", "labels"):
        for suffix in ("bin", "idx"):
            (tmp_path / f"rl_packed_{key}_document.{suffix}").write_bytes(b"data")


@pytest.mark.parametrize("omega", [False, True])
@pytest.mark.parametrize("test", [False, True])
def test_complete_packed_prefix_passes_without_prefix_file(tmp_path, test, omega):
    make_prefix(tmp_path)
    assert not (tmp_path / "rl").exists()
    assert validate_prefix(tmp_path, test=test, omega=omega) == []


@pytest.mark.parametrize("key", ["input_ids", "attention_mask", "labels"])
@pytest.mark.parametrize("suffix", ["bin", "idx"])
@pytest.mark.parametrize("failure", ["missing", "empty", "unreadable", "directory"])
def test_incomplete_or_unreadable_prefix_is_rejected(tmp_path, monkeypatch, key, suffix, failure):
    make_prefix(tmp_path)
    member = tmp_path / f"rl_packed_{key}_document.{suffix}"
    if failure == "missing":
        member.unlink()
    elif failure == "empty":
        member.write_bytes(b"")
    elif failure == "directory":
        member.unlink()
        member.mkdir()
    else:
        real_access = os.access
        monkeypatch.setattr(os, "access", lambda path, mode: False if path == member else real_access(path, mode))
    issues = validate_prefix(tmp_path, test=True)
    assert {issue.rule_id for issue in issues} == {"paths.train_data", "paths.test_data"}
    assert all(member.name in issue.message for issue in issues)


def test_other_prefix_and_disabled_test_data(tmp_path):
    make_prefix(tmp_path)
    member = tmp_path / "rl_packed_labels_document.bin"
    member.rename(tmp_path / "rl_other_packed_labels_document.bin")
    issues = validate_prefix(tmp_path)
    assert [issue.rule_id for issue in issues] == ["paths.train_data"]
