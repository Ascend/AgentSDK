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

"""Capacity, truncation semantics, startup-only I/O and strict-mode regression."""
import ast
import copy
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import (
    ConfigValidationError, RuleRegistry, Severity, ValidationContext,
)
from aura.trainer.config_validation.rules.validation_samples import ValidationSamplesRule, indexed_sample_count
from aura.trainer.config_validation.models import ValidationReport
from aura.trainer.config_validation.registry import RuleRegistry


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
    registry.register(ValidationSamplesRule())
    return registry

def make_index(tmp_path, count=4, boundaries=2):
    path = tmp_path / "rl_packed_input_ids_document.idx"
    arrays = (b"".join(struct.pack("<i", 1) for _ in range(count)) +
              b"".join(struct.pack("<q", 4 * i) for i in range(count)) +
              b"".join(struct.pack("<q", count * i // (boundaries - 1)) for i in range(boundaries)))
    path.write_bytes(b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, count, boundaries) +
                     arrays)
    return path


def context(tmp_path, requested=2, frequency=1, omega=False, mode="one_step_off"):
    config = {"extras": {"validate_num_samples": requested, "data_loader": {
        "test_data_path": str(tmp_path / "rl"),
    }}, "rollout": {"test_freq": frequency}}
    return ValidationContext(train_config=OmegaConf.create(config) if omega else config, work_mode=mode,
                             source_map={"train_config.extras.validate_num_samples": "user_explicit"})


def validator():
    registry = RuleRegistry()
    registry.register(ValidationSamplesRule())
    return ConfigValidator(registry)


@pytest.mark.parametrize("omega", [False, True])
@pytest.mark.parametrize("requested,severity,effective", [
    (1, Severity.INFO, 1), (4, Severity.INFO, 4), (5, Severity.WARNING, 4),
])
def test_capacity_and_runtime_clamping(tmp_path, omega, requested, severity, effective):
    make_index(tmp_path)
    ctx = context(tmp_path, requested, omega=omega)
    before = ctx.snapshot()
    report = validator().validate_or_raise(ctx, scopes={"paths"})
    assert report.valid and len(report.issues) == 1
    issue = report.issues[0]
    assert issue.severity == severity
    assert issue.value["indexed_samples"] == 4
    assert issue.value["effective_samples"] == effective
    assert issue.source == "user_explicit"
    assert ctx.snapshot() == before


@pytest.mark.parametrize("requested,count", [(0, 4), (1, 0)])
def test_zero_selection_or_empty_dataset_fails_strict_mode(tmp_path, requested, count):
    make_index(tmp_path, count)
    with pytest.raises(ConfigValidationError):
        validator().validate_or_raise(context(tmp_path, requested), scopes={"paths"})


def test_supplied_test_data_is_built_even_with_zero_frequency(tmp_path):
    make_index(tmp_path)
    with pytest.raises(ConfigValidationError):
        validator().validate_or_raise(context(tmp_path, 0, frequency=0), scopes={"paths"})


@pytest.mark.parametrize("mode,frequency,path", [
    ("hybrid", 1, "rl"), ("one_step_off", 0, None), ("one_step_off", -1, ""),
    ("one_step_off", 100, None), ("one_step_off", 100, ""),
    ("fully_async", 100, None), ("fully_async", 100, ""),
])
@pytest.mark.parametrize("requested", [0, 2])
def test_inactive_checks_do_not_open_dataset(tmp_path, monkeypatch, mode, frequency, path, requested):
    ctx = context(tmp_path, requested=requested, frequency=frequency, mode=mode)
    ctx.train_config["extras"]["data_loader"]["test_data_path"] = path
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("inactive dataset I/O"))
    assert validator().validate(ctx, scopes={"paths"}).issues == ()


@pytest.mark.parametrize("failure", ["missing", "permission", "header", "arrays", "dtype"])
def test_unreadable_or_malformed_known_index_fails(tmp_path, monkeypatch, failure):
    index = make_index(tmp_path)
    data = index.read_bytes()
    if failure == "missing":
        index.unlink()
    elif failure == "permission":
        real_open = Path.open
        def denied(path, *args, **kwargs):
            if path == index:
                raise PermissionError("denied")
            return real_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", denied)
    elif failure == "header":
        index.write_bytes(data[:20])
    elif failure == "arrays":
        index.write_bytes(data[:-1])
    else:
        index.write_bytes(data[:17] + b"\xff" + data[18:])
    with pytest.raises(ConfigValidationError):
        validator().validate_or_raise(context(tmp_path), scopes={"paths"})


@pytest.mark.parametrize("unsupported", ["magic", "version", "raw_file", "filter"])
def test_unknown_capacity_is_explicit_not_a_false_pass(tmp_path, unsupported):
    index = make_index(tmp_path)
    ctx = context(tmp_path, 100)
    if unsupported == "magic":
        index.write_bytes(b"OTHERIDX!" + index.read_bytes()[9:])
    elif unsupported == "version":
        data = index.read_bytes()
        index.write_bytes(data[:9] + struct.pack("<Q", 2) + data[17:])
    elif unsupported == "raw_file":
        (tmp_path / "rl").write_text("raw", encoding="utf-8")
    else:
        ctx.train_config["data"] = {"max_prompt_length": 512}
    report = validator().validate_or_raise(ctx, scopes={"paths"})
    assert report.issues[0].severity == Severity.WARNING
    assert "unverified" in report.issues[0].message
    value = report.issues[0].value
    assert not isinstance(value, dict) or "effective_samples" not in value


@pytest.mark.parametrize("mode", ["one_step_off", "fully_async"])
@pytest.mark.parametrize("omega", [False, True])
@pytest.mark.parametrize("limit", [None, 512])
def test_data_prompt_limit_marks_filtered_capacity_as_unverified(tmp_path, mode, omega, limit):
    make_index(tmp_path, count=4)
    ctx = context(tmp_path, requested=2, omega=omega, mode=mode)
    ctx.train_config["data"] = {"max_prompt_length": limit}
    before = ctx.snapshot()
    report = validator().validate_or_raise(ctx, scopes={"paths"})
    assert report.valid and len(report.issues) == 1
    issue = report.issues[0]
    assert issue.value["indexed_samples"] == 4
    if limit is None:
        assert issue.severity is Severity.INFO
        assert issue.value["effective_samples"] == 2
    else:
        assert issue.severity is Severity.WARNING
        assert "filtering is unverified" in issue.message
        assert "effective_samples" not in issue.value
    assert ctx.snapshot() == before


def test_packed_loader_prompt_limit_remains_supported(tmp_path):
    make_index(tmp_path)
    ctx = context(tmp_path)
    ctx.train_config["data"] = {"max_prompt_length": None}
    ctx.train_config["extras"]["data_loader"]["max_prompt_length"] = 512
    issue = validator().validate_or_raise(ctx, scopes={"paths"}).issues[0]
    assert issue.severity is Severity.WARNING
    assert "effective_samples" not in issue.value


def test_only_index_header_is_read_not_token_payload(tmp_path, monkeypatch):
    index = make_index(tmp_path, count=7, boundaries=3)
    real_open = Path.open
    reads = []
    class Tracked:
        def __enter__(self):
            self.stream = real_open(index, "rb")
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, size):
            reads.append(size)
            return self.stream.read(size)
        def fileno(self):
            return self.stream.fileno()
    def open_index_only(path, *args, **kwargs):
        assert path == index
        return Tracked()
    monkeypatch.setattr(Path, "open", open_index_only)
    assert indexed_sample_count(index) == 7
    assert reads == [34]


@pytest.mark.parametrize("requested", [0, 1, 4, 9])
def test_actual_dataset_builder_uses_min_and_full_test_set(requested):
    path = Path(__file__).resolve().parents[4] / "aura/trainer/train_adapter/mindspeed_rl/utils/data_loader_validate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = copy.deepcopy(next(node for node in tree.body if getattr(node, "name", None) == "build_validate_test_dataset"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"np": SimpleNamespace(arange=lambda **kw: list(range(kw["start"], kw["stop"]))),
                 "get_packed_indexed_dataset": lambda **kw: SimpleNamespace(datasets={"input_ids": range(4)})}
    namespace["np"].int32 = int
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    valid, test = namespace["build_validate_test_dataset"](
        ["rl"], 1024, requested, lambda **kw: kw, extra_param=SimpleNamespace())
    assert (0 if valid is None else valid["num_samples"]) == min(requested, 4)
    assert test["num_samples"] == 4


def test_rule_registered_as_startup_io():
    rule = next(rule for rule in build_default_registry().get_rules({"paths"})
                if rule.rule_id == "paths.validation_samples")
    assert rule.io_bound


@pytest.mark.parametrize("path_value", [None, ""])
@pytest.mark.parametrize("requested", [0, 2])
def test_actual_loader_skips_validation_when_path_is_unset(path_value, requested):
    path = Path(__file__).resolve().parents[4] / "aura/trainer/train_adapter/mindspeed_rl/utils/default_train_dataloader.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in tree.body if getattr(node, "name", None) == "default_train_dataloader")
    validation_loader = Mock(side_effect=AssertionError("Validation data must not be loaded"))
    namespace = {
        "Any": object, "logger": Mock(), "PromptDataset": object,
        "build_train_valid_test_datasets": Mock(return_value=("train-data", None, None)),
        "PromptDataLoader": Mock(return_value=["train-example"]),
        "build_validate_test_dataloader": validation_loader,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(
        train_data_path="train", test_data_path=path_value, split="100,0,0", seq_length=1024,
        train_iters=1, global_batch_size=1, num_workers=0, seed=0, dataset_additional_keys=[], no_shuffle=True,
    )
    train, valid, test = namespace["default_train_dataloader"](config, requested, 0)
    assert list(train) == ["train-example"]
    assert valid is None and test is None
    validation_loader.assert_not_called()
