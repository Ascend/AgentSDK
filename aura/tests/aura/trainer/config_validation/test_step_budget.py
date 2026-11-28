"""Step/epoch semantics: CPU tests without Ray, model weights or NPU imports."""
import pytest
from omegaconf import OmegaConf

from aura.trainer.config_validation import ValidationContext
from aura.trainer.config_validation.models import ConfigValidationError, Severity
from aura.trainer.config_validation.registry import RuleRegistry
from aura.trainer.config_validation.rules.step_budget import OneStepOffStepBudgetRule
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
    registry.register(OneStepOffStepBudgetRule())
    return registry

def validate(total=3, epochs=1000, data_iters=3, rollout_iters=3, declared=3,
             trainer_steps=None, mode="one_step_off"):
    train = OmegaConf.create({
        "total_train_steps": total,
        "trainer": {"total_epochs": epochs, "total_training_steps": trainer_steps},
        "extras": {"data_loader": {"train_iters": data_iters}},
        "rollout": {"total_rollout_steps": declared},
    })
    registry = RuleRegistry()
    registry.register(OneStepOffStepBudgetRule())
    return ConfigValidator(registry).validate(ValidationContext(
        train_config=train, rollout_config={"train_iters": rollout_iters}, work_mode=mode,
        source_map={"train_config.trainer.total_epochs": "user_explicit"},
    ))


@pytest.mark.parametrize("epochs", [1, 3, 1000])
def test_epoch_count_is_not_a_step_limit(epochs):
    report = validate(epochs=epochs)
    assert report.valid
    assert len(report.issues) == 1
    assert report.issues[0].severity == Severity.INFO
    assert report.issues[0].source == "user_explicit"


@pytest.mark.parametrize("field", ["total", "epochs", "data_iters", "rollout_iters", "declared"])
@pytest.mark.parametrize("value", [0, -1, True, "3", 1.5, None])
def test_invalid_step_or_epoch_values(field, value):
    assert validate(**{field: value}).has_errors


@pytest.mark.parametrize("field", ["data_iters", "rollout_iters", "trainer_steps"])
@pytest.mark.parametrize("value", [2, 4])
def test_effective_step_conflicts_are_errors(field, value):
    report = validate(**{field: value})
    assert report.has_errors
    assert any(issue.severity == Severity.ERROR and "equal total_train_steps" in issue.expected
               for issue in report.issues)


def test_declared_rollout_steps_are_a_warning_not_a_loop_limit():
    report = validate(declared=100)
    assert report.valid
    assert any(issue.severity == Severity.WARNING for issue in report.issues)


def test_hybrid_mode_is_not_subject_to_step_driven_policy():
    assert validate(total=0, epochs=0, mode="hybrid").issues == ()


def test_rule_registered_in_startup_linkage_scope():
    ids = {rule.rule_id for rule in build_default_registry().get_rules({"linkage"})}
    assert OneStepOffStepBudgetRule.rule_id in ids


def test_strict_validation_rejects_conflict():
    registry = RuleRegistry()
    registry.register(OneStepOffStepBudgetRule())
    context = ValidationContext(train_config={
        "total_train_steps": 3, "trainer": {"total_epochs": 1000},
        "extras": {"data_loader": {"train_iters": 2}},
        "rollout": {"total_rollout_steps": 3},
    }, work_mode="one_step_off")
    with pytest.raises(ConfigValidationError):
        ConfigValidator(registry).validate_or_raise(context)


def test_missing_budgets_are_reported():
    rule = OneStepOffStepBudgetRule()
    issues = rule.validate(ValidationContext(
        train_config={}, rollout_config={}, work_mode="one_step_off",
    ))
    paths = {issue.path for issue in issues if issue.severity == Severity.ERROR}
    assert paths == {
        "train_config.total_train_steps", "train_config.trainer.total_epochs",
        "train_config.extras.data_loader.train_iters", "rollout_config.train_iters",
        "train_config.rollout.total_rollout_steps",
    }
