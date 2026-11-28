"""Startup-only validation-set capacity checks without loading tokens or accelerators."""
import os
import struct
from pathlib import Path

from aura.trainer.config_validation.access import get_config_value
from aura.trainer.config_validation.models import Severity, ValidationIssue
from aura.trainer.config_validation.registry import ValidationRule


class UnsupportedIndexFormat(ValueError):
    pass


def indexed_sample_count(path):
    """Read MMIDIDX v1 sequence count, NOT document-boundary count or token count."""
    with Path(path).open("rb") as stream:
        header = stream.read(34)
        if len(header) < 9:
            raise ValueError("truncated index header")
        if header[:9] != b"MMIDIDX\x00\x00":
            raise UnsupportedIndexFormat("unrecognized index magic")
        if len(header) != 34:
            raise ValueError("truncated MMIDIDX header")
        version, dtype, sequences, boundaries = struct.unpack("<QBQQ", header[9:])
        if version != 1:
            raise UnsupportedIndexFormat("unsupported index version: %s" % version)
        if dtype not in range(1, 9) or boundaries < 1:
            raise ValueError("invalid index dtype or document-boundary count")
        if os.fstat(stream.fileno()).st_size < 34 + 12 * sequences + 8 * boundaries:
            raise ValueError("index arrays are truncated")
        return sequences


class ValidationSamplesRule(ValidationRule):
    rule_id = "paths.validation_samples"
    scopes = frozenset({"paths"})
    io_bound = True

    def is_enabled(self, context):
        requested = get_config_value(context.train_config, "extras.validate_num_samples")
        path = get_config_value(context.train_config, "extras.data_loader.test_data_path")
        supplied = isinstance(path, (str, os.PathLike)) and bool(os.fspath(path).strip())
        return (context.work_mode in {"one_step_off", "fully_async"} and
                isinstance(requested, int) and not isinstance(requested, bool) and requested >= 0 and
                supplied)

    def issue(self, context, severity, value, message):
        return ValidationIssue(
            rule_id=self.rule_id, severity=severity,
            path="train_config.extras.validate_num_samples", value=value,
            expected="positive validation sample count; runtime selects min(requested, available)",
            source=context.source_for("extras.validate_num_samples"), message=message,
            suggestion="Use a readable packed validation prefix and a positive count within its capacity.",
        )

    def validate(self, context):
        requested = get_config_value(context.train_config, "extras.validate_num_samples")
        if requested == 0:
            return [self.issue(context, Severity.ERROR, requested,
                               "Validation data is supplied, but zero samples produces no validation dataset.")]
        raw = get_config_value(context.train_config, "extras.data_loader.test_data_path")
        if not isinstance(raw, (str, os.PathLike)) or not os.fspath(raw).strip():
            return []  # paths.test_data reports invalid explicitly supplied paths.
        prefix = Path(os.path.expandvars(os.path.expanduser(os.fspath(raw))))
        index = prefix.parent / (prefix.name + "_packed_input_ids_document.idx")
        if prefix.exists():
            return [self.issue(context, Severity.WARNING, requested,
                               "Capacity is unverified: this rule supports packed MMIDIDX prefixes, not raw files/directories.")]
        try:
            available = indexed_sample_count(index)
        except UnsupportedIndexFormat as error:
            return [self.issue(context, Severity.WARNING, requested,
                               "Capacity is unverified: %s." % error)]
        except (OSError, ValueError) as error:
            return [self.issue(context, Severity.ERROR, requested,
                               "Cannot read validation index %s: %s." % (index, error))]
        value = {"requested": requested, "indexed_samples": available, "index": str(index)}
        if available == 0:
            return [self.issue(context, Severity.ERROR, value, "The indexed validation dataset has no samples.")]
        # Templates use data.*; the packed loader also accepts an extras override.
        if any(get_config_value(context.train_config, path, None) is not None for path in (
            "data.max_prompt_length", "extras.data_loader.max_prompt_length",
        )):
            return [self.issue(context, Severity.WARNING, value,
                               "Capacity after max_prompt_length filtering is unverified; indexed_samples is only the raw count.")]
        value["effective_samples"] = min(requested, available)
        severity = Severity.WARNING if requested > available else Severity.INFO
        return [self.issue(context, severity, value,
                           "Validation selection uses min(requested, available); the test dataset still uses all available samples.")]
