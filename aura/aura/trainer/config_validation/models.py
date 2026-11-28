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

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class Severity(str, Enum):
    """Severity assigned to a configuration validation result."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """A single structured issue produced by a validation rule."""

    rule_id: str
    severity: Severity
    path: str
    value: Any
    expected: str
    source: str = "unknown"
    message: str = ""
    suggestion: str = ""
    stages: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert this issue to a JSON-ready dictionary."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "path": self.path,
            "value": self.value,
            "expected": self.expected,
            "source": self.source,
            "message": self.message,
            "suggestion": self.suggestion,
            "stages": list(self.stages),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationIssue":
        """Restore an issue from a previously written JSON report."""
        return cls(
            rule_id=payload["rule_id"],
            severity=Severity(payload["severity"]),
            path=payload["path"],
            value=payload.get("value"),
            expected=payload.get("expected", ""),
            source=payload.get("source", "unknown"),
            message=payload.get("message", ""),
            suggestion=payload.get("suggestion", ""),
            stages=tuple(payload.get("stages", ())),
        )

    def with_stage(self, stage: str) -> "ValidationIssue":
        """Return this issue with one additional origin stage."""
        return replace(self, stages=_ordered_union(self.stages, (stage,)))


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated output from one validator execution."""

    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    duration_ms: float = 0.0
    scopes: tuple[str, ...] = field(default_factory=tuple)
    config: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = "1.0"
    stages: tuple[str, ...] = field(default_factory=tuple)
    run_id: str | None = None

    @property
    def error_count(self) -> int:
        return self._count(Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return self._count(Severity.WARNING)

    @property
    def info_count(self) -> int:
        return self._count(Severity.INFO)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def valid(self) -> bool:
        return not self.has_errors

    def _count(self, severity: Severity) -> int:
        return sum(issue.severity is severity for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Convert the full report to a JSON-ready dictionary."""
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "duration_ms": round(self.duration_ms, 3),
            "scopes": list(self.scopes),
            "stages": list(self.stages),
            "run_id": self.run_id,
            "config": self.config,
            "sources": dict(self.sources),
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": {
                "error": self.error_count,
                "warning": self.warning_count,
                "info": self.info_count,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationReport":
        """Restore a report from JSON for cross-process stage aggregation."""
        issues = payload.get("issues", ())
        if not isinstance(issues, list):
            raise TypeError("validation report issues must be a list")
        return cls(
            issues=tuple(ValidationIssue.from_dict(issue) for issue in issues),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            scopes=tuple(payload.get("scopes", ())),
            config=payload.get("config", {}),
            sources=payload.get("sources", {}),
            schema_version=payload.get("schema_version", "1.0"),
            stages=tuple(payload.get("stages", ())),
            run_id=payload.get("run_id"),
        )

    def with_stage(self, stage: str, run_id: str) -> "ValidationReport":
        """Attach execution identity to a single-stage validation result."""
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        return replace(
            self,
            issues=tuple(issue.with_stage(stage) for issue in self.issues),
            stages=_ordered_union(self.stages, (stage,)),
            run_id=run_id,
        )


def merge_validation_reports(*reports: ValidationReport) -> ValidationReport:
    """Merge stage reports deterministically and deduplicate equivalent issues."""
    if not reports:
        raise ValueError("at least one validation report is required")

    run_ids = {report.run_id for report in reports}
    if len(run_ids) != 1:
        raise ValueError("validation reports must have the same run_id")

    issues_by_key: dict[tuple[str, str, Severity, str], ValidationIssue] = {}
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    scopes: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()

    for report in reports:
        scopes = _ordered_union(scopes, report.scopes)
        stages = _ordered_union(stages, report.stages)
        sources.update(report.sources)
        config = _merge_config_snapshots(config, report.config)
        for issue in report.issues:
            key = (issue.rule_id, issue.path, issue.severity, issue.message)
            previous = issues_by_key.get(key)
            if previous is None:
                issues_by_key[key] = issue
            else:
                issues_by_key[key] = replace(
                    previous,
                    stages=_ordered_union(previous.stages, issue.stages),
                )

    severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    issues = tuple(
        sorted(
            issues_by_key.values(),
            key=lambda issue: (severity_order[issue.severity], issue.rule_id, issue.path),
        )
    )
    return ValidationReport(
        issues=issues,
        duration_ms=sum(report.duration_ms for report in reports),
        scopes=tuple(sorted(scopes)),
        config=config,
        sources=sources,
        schema_version=reports[-1].schema_version,
        stages=stages,
        run_id=reports[-1].run_id,
    )


def _ordered_union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*first, *second)))


def _merge_config_snapshots(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(first)
    for key, value in second.items():
        if value is not None or key not in merged:
            merged[key] = value
    return merged


class ConfigValidationError(RuntimeError):
    """Raised after all rules finish when strict validation has errors."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(
            "Configuration validation failed: "
            f"{report.error_count} error(s), "
            f"{report.warning_count} warning(s), "
            f"{report.info_count} info item(s)."
        )
