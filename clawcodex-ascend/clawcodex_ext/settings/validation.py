#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Settings validation matching TypeScript settings/validation.ts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.settings.constants import (
    VALID_EFFORT_VALUES,
    VALID_PERMISSION_MODES,
    VALID_SPINNER_VERB_MODES,
)
from clawcodex_ext.settings.types import SettingsSchema


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str
    value: Any = None


def _effective_default_permission_mode(settings: SettingsSchema) -> str | None:
    """Resolve the effective default permission mode from F-47 channels.

    Priority:

    1. ``settings.permissions.default_mode`` (preferred, structured).
    2. Top-level ``settings.permission_mode`` (back-compat reading channel
       kept for older binaries that wrote the mode outside the
       ``permissions`` block).
    """
    pc = settings.permissions
    default_mode = getattr(pc, "default_mode", None)
    if default_mode:
        return default_mode
    legacy = (settings.permission_mode or "").strip()
    return legacy or None


def validate_settings(settings: SettingsSchema) -> list[ValidationError]:
    """Validate a SettingsSchema, returning a list of errors (empty = valid)."""
    errors: list[ValidationError] = []

    # Effort
    if settings.effort and settings.effort not in VALID_EFFORT_VALUES:
        errors.append(
            ValidationError(
                field="effort",
                message=f"Invalid effort value: {settings.effort!r}. Must be one of {VALID_EFFORT_VALUES}",
                value=settings.effort,
            )
        )

    # F-47: permission default mode is read from ``permissions.default_mode``
    # first, then the top-level ``permission_mode`` (back-compat). Empty
    # strings on both are treated as "unset" and skip the enum check --
    # this avoids F-47 defaulting `permission_mode` to ``""`` from being
    # reported as an invalid mode.
    effective_default_mode = _effective_default_permission_mode(settings)
    if effective_default_mode is not None and effective_default_mode not in VALID_PERMISSION_MODES:
        permission_field = (
            "permissions.defaultMode" if getattr(settings.permissions, "default_mode", None) else "permission_mode"
        )
        errors.append(
            ValidationError(
                field=permission_field,
                message=f"Invalid default permission mode: {effective_default_mode!r}",
                value=effective_default_mode,
            )
        )

    # Output style names are free-form. Unknown names fall back at resolve
    # time, matching the upstream TS string schema and custom user styles.

    # Max width
    if settings.output_style.max_width < 40:
        errors.append(
            ValidationError(
                field="output_style.max_width",
                message="max_width must be >= 40",
                value=settings.output_style.max_width,
            )
        )

    # Spinner verbs (optional; only the merge mode is constrained)
    if settings.spinner_verbs is not None and (settings.spinner_verbs.mode not in VALID_SPINNER_VERB_MODES):
        errors.append(
            ValidationError(
                field="spinner_verbs.mode",
                message=f"Invalid spinner verbs mode: {settings.spinner_verbs.mode!r}. "
                f"Must be one of {VALID_SPINNER_VERB_MODES}",
                value=settings.spinner_verbs.mode,
            )
        )

    # Max turns (0 = unlimited, otherwise must be positive)
    if settings.max_turns < 0:
        errors.append(
            ValidationError(
                field="max_turns",
                message="max_turns must be >= 0",
                value=settings.max_turns,
            )
        )

    # Max cost
    if settings.max_cost_usd < 0:
        errors.append(
            ValidationError(
                field="max_cost_usd",
                message="max_cost_usd must be >= 0",
                value=settings.max_cost_usd,
            )
        )

    # Session retention
    if settings.session_retention_days < 1:
        errors.append(
            ValidationError(
                field="session_retention_days",
                message="session_retention_days must be >= 1",
                value=settings.session_retention_days,
            )
        )

    # Compact threshold
    if settings.compact.threshold_tokens < 1000:
        errors.append(
            ValidationError(
                field="compact.threshold_tokens",
                message="compact threshold must be >= 1000",
                value=settings.compact.threshold_tokens,
            )
        )

    # Hooks timeout
    if settings.hooks.timeout_ms < 1000:
        errors.append(
            ValidationError(
                field="hooks.timeout_ms",
                message="hooks timeout must be >= 1000ms",
                value=settings.hooks.timeout_ms,
            )
        )

    # F-47: permissions.rules is a dict[str, list[str]] (allow/deny/ask).
    # The legacy list[PermissionRule] path is gone (Sub-H); rule strings
    # are kept verbatim on disk and validated as non-empty here.
    permissions = settings.permissions
    legacy_rules = permissions if isinstance(permissions, list) else getattr(permissions, "legacy_rules", [])
    if legacy_rules:
        from src.settings.permission_validation import validate_permission_rules

        for message in validate_permission_rules(list(legacy_rules)):
            field, _, detail = message.partition(": ")
            errors.append(
                ValidationError(
                    field=field,
                    message=detail or message,
                )
            )

    rules = {} if isinstance(permissions, list) else permissions.rules
    for behavior in ("allow", "deny", "ask"):
        bucket = rules.get(behavior, [])
        if not isinstance(bucket, list):
            errors.append(
                ValidationError(
                    field=f"permissions.rules.{behavior}",
                    message=f"permissions.rules.{behavior} must be a list",
                    value=bucket,
                )
            )
            continue
        for j, rule_str in enumerate(bucket):
            if not isinstance(rule_str, str) or not rule_str.strip():
                errors.append(
                    ValidationError(
                        field=f"permissions.rules.{behavior}[{j}]",
                        message="Rule must be a non-empty string",
                        value=rule_str,
                    )
                )

    return errors
