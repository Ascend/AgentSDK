#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Stage contract validator.

Machine-verifiable DoD checks against stage output.
Built-in types: file_exists, file_size, regex, json_schema, line_count, llm_judge, custom
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result."""

    passed: bool
    validator_type: str
    message: str = ""
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)  # Backward-compatible extra fields


class ContractValidator:
    """Stage contract validator registry.

    Supports both sync and async validators; ``workspace_dir`` and ``llm_client`` may be injected at construction time,
    for use by the ``custom`` and ``llm_judge`` validators.
    """

    def __init__(
        self,
        workspace_dir: str | Path = "",
        llm_client: Any = None,
    ) -> None:
        self._workspace_dir = str(workspace_dir)
        self._llm_client = llm_client
        self._validators: dict[str, Callable[..., Any]] = {}
        self._async_validators: set[str] = {"llm_judge", "custom"}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register a built-in validator."""
        self._validators["file_exists"] = _validate_file_exists
        self._validators["file_size"] = _validate_file_size
        self._validators["regex"] = _validate_regex
        self._validators["line_count"] = _validate_line_count
        self._validators["json_schema"] = _validate_json_schema

    def register(self, name: str, fn: Callable[..., Any], is_async: bool = False) -> None:
        """Register a custom validator."""
        self._validators[name] = fn
        if is_async:
            self._async_validators.add(name)

    async def validate(self, spec: dict[str, Any]) -> ValidationResult:
        """Run a single validator (async-capable)."""
        validator_type = spec.get("type", "")

        if validator_type == "custom":
            from .custom import validate_custom

            return await validate_custom(spec, workspace_dir=self._workspace_dir)

        if validator_type == "llm_judge":
            from .llm_judge import validate_llm_judge

            return await validate_llm_judge(spec, llm_client=self._llm_client)

        fn = self._validators.get(validator_type)
        if fn is None:
            return ValidationResult(
                passed=False,
                validator_type=validator_type,
                message=f"Unknown validator type: {validator_type}",
            )

        try:
            kwargs = {k: v for k, v in spec.items() if k != "type"}
            result = fn(**kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        except Exception as exc:
            return ValidationResult(
                passed=False,
                validator_type=validator_type,
                message=f"Validator error: {exc}",
            )

    async def validate_all(self, specs: list[dict[str, Any]]) -> list[ValidationResult]:
        """Run all validators (async-capable)."""
        results = []
        for spec in specs:
            results.append(await self.validate(spec))
        return results

    def validate_sync(self, spec: dict[str, Any]) -> ValidationResult:
        """Run a single validator synchronously.

        Note: if the current thread already has a running event loop, use the async ``validate()`` interface instead,
        otherwise coroutines may be blocked. This method is only usable when no event loop is running.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.validate(spec))
        raise RuntimeError("validate_sync cannot be called from a running event loop; use validate() instead")


# -- Built-in validator implementations --─────────────────────────────────────────────────


def _resolve_path(path: str) -> Path:
    """Resolve path with ~ expansion."""
    return Path(path).expanduser().resolve()


def _validate_file_exists(path: str = "", **kwargs: Any) -> ValidationResult:
    """Validate that a file exists."""
    if not path:
        return ValidationResult(passed=False, validator_type="file_exists", message="No path specified")
    p = _resolve_path(path)
    if p.exists():
        return ValidationResult(passed=True, validator_type="file_exists", message=f"{path} exists")
    return ValidationResult(passed=False, validator_type="file_exists", message=f"{path} not found")


def _validate_file_size(
    path: str = "", min_bytes: int = 0, max_bytes: int | None = None, **kwargs: Any
) -> ValidationResult:
    """Validate file size."""
    p = _resolve_path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return ValidationResult(passed=False, validator_type="file_size", message=f"{path} not found")

    if size < min_bytes:
        return ValidationResult(
            passed=False,
            validator_type="file_size",
            message=f"{path}: {size} bytes < min {min_bytes} bytes",
            details={"size": size, "min_bytes": min_bytes},
        )
    if max_bytes is not None and size > max_bytes:
        return ValidationResult(
            passed=False,
            validator_type="file_size",
            message=f"{path}: {size} bytes > max {max_bytes} bytes",
            details={"size": size, "max_bytes": max_bytes},
        )
    return ValidationResult(
        passed=True,
        validator_type="file_size",
        message=f"{path}: {size} bytes",
        details={"size": size},
    )


def _validate_regex(path: str = "", pattern: str = "", min_matches: int = 1, **kwargs: Any) -> ValidationResult:
    """Regex match validation."""
    p = _resolve_path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ValidationResult(passed=False, validator_type="regex", message=f"{path} not found")
    except Exception as exc:
        return ValidationResult(passed=False, validator_type="regex", message=f"Read error: {exc}")

    try:
        matches = re.findall(pattern, content)
    except re.error as exc:
        return ValidationResult(passed=False, validator_type="regex", message=f"Invalid pattern: {exc}")

    if len(matches) < min_matches:
        return ValidationResult(
            passed=False,
            validator_type="regex",
            message=f"Pattern '{pattern}' matched {len(matches)} times, min {min_matches}",
            details={"match_count": len(matches), "min_matches": min_matches},
        )
    return ValidationResult(
        passed=True,
        validator_type="regex",
        message=f"Pattern '{pattern}' matched {len(matches)} times",
        details={"match_count": len(matches)},
    )


def _validate_line_count(
    path: str = "", min_lines: int = 1, max_lines: int | None = None, **kwargs: Any
) -> ValidationResult:
    """Line count validation."""
    p = _resolve_path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ValidationResult(passed=False, validator_type="line_count", message=f"{path} not found")

    count = len(content.splitlines())
    if count < min_lines:
        return ValidationResult(
            passed=False,
            validator_type="line_count",
            message=f"{path}: {count} lines < min {min_lines}",
            details={"line_count": count, "min_lines": min_lines},
        )
    if max_lines is not None and count > max_lines:
        return ValidationResult(
            passed=False,
            validator_type="line_count",
            message=f"{path}: {count} lines > max {max_lines}",
            details={"line_count": count, "max_lines": max_lines},
        )
    return ValidationResult(
        passed=True,
        validator_type="line_count",
        message=f"{path}: {count} lines",
        details={"line_count": count},
    )


def _validate_json_schema(path: str = "", schema: dict[str, Any] | None = None, **kwargs: Any) -> ValidationResult:
    """JSON Schema validation."""
    p = _resolve_path(path)
    try:
        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
    except FileNotFoundError:
        return ValidationResult(passed=False, validator_type="json_schema", message=f"{path} not found")
    except json.JSONDecodeError as exc:
        return ValidationResult(passed=False, validator_type="json_schema", message=f"Invalid JSON: {exc}")

    if schema is None:
        return ValidationResult(passed=True, validator_type="json_schema", message="No schema provided, assumed valid")

    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
        return ValidationResult(passed=True, validator_type="json_schema", message="JSON schema valid")
    except ImportError:
        return ValidationResult(passed=False, validator_type="json_schema", message="jsonschema library not installed")
    except jsonschema.ValidationError as exc:
        return ValidationResult(passed=False, validator_type="json_schema", message=f"Schema violation: {exc.message}")
