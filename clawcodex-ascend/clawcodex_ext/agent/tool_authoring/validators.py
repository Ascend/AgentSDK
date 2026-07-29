"""Validators for AgentToolSpec — enforces security constraints on call_impl."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec


# ---------------------------------------------------------------------------
# Whitelists
# ---------------------------------------------------------------------------

ALLOWED_BASH_COMMANDS: frozenset[str] = frozenset(
    {
        "git",
        "gh",
        "glab",
        "curl",
        "wget",
        "kubectl",
        "docker",
        "npm",
        "pip",
        "python",
        "python3",
    }
)

ALLOWED_HTTP_METHODS: frozenset[str] = frozenset(
    {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    }
)

# Python functions registered by the host application for agent use.
# Maps name → callable. Extend via ``register_python_function()``.
_PYTHON_FUNCTION_REGISTRY: dict[str, callable] = {}


def register_python_function(name: str, fn: callable) -> None:
    """Register a callable so agents can reference it by name in a tool spec."""
    _PYTHON_FUNCTION_REGISTRY[name] = fn


def list_python_functions() -> frozenset[str]:
    """Return the set of registered python function names."""
    return frozenset(_PYTHON_FUNCTION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when an AgentToolSpec fails security validation."""

    pass


def validate_spec(spec: AgentToolSpec) -> None:
    """Validate an AgentToolSpec in full.

    Raises:
        ValidationError: If the spec violates any security constraint.
    """
    _validate_name(spec.name)
    _validate_call_impl(spec.call_type, spec.call_impl)
    if spec.output_schema is not None and not isinstance(spec.output_schema, dict):
        raise ValidationError("output_schema must be a JSON schema object")


_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise ValidationError(
            f"Invalid tool name '{name}'. Must be lowercase, start with a letter, "
            "and contain only letters, numbers, hyphens, and underscores."
        )


def _validate_call_impl(call_type: str, call_impl: str | dict) -> None:
    if call_type == "bash":
        _validate_bash_impl(call_impl)
    elif call_type == "http":
        _validate_http_impl(call_impl)
    elif call_type == "python":
        _validate_python_impl(call_impl)
    elif call_type == "workflow":
        _validate_workflow_impl(call_impl)
    else:
        raise ValidationError(f"Unknown call_type: {call_type}")


def _validate_bash_impl(call_impl: str | dict) -> None:
    if not isinstance(call_impl, str):
        raise ValidationError("bash call_impl must be a string template")
    if not call_impl.strip():
        raise ValidationError("bash call_impl cannot be empty")

    # Extract the leading command word (before any space or redirection).
    head = call_impl.split()[0] if call_impl.split() else ""
    if head not in ALLOWED_BASH_COMMANDS:
        raise ValidationError(
            f"bash command '{head}' is not in the allowlist. Allowed: {', '.join(sorted(ALLOWED_BASH_COMMANDS))}"
        )


def _validate_http_impl(call_impl: str | dict) -> None:
    if not isinstance(call_impl, dict):
        raise ValidationError("http call_impl must be a dict with 'method' and 'url'")

    method = call_impl.get("method", "").upper()
    if method not in ALLOWED_HTTP_METHODS:
        raise ValidationError(
            f"HTTP method '{method}' is not in the allowlist. Allowed: {', '.join(sorted(ALLOWED_HTTP_METHODS))}"
        )

    url = call_impl.get("url", "")
    if not isinstance(url, str) or not url.startswith("http"):
        raise ValidationError("http call_impl 'url' must be a valid http:// or https:// URL")


def _validate_python_impl(call_impl: str | dict) -> None:
    if not isinstance(call_impl, str):
        raise ValidationError("python call_impl must be a string function name")
    name = call_impl.strip()
    if not name:
        raise ValidationError("python call_impl function name cannot be empty")
    if name not in _PYTHON_FUNCTION_REGISTRY:
        available = ", ".join(sorted(_PYTHON_FUNCTION_REGISTRY.keys())) or "(none registered)"
        raise ValidationError(f"python function '{name}' is not registered. Available: {available}")


def _validate_workflow_impl(call_impl: str | dict) -> None:
    if not isinstance(call_impl, dict):
        raise ValidationError("workflow call_impl must be a dict catalog reference")
    keys = [key for key in ("catalog_id", "manifest") if call_impl.get(key)]
    if len(keys) != 1:
        raise ValidationError("workflow call_impl must contain exactly one catalog_id or manifest")
    if "catalog_id" in keys:
        catalog_id = call_impl["catalog_id"]
        if not isinstance(catalog_id, str) or not re.fullmatch(
            r"(?:builtin|bundle|session):[A-Za-z0-9][A-Za-z0-9._:-]*",
            catalog_id,
        ):
            raise ValidationError("workflow catalog_id must use builtin:, bundle:, or session: scope")
        return
    manifest = call_impl["manifest"]
    if not isinstance(manifest, str) or not manifest.strip():
        raise ValidationError("workflow manifest must be a non-empty relative path")
    if manifest.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", manifest):
        raise ValidationError("workflow manifest must be relative to its bundle")
    if any(part == ".." for part in re.split(r"[\\/]", manifest)):
        raise ValidationError("workflow manifest cannot escape its bundle")
