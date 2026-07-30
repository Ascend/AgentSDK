"""Acceptance criteria verifier.

Each :class:`AcceptanceCriteria` carries a ``kind`` and a ``target`` that
the verifier dispatches to a registered check function. The default
registry supports:

* ``FILE_EXISTS`` — ``target`` is a filesystem path.
* ``FILE_CONTAINS`` — ``target`` is a path; ``args`` must include
  ``substring``.
* ``PYTHON_PREDICATE`` — ``target`` is a Python expression evaluated
  with a tiny, locked-down namespace (``len``, ``min``, ``max``,
  ``sum``, ``any``, ``all``, ``int``, ``str``, ``bool`` plus anything
  the caller injects via ``registry``). Import statements and dunder
  attribute access are rejected.
* ``SHELL_COMMAND`` — ``target`` is a shell command run via
  :mod:`subprocess` with a hard timeout. The check passes when the
  command exits 0. Commands containing ``;``, ``&&``, ``|``, or other
  shell metacharacters are still run via ``shell=False`` (tokenized
  via :func:`shlex.split`) to avoid shell-injection.
* ``CUSTOM`` — the verifier looks up ``target`` in the registry as a
  callable and invokes it with ``(target, args)``.

The verifier is intentionally synchronous and side-effect-isolated; the
executor or a CLI layer can decide when to run it.
"""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import (
    UnknownCheckKindError,
    UnsafeCheckExpressionError,
    VerificationCheckFailedError,
)
from .models import AcceptanceCriteria, CheckKind, Plan, Step

DEFAULT_SHELL_TIMEOUT_SECONDS = 30.0


CheckFn = Callable[[str, dict[str, Any]], "CheckResult"]
PredicateFn = Callable[[Any], bool]


@dataclass
class CheckResult:
    passed: bool
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "details": self.details,
            "evidence": dict(self.evidence),
        }


# Safe globals for PYTHON_PREDICATE expressions. Anything that can mutate
# state or escape the sandbox is excluded.
_PREDICATE_GLOBALS: dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "any": any,
    "all": all,
    "abs": abs,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "range": range,
    "True": True,
    "False": False,
    "None": None,
}


def _validate_predicate_expression(expr: str) -> ast.Expression:
    """Parse and statically validate a PYTHON_PREDICATE expression.

    Rejects:
      * Import / import-from statements.
      * Dunder attribute access (``__class__``, ``__bases__``, etc.).
      * Calls to attribute-named builtins (no method calls on objects).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeCheckExpressionError(f"PYTHON_PREDICATE expression is not valid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCheckExpressionError("PYTHON_PREDICATE expression must not contain import statements")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCheckExpressionError("PYTHON_PREDICATE expression must not access dunder attributes")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Allow attribute-name calls only if the attribute does not
            # start with an underscore. This blocks ``x.__class__`` style
            # escape hatches and similarly suspicious lookups.
            if node.func.attr.startswith("_"):
                raise UnsafeCheckExpressionError("PYTHON_PREDICATE expression must not call underscored methods")
    return tree


def _check_file_exists(target: str, args: dict[str, Any]) -> CheckResult:
    p = Path(target)
    if not p.is_absolute():
        return CheckResult(False, details=f"path is not absolute: {target!r}")
    if not p.exists():
        return CheckResult(False, details=f"path does not exist: {target!r}")
    if p.is_dir():
        return CheckResult(True, details=f"directory exists: {target!r}")
    return CheckResult(True, details=f"file exists: {target!r}")


def _check_file_contains(target: str, args: dict[str, Any]) -> CheckResult:
    substring = args.get("substring")
    if not isinstance(substring, str) or not substring:
        return CheckResult(False, details="args.substring must be a non-empty string")
    p = Path(target)
    if not p.is_absolute():
        return CheckResult(False, details=f"path is not absolute: {target!r}")
    if not p.is_file():
        return CheckResult(False, details=f"not a regular file: {target!r}")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return CheckResult(False, details=f"read error: {exc}")
    if substring in content:
        return CheckResult(True, details="substring present", evidence={"len": len(content)})
    return CheckResult(
        False,
        details="substring not found",
        evidence={"len": len(content), "substring": substring},
    )


def _check_shell_command(target: str, args: dict[str, Any]) -> CheckResult:
    timeout = float(args.get("timeout", DEFAULT_SHELL_TIMEOUT_SECONDS))
    if timeout <= 0 or timeout > 600:
        return CheckResult(False, details=f"unsafe timeout: {timeout!r}")
    try:
        argv = shlex.split(target, posix=os.name != "nt")
    except ValueError as exc:
        return CheckResult(False, details=f"command parse error: {exc}")
    if not argv:
        return CheckResult(False, details="empty command")
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            False,
            details=f"command timed out after {timeout}s",
            evidence={"stdout": exc.stdout.decode("utf-8", "replace") if exc.stdout else ""},
        )
    except FileNotFoundError as exc:
        return CheckResult(False, details=f"command not found: {exc}")
    except OSError as exc:
        return CheckResult(False, details=f"command failed: {exc}")
    if completed.returncode != 0:
        return CheckResult(
            False,
            details=f"command exited {completed.returncode}",
            evidence={
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", "replace")[-2000:],
            },
        )
    return CheckResult(
        True,
        details="command exited 0",
        evidence={"stdout_tail": completed.stdout.decode("utf-8", "replace")[-2000:]},
    )


def _check_python_predicate(target: str, args: dict[str, Any]) -> CheckResult:
    tree = _validate_predicate_expression(target)
    context = args.get("context", {})
    if not isinstance(context, dict):
        return CheckResult(False, details="args.context must be a dict")
    namespace = dict(_PREDICATE_GLOBALS)
    namespace.update(context)
    try:
        value = eval(  # nosec B307  # pylint: disable=eval-used
            compile(tree, "<ultraplan-predicate>", "eval"),
            {"__builtins__": {}},
            namespace,
        )
    except Exception as exc:
        return CheckResult(False, details=f"predicate raised: {exc}")
    if bool(value):
        return CheckResult(True, details="predicate returned truthy")
    return CheckResult(False, details="predicate returned falsy", evidence={"value": repr(value)})


class AcceptanceVerifier:
    """Run acceptance criteria for a :class:`Plan`."""

    def __init__(
        self,
        plan: Plan,
        *,
        custom_checks: dict[str, CheckFn] | None = None,
        predicate_context: dict[str, Any] | None = None,
        default_timeout: float = DEFAULT_SHELL_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(plan, Plan):
            raise TypeError("AcceptanceVerifier requires a Plan instance")
        self._plan = plan
        self._lock = threading.RLock()
        self._default_timeout = default_timeout
        self._predicate_context = dict(predicate_context or {})
        self._registry: dict[CheckKind, CheckFn] = {
            CheckKind.FILE_EXISTS: _check_file_exists,
            CheckKind.FILE_CONTAINS: _check_file_contains,
            CheckKind.PYTHON_PREDICATE: self._run_python_predicate,
            CheckKind.SHELL_COMMAND: self._run_shell_command,
            CheckKind.CUSTOM: self._run_custom,
        }
        # Custom checks can override CUSTOM kind only, or add a new
        # stringly-keyed lookup. The CUSTOM dispatch accepts either the
        # built-in ``CheckKind.CUSTOM`` (target=callable name) or a
        # name in ``custom_checks`` to be looked up directly.
        self._custom_checks: dict[str, CheckFn] = dict(custom_checks or {})

    def register_custom_check(self, name: str, fn: CheckFn) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("custom check name must be a non-empty string")
        if not callable(fn):
            raise TypeError("custom check must be callable")
        with self._lock:
            self._custom_checks[name] = fn

    def verify_criterion(self, criterion: AcceptanceCriteria) -> CheckResult:
        if not isinstance(criterion, AcceptanceCriteria):
            raise TypeError("verify_criterion expects an AcceptanceCriteria")
        with self._lock:
            try:
                handler = self._registry.get(criterion.kind)
            except Exception as exc:
                return CheckResult(False, details=f"invalid kind: {exc}")
            if handler is None:
                return CheckResult(
                    False,
                    details=f"no check handler registered for kind {criterion.kind!r}",
                )
            try:
                return handler(criterion.target, dict(criterion.args))
            except UnsafeCheckExpressionError as exc:
                return CheckResult(False, details=str(exc))
            except UnknownCheckKindError as exc:
                return CheckResult(False, details=str(exc))

    def _run_python_predicate(self, target: str, args: dict[str, Any]) -> CheckResult:
        merged = dict(args)
        merged.setdefault("context", {})
        if isinstance(merged["context"], dict):
            merged["context"] = {**self._predicate_context, **merged["context"]}
        return _check_python_predicate(target, merged)

    def _run_shell_command(self, target: str, args: dict[str, Any]) -> CheckResult:
        merged = dict(args)
        merged.setdefault("timeout", self._default_timeout)
        return _check_shell_command(target, merged)

    def _run_custom(self, target: str, args: dict[str, Any]) -> CheckResult:
        fn = self._custom_checks.get(target)
        if fn is None:
            raise UnknownCheckKindError(f"no custom check registered under name {target!r}")
        return fn(target, args)

    # ------------------------------------------------------------------
    # Plan-level verification
    # ------------------------------------------------------------------

    def verify_step(self, step_id: str) -> dict[str, CheckResult]:
        result = self._plan.find_step(step_id)
        if result is None:
            raise ValueError(f"step {step_id!r} not found in plan {self._plan.id!r}")
        _, step = result
        return self._verify_step(step)

    def _verify_step(self, step: Step) -> dict[str, CheckResult]:
        with self._lock:
            out: dict[str, CheckResult] = {}
            for criterion in step.criteria:
                try:
                    out[criterion.id] = self.verify_criterion(criterion)
                except (
                    UnknownCheckKindError,
                    UnsafeCheckExpressionError,
                ) as exc:
                    out[criterion.id] = CheckResult(False, details=str(exc))
            return out

    def verify_sub_plan(self, sub_plan_id: str) -> dict[str, dict[str, CheckResult]]:
        sp = self._plan.find_sub_plan(sub_plan_id)
        if sp is None:
            raise ValueError(f"sub_plan {sub_plan_id!r} not found in plan {self._plan.id!r}")
        with self._lock:
            return {step.id: self._verify_step(step) for step in sp.steps}

    def verify_plan(self) -> dict[str, dict[str, CheckResult]]:
        with self._lock:
            return {sp.id: {step.id: self._verify_step(step) for step in sp.steps} for sp in self._plan.sub_plans}

    def is_plan_passing(self) -> bool:
        """Return True iff every required criterion passes for every step.

        Non-required criteria are reported but don't block the verdict.
        """
        for sp_id, step_results in self.verify_plan().items():
            for step_id, criteria in step_results.items():
                for cid, result in criteria.items():
                    if not result.passed:
                        # Find the criterion to know if it's required.
                        sub_plan = self._plan.find_sub_plan(sp_id)
                        if sub_plan is None:
                            continue
                        step = sub_plan.find_step(step_id)
                        if step is None:
                            continue
                        criterion = next((c for c in step.criteria if c.id == cid), None)
                        if criterion is not None and criterion.required:
                            return False
        return True

    def assert_all_required_passing(self) -> None:
        """Raise VerificationCheckFailedError if any required criterion fails."""
        for sp_id, step_results in self.verify_plan().items():
            for step_id, criteria in step_results.items():
                for cid, result in criteria.items():
                    if not result.passed:
                        sub_plan = self._plan.find_sub_plan(sp_id)
                        if sub_plan is None:
                            continue
                        step = sub_plan.find_step(step_id)
                        if step is None:
                            continue
                        criterion = next((c for c in step.criteria if c.id == cid), None)
                        if criterion is not None and criterion.required:
                            raise VerificationCheckFailedError(
                                f"criterion {cid!r} for step {step_id!r} failed: {result.details}"
                            )
