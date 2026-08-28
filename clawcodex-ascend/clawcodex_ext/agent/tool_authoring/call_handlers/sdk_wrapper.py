"""In-process execution of pos-converter SDK wrapper scripts."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

from clawcodex_ext.agent.sdk_context_registry import (
    ContextKey,
    get_sdk_context_registry,
)
from clawcodex_ext.agent.sdk_instance_registry import (
    BucketKey,
    get_sdk_instance_registry,
)
from clawcodex_ext.agent.tool_authoring.persistence import (
    TOOL_DIR,
)
from extensions.sop_converter.core.sdk_serialization import to_jsonable

_WRAPPER_CALL_RE = re.compile(r'(?:python3?|[^\s]*python[^\s]*)\s+"([^"]+)"\s+(\w+)\s+\'\{json_args\}\'')

_StatFingerprint = tuple[int, int]
_BootstrapInfo = tuple[str, tuple[str, ...]]
_ModuleFingerprint = tuple[int, int, str, str]

_MODULE_CACHE: dict[str, Any] = {}
_MODULE_CACHE_FINGERPRINT: dict[str, _ModuleFingerprint] = {}
_MODULE_CACHE_LOCK = threading.Lock()
_CWD_LOCK = threading.RLock()
_NO_CATALOG_FALLBACK = object()

_SCRIPT_USES_INSTANCE_CACHE: dict[str, bool] = {}
_SCRIPT_BUNDLE_BOOTSTRAP_CACHE: dict[
    str,
    tuple[_StatFingerprint, _BootstrapInfo | None],
] = {}


class SdkWrapperCallError(Exception):
    pass


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop bridge-internal keys the target stub does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def parse_sdk_wrapper_call_impl(call_impl: str) -> tuple[Path, str] | None:
    """Return ``(script_path, method_name)`` when *call_impl* is a wrapper template."""
    match = _WRAPPER_CALL_RE.search(call_impl.strip())
    if not match:
        return None
    return Path(match.group(1)), match.group(2)


def wrapper_requires_subprocess(call_impl: str) -> bool:
    """Return True when the wrapper command must run its CLI post-processing.

    Create-result persistence is implemented in the generated ``__main__``
    dispatch, so it cannot use the direct function fast path. Catalog fallback
    is handled in-process by :func:`execute_sdk_wrapper_in_process` so invoke
    tools keep their session-scoped SDK state.
    """
    return "--catalog-metadata" in parse_sdk_wrapper_cli_options(call_impl)


def parse_sdk_wrapper_cli_options(call_impl: str) -> dict[str, Any]:
    """Parse JSON-valued options following a generated wrapper call."""
    match = _WRAPPER_CALL_RE.search(call_impl.strip())
    if not match:
        return {}
    suffix = call_impl.strip()[match.end() :].strip()
    if not suffix:
        return {}
    try:
        argv = shlex.split(suffix, posix=True)
    except ValueError:
        return {}

    options: dict[str, Any] = {}
    idx = 0
    while idx < len(argv):
        flag = argv[idx]
        if not flag.startswith("--") or idx + 1 >= len(argv):
            idx += 1
            continue
        try:
            options[flag] = json.loads(argv[idx + 1])
        except json.JSONDecodeError:
            options[flag] = argv[idx + 1]
        idx += 2
    return options


def wrapper_uses_instance_cache(script_path: Path) -> bool:
    """True when the wrapper defines ``_get_instance`` (stateful class methods)."""
    resolved = str(script_path.resolve())
    cached = _SCRIPT_USES_INSTANCE_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError:
        uses = False
    else:
        uses = "def _get_instance(" in text
    _SCRIPT_USES_INSTANCE_CACHE[resolved] = uses
    return uses


def is_allowed_wrapper_script(script_path: Path) -> bool:
    """Only execute wrappers from known agent-tools script directories."""
    try:
        resolved = script_path.resolve()
    except OSError:
        return False

    allowed_roots: list[Path] = [(TOOL_DIR / "scripts").resolve()]
    bundles_root = TOOL_DIR / "bundles"
    if bundles_root.is_dir():
        for child in bundles_root.iterdir():
            scripts = child / "scripts"
            if scripts.is_dir():
                allowed_roots.append(scripts.resolve())

    for parent in resolved.parents:
        if parent.name == "scripts" and parent.parent.name == "agent-tools":
            allowed_roots.append(parent.resolve())
            break

    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def _literal_from_normalizer_call(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    if not isinstance(node.func, ast.Name) or node.func.id != "_normalize_bootstrap_path":
        return ""
    if not node.args:
        return ""
    try:
        value = ast.literal_eval(node.args[0])
    except (TypeError, ValueError, SyntaxError):
        return ""
    return value if isinstance(value, str) else ""


def _read_wrapper_bundle_bootstrap(
    script_path: Path,
) -> _BootstrapInfo | None:
    key = str(script_path.resolve())
    try:
        stat = script_path.stat()
    except OSError:
        _SCRIPT_BUNDLE_BOOTSTRAP_CACHE.pop(key, None)
        return None

    stat_fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _SCRIPT_BUNDLE_BOOTSTRAP_CACHE.get(key)
    if cached is not None and cached[0] == stat_fingerprint:
        return cached[1]

    try:
        tree = ast.parse(
            script_path.read_text(encoding="utf-8"),
            filename=str(script_path),
        )
    except (OSError, SyntaxError):
        _SCRIPT_BUNDLE_BOOTSTRAP_CACHE[key] = (stat_fingerprint, None)
        return None

    bundle_dir = ""
    requirements: tuple[str, ...] = ()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "_BUNDLE_DIR":
                bundle_dir = _literal_from_normalizer_call(stmt.value)
            elif target.id == "_SDK_REQUIREMENTS":
                try:
                    parsed = ast.literal_eval(stmt.value)
                except (TypeError, ValueError, SyntaxError):
                    parsed = ()
                if isinstance(parsed, (tuple, list)):
                    requirements = tuple(item for item in parsed if isinstance(item, str))

    result = (bundle_dir, requirements) if bundle_dir and requirements else None
    _SCRIPT_BUNDLE_BOOTSTRAP_CACHE[key] = (stat_fingerprint, result)
    return result


def _requirements_fingerprint(requirements: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(requirements).encode("utf-8")).hexdigest()


def _wrapper_module_fingerprint(script_path: Path) -> _ModuleFingerprint:
    stat = script_path.stat()
    bootstrap = _read_wrapper_bundle_bootstrap(script_path)
    if bootstrap is None:
        bundle_dir = ""
        requirements: tuple[str, ...] = ()
    else:
        bundle_dir, requirements = bootstrap
    return (
        stat.st_mtime_ns,
        stat.st_size,
        bundle_dir,
        _requirements_fingerprint(requirements),
    )


def _ensure_bundle_imports_for_in_process(script_path: Path) -> None:
    bootstrap = _read_wrapper_bundle_bootstrap(script_path)
    if bootstrap is None:
        return

    bundle_dir, requirements = bootstrap
    from extensions.sop_converter.bundle_venv import (
        activate_bundle_venv_imports,
        ensure_bundle_venv,
    )
    from extensions.sop_converter.sdk_dependency_resolver import SdkDependencySpec

    deps = SdkDependencySpec(
        requirements=requirements,
        source="manifest",
        raw_path="",
    )
    ensure_bundle_venv(bundle_dir, deps)
    activate_bundle_venv_imports(bundle_dir)


def _exec_wrapper_module_in_process(spec: Any, module: Any, script_path: Path) -> None:
    from extensions.sop_converter import bundle_venv as bundle_venv_mod

    # Forces soft venv activation: ensure_bundle_venv_and_reexec will not
    # os.execv while this context is active (keeps Agent/REPL alive).
    with bundle_venv_mod.in_process_bundle_venv_reexec():
        with _CWD_LOCK:
            old_cwd = os.getcwd()
            try:
                spec.loader.exec_module(module)
            finally:
                os.chdir(old_cwd)


def _call_with_catalog_fallback(
    module: Any,
    fn: Any,
    kwargs: dict[str, Any],
    catalog_fallback: dict[str, Any] | None,
) -> Any:
    def _recover(value: Any) -> Any:
        if not catalog_fallback:
            return _NO_CATALOG_FALLBACK
        should_recover = getattr(module, "_should_catalog_fallback", None)
        recover = getattr(module, "_try_catalog_fallback", None)
        if not callable(should_recover) or not callable(recover):
            return _NO_CATALOG_FALLBACK
        if not should_recover(value):
            return _NO_CATALOG_FALLBACK
        recovered = recover(catalog_fallback, kwargs, original_error=value)
        return _NO_CATALOG_FALLBACK if recovered is None else recovered

    try:
        result = fn(**kwargs)
    except Exception as exc:
        recovered = _recover(exc)
        if recovered is not _NO_CATALOG_FALLBACK:
            return recovered
        raise

    recovered = _recover(result)
    return result if recovered is _NO_CATALOG_FALLBACK else recovered


def _call_wrapper_fn(
    module: Any,
    fn: Any,
    kwargs: dict[str, Any],
    catalog_fallback: dict[str, Any] | None,
) -> Any:
    module_dir = getattr(module, "_MODULE_DIR", "")
    if not module_dir or not os.path.isdir(module_dir):
        return _call_with_catalog_fallback(module, fn, kwargs, catalog_fallback)

    with _CWD_LOCK:
        old_cwd = os.getcwd()
        os.chdir(module_dir)
        try:
            return _call_with_catalog_fallback(module, fn, kwargs, catalog_fallback)
        finally:
            os.chdir(old_cwd)


def _load_wrapper_module(script_path: Path) -> Any:
    key = str(script_path.resolve())
    with _MODULE_CACHE_LOCK:
        fingerprint = _wrapper_module_fingerprint(script_path)
        module = _MODULE_CACHE.get(key)
        if module is not None and _MODULE_CACHE_FINGERPRINT.get(key) == fingerprint:
            return module
        if module is not None:
            _MODULE_CACHE.pop(key, None)
            _MODULE_CACHE_FINGERPRINT.pop(key, None)

        try:
            _ensure_bundle_imports_for_in_process(script_path)
        except Exception as exc:  # noqa: BLE001
            raise SdkWrapperCallError(f"Failed to prepare bundle venv imports for {script_path}: {exc}") from exc

        spec = importlib.util.spec_from_file_location(
            f"sop_wrapper_{abs(hash(key)) & 0xFFFFFFFF:08x}",
            script_path,
        )
        if spec is None or spec.loader is None:
            raise SdkWrapperCallError(f"Cannot load wrapper module: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            _exec_wrapper_module_in_process(spec, module, script_path)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _MODULE_CACHE[key] = module
        _MODULE_CACHE_FINGERPRINT[key] = fingerprint
        return module


def should_use_in_process_sdk_wrapper(spec: Any) -> bool:
    """Whether *spec* should run via session-scoped in-process wrapper dispatch."""
    if getattr(spec, "source", None) not in {"sop-converter", "pos-converter"}:
        return False
    call_impl = str(getattr(spec, "call_impl", ""))
    parsed = parse_sdk_wrapper_call_impl(call_impl)
    if parsed is None:
        return False
    if wrapper_requires_subprocess(call_impl):
        return False
    script_path, _ = parsed
    if not script_path.is_file():
        return False
    if getattr(spec, "stateful_wrapper", False):
        return True
    # Legacy specs: class wrappers expose _get_instance; standalone wrappers
    # are enabled once the script exists (Plan A phase 2).
    return True


def execute_sdk_wrapper_in_process(
    *,
    script_path: Path,
    method_name: str,
    kwargs: dict[str, Any],
    session_id: str | None,
    agent_id: str | None,
    catalog_fallback: dict[str, Any] | None = None,
) -> Any:
    if not is_allowed_wrapper_script(script_path):
        raise SdkWrapperCallError(f"Wrapper script outside allowed agent-tools directories: {script_path}")

    module = _load_wrapper_module(script_path)
    fn = getattr(module, method_name, None)
    if fn is None or not callable(fn):
        raise SdkWrapperCallError(f"Wrapper {script_path.name} has no callable method {method_name!r}")

    interactive_inputs = kwargs.pop("__interactive_inputs", None)
    if interactive_inputs is not None:
        set_interactive_inputs = getattr(module, "_set_interactive_inputs", None)
        if callable(set_interactive_inputs):
            set_interactive_inputs(interactive_inputs)

    bridge_stdin_config = kwargs.pop("__stdin_config", None)
    bridge_env = kwargs.pop("__env", None)
    if bridge_stdin_config is not None:
        module._bridge_stdin_config = bridge_stdin_config
    if bridge_env is not None:
        module._bridge_subprocess_env = bridge_env

    call_kwargs = _filter_kwargs_for_callable(fn, kwargs)

    context_registry = get_sdk_context_registry()
    context_key: ContextKey = context_registry.context_key(
        session_id=session_id,
        agent_id=agent_id,
    )
    ctx = context_registry.get_context(context_key)
    ctx_lock = context_registry.lock_for(context_key)

    if wrapper_uses_instance_cache(script_path):
        instance_registry = get_sdk_instance_registry()
        bucket_key: BucketKey = instance_registry.bucket_key(
            session_id=session_id,
            agent_id=agent_id,
            script_path=script_path,
        )
        bucket_lock = instance_registry.lock_for(bucket_key)

        def _run_class_method() -> Any:
            module._instances = instance_registry.get_bucket(bucket_key)
            return to_jsonable(_call_wrapper_fn(module, fn, call_kwargs, catalog_fallback))

        with bucket_lock:
            with ctx_lock:
                return ctx.run(_run_class_method)

    def _run_standalone() -> Any:
        return to_jsonable(_call_wrapper_fn(module, fn, call_kwargs, catalog_fallback))

    with ctx_lock:
        return ctx.run(_run_standalone)
