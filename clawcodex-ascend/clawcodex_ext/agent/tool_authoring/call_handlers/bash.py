# pylint: disable=subprocess-run-check,consider-using-with
"""Bash call handler — executes whitelisted shell commands safely."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_DEFAULT_TIMEOUT_SEC = 300.0

# 轮询间隔：与 bash_tool._ABORT_POLL_INTERVAL_S 对齐，ESC 在 ~50ms 内生效。
_ABORT_POLL_INTERVAL_S = 0.05
# SIGKILL 后等待内核回收进程的上限，超时则放弃回收直接 communicate()。
_KILL_REAP_TIMEOUT_S = 2.0


def _argv_for_json_args_template(
    command_template: str,
    json_args: str,
) -> list[str]:
    """Split a SOP wrapper command without shell-parsing the JSON payload."""
    marker = "__CLAWCODEX_JSON_ARGS__"
    templated = command_template.replace("'{json_args}'", marker).replace('"{json_args}"', marker)
    templated = templated.replace("{json_args}", marker)
    argv = shlex.split(templated, posix=True)
    argv = [str(json_args) if arg == marker else arg for arg in argv]
    if argv and argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


class BashCallError(Exception):
    """Raised when a bash command fails or times out."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def resolve_bundle_venv_environment(context: Any | None) -> dict[str, str]:
    """Return environment overrides exposing ready bundle dependencies.

    Runtime execution never creates or repairs the environment.  ``sop
    convert`` owns dependency installation; a missing or stale marker is a
    conversion/deployment error and must not be recovered with an ad-hoc pip
    install from an Agent turn. The host interpreter remains in use so
    ClawCodex runtime dependencies stay available; bundle site-packages are
    prepended through ``PYTHONPATH`` to mirror in-process activation.
    """
    if context is None:
        return {}
    bundle = getattr(context, "bundle_context", None)
    if bundle is None:
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            bundle = get_active_bundle()
        except ImportError:
            return {}
    if bundle is None:
        return {}

    bundle_path = getattr(bundle, "bundle_path", None)
    if bundle_path is None:
        return {}

    try:
        from extensions.sop_converter.bundle_manifest import read_bundle_manifest
        from extensions.sop_converter.bundle_venv import (
            bundle_venv_site_packages,
            bundle_venv_python,
            is_venv_ready,
        )

        manifest = read_bundle_manifest(Path(bundle_path))
    except (ImportError, OSError):
        return {}
    if manifest is None or not manifest.sdk_requirements:
        return {}

    requirements = tuple(manifest.sdk_requirements)
    if not is_venv_ready(bundle_path, requirements):
        raise BashCallError(
            "bundle_venv_not_ready: converted bundle dependencies are missing or "
            f"stale for {bundle_path}. Re-run sop convert to rebuild the bundle "
            "venv; runtime tool execution will not install packages."
        )

    python_path = bundle_venv_python(bundle_path)
    if not python_path.is_file():
        raise BashCallError(
            "bundle_venv_not_ready: bundle Python is missing at "
            f"{python_path}. Re-run sop convert; runtime tool execution will not "
            "install packages."
        )
    site_packages = tuple(path for path in bundle_venv_site_packages(bundle_path) if path.is_dir())
    if not site_packages:
        raise BashCallError(
            "bundle_venv_not_ready: bundle site-packages are missing for "
            f"{bundle_path}. Re-run sop convert; runtime tool execution will "
            "not install packages."
        )

    python_path_entries = [str(path) for path in site_packages]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    existing_path = os.environ.get("PATH", "")
    executable_dir = str(python_path.parent)
    return {
        "CLAWCODEX_BUNDLE_VENV": str(python_path.parent.parent),
        "PYTHONPATH": os.pathsep.join(python_path_entries),
        "PATH": (executable_dir if not existing_path else executable_dir + os.pathsep + existing_path),
        "VIRTUAL_ENV": str(python_path.parent.parent),
    }


def resolve_agent_tool_bash_timeout_sec() -> float:
    """Return subprocess timeout for agent-tool bash handlers.

    ``AGENT_TOOL_BASH_TIMEOUT_SEC`` overrides the default (300s).
    """
    raw = os.environ.get("AGENT_TOOL_BASH_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_TIMEOUT_SEC


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants.

    On Windows, ``TerminateProcess`` only kills the direct child, leaving
    grandchildren (e.g. ``pip`` spawned by a wrapper script) as orphans
    that hold inherited pipe handles and cause ``communicate()`` to
    deadlock. ``taskkill /T /F`` recursively walks the tree. On POSIX,
    ``killpg`` sends the signal to the whole process group established
    via ``start_new_session=True``.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(  # nosec
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _run_subprocess_with_abort(
    argv: list[str] | str,
    *,
    use_argv: bool,
    timeout_sec: float,
    abort_signal: Any | None,
    env: dict[str, str] | None,
    stdin_input: str | None,
) -> tuple[int | None, str, str, bool, bool]:
    """Run a subprocess with abort + timeout supervision.

    Replaces the previous ``subprocess.run(  # nosec..., timeout=...)`` call which
    could not respond to ESC and deadlocked on Windows when the killed
    child left orphaned pipe holders. Mirrors the pattern in
    ``bash_tool._run_bash_with_abort``: launch in a new session/process
    group, poll for completion while watching ``abort_signal.aborted``
    and the deadline, then kill the whole tree on either trigger.

    Returns ``(returncode, stdout, stderr, interrupted, timed_out)``.
    """
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if stdin_input is not None:
        popen_kwargs["stdin"] = subprocess.PIPE
    else:
        popen_kwargs["stdin"] = subprocess.DEVNULL

    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    if use_argv:
        proc = subprocess.Popen(argv, **popen_kwargs)
    else:
        proc = subprocess.Popen(argv, shell=True, **popen_kwargs)  # nosec

    deadline = time.monotonic() + timeout_sec
    interrupted = False
    timed_out = False

    while True:
        if proc.poll() is not None:
            break
        if abort_signal is not None and getattr(abort_signal, "aborted", False):
            interrupted = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(_ABORT_POLL_INTERVAL_S)

    if interrupted or timed_out:
        _kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=_KILL_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass

    try:
        stdout, stderr = proc.communicate(input=stdin_input, timeout=_KILL_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""

    return (
        proc.returncode,
        stdout or "",
        stderr or "",
        interrupted,
        timed_out,
    )


def execute_bash(
    command_template: str,
    params: dict[str, Any],
    *,
    timeout_sec: float | None = None,
    context: Any | None = None,
    abort_signal: Any | None = None,
) -> str:
    """Execute a bash command from a validated template.

    Args:
        command_template: A format-string command, e.g. ``"glab project view {project_id}"``.
        params: Mapping of placeholder names to resolved values.
        context: Optional tool context. When provided and a bundle context is
            attached, ``CLAWCODEX_BUNDLE_PATH`` is injected into the subprocess
            environment so wrapper scripts can locate bundle-local artifacts
            regardless of the current working directory.
        abort_signal: Optional ``AbortSignal`` from the tool context. When
            ``abort_signal.aborted`` becomes True (e.g. user presses ESC),
            the subprocess tree is killed immediately instead of waiting
            for the timeout to elapse.

    Returns:
        stdout from the subprocess.

    Raises:
        BashCallError: If the command is aborted, times out, or exits non-zero.
    """
    use_argv = "{json_args}" in command_template and "json_args" in params
    if use_argv:
        command = command_template.replace("{json_args}", str(params["json_args"]))
    else:
        try:
            command = command_template.format(**params)
        except KeyError as exc:
            raise BashCallError(f"Missing parameter in template: {exc}") from exc
        except Exception as exc:
            raise BashCallError(f"Failed to format command template: {exc}") from exc

    if timeout_sec is None:
        timeout_sec = resolve_agent_tool_bash_timeout_sec()

    env: dict[str, str] | None = None
    if context is not None:
        bundle = getattr(context, "bundle_context", None)
        if bundle is not None:
            bundle_path = getattr(bundle, "bundle_path", None)
            if bundle_path is not None:
                env = dict(os.environ)
                env["CLAWCODEX_BUNDLE_PATH"] = str(bundle_path)

    for key in ("CLAWCODEX_SESSION_ID", "CLAWCODEX_CATALOG_DUAL_WRITE"):
        value = os.environ.get(key)
        if value:
            if env is None:
                env = dict(os.environ)
            env[key] = value

    if context is not None:
        session_id = getattr(context, "session_id", None)
        if session_id:
            if env is None:
                env = dict(os.environ)
            env["CLAWCODEX_SESSION_ID"] = str(session_id)

    # ponytail: feed __interactive_inputs through stdin pipe so tools
    # that call input()/getpass()/sys.stdin.read() in non-TTY subprocesses
    # receive pre-collected answers instead of blocking on inherited stdin.
    interactive_inputs = params.pop("__interactive_inputs", None)
    stdin_input: str | None = None
    if interactive_inputs:
        stdin_input = "\n".join(str(v) for v in interactive_inputs) + "\n"

    # SOP-generated Python wrappers use POSIX single quotes around JSON in
    # their persisted call_impl.  That is valid under /bin/sh but not under
    # Windows cmd.exe, so execute those wrapper commands as argv instead of
    # routing through the shell.
    argv: list[str] | None = None
    if use_argv:
        if re.match(r"^\s*python3?\s", command_template):
            bundle_env = resolve_bundle_venv_environment(context)
            if bundle_env:
                env = dict(env or os.environ)
                env.update(bundle_env)
        argv = _argv_for_json_args_template(
            command_template,
            str(params["json_args"]),
        )
    try:
        returncode, stdout, stderr, interrupted, timed_out = _run_subprocess_with_abort(
            argv if use_argv else command,
            use_argv=use_argv,
            timeout_sec=timeout_sec,
            abort_signal=abort_signal,
            env=env,
            stdin_input=stdin_input,
        )
    except OSError as exc:
        raise BashCallError(f"Failed to execute: {exc}") from exc

    if interrupted:
        raise BashCallError(
            f"Command aborted: {command[:80]}",
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    if timed_out:
        raise BashCallError(
            f"Command timed out after {int(timeout_sec)}s: {command[:80]}",
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    if returncode != 0:
        raise BashCallError(
            f"Command exited with {returncode}: {stderr.strip() or stdout.strip()}",
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )

    return stdout


def parse_sop_wrapper_stdout(raw: str) -> Any:
    """Extract the JSON payload printed by a sop-converter wrapper script.

    Wrapper subprocesses may emit SDK init logs on stdout before the final
    ``json.dumps(...)`` line.  Walk lines bottom-up and return the last line
    that parses as JSON; fall back to the trimmed raw text when none match.
    """
    text = raw.strip()
    if not text:
        return text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return text
