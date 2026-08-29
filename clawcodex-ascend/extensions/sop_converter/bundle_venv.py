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

"""Per-bundle virtual environment management for SOP-converted SDK tools."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import site
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .runtime_paths import is_wsl_runtime, normalize_runtime_path
from .sdk_dependency_resolver import SdkDependencySpec

logger = logging.getLogger(__name__)

_VENV_MARKER = ".bundle-venv-ready"
_IN_PROCESS_REEXEC_ENV = "CLAWCODEX_IN_PROCESS_SDK_WRAPPER"
_IN_PROCESS_REEXEC_STATE = threading.local()

_ACTIVE_IMPORT_BUNDLE_DIR: Path | None = None
_ACTIVE_IMPORT_BUNDLE_WARNED = False
_ACTIVE_IMPORT_BUNDLE_LOCK = threading.Lock()


def _wsl_bundle_venv_slug(bundle_path: Path) -> tuple[str, str]:
    """Return ``(preferred_dirname, legacy_hash)`` for the WSL cache layout."""

    digest = hashlib.sha256(str(bundle_path).encode("utf-8")).hexdigest()[:16]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", bundle_path.name).strip("._-") or "bundle"
    return f"{name[:64]}-{digest}", digest


def bundle_venv_dir(bundle_dir: str | Path) -> Path:
    """Return the standard virtualenv directory for a converted bundle.

    On WSL, the venv is placed on the native ext4 filesystem
    (``~/.cache/clawcodex/bundle-venvs/<bundle-name>-<hash>``) instead of
    ``/mnt/d/`` to avoid the extreme I/O penalty of the 9P bridge to Windows
    NTFS. Writing thousands of package files on ``/mnt/`` can be 10-100x
    slower than on the native filesystem.

    Existing hash-only directories under ``bundle-venvs/`` are still reused
    when the named path does not exist yet.
    """

    bundle_path = normalize_runtime_path(bundle_dir)
    if is_wsl_runtime():
        cache_root = Path.home() / ".cache" / "clawcodex" / "bundle-venvs"
        preferred_name, legacy_hash = _wsl_bundle_venv_slug(bundle_path)
        preferred = cache_root / preferred_name
        legacy = cache_root / legacy_hash
        if not preferred.exists() and legacy.exists():
            return legacy
        return preferred
    return bundle_path / ".venv"


def bundle_venv_python(bundle_dir: str | Path) -> Path:
    """Return the virtualenv Python executable path for *bundle_dir*."""

    venv = bundle_venv_dir(bundle_dir)
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def bundle_venv_site_packages(bundle_dir: str | Path) -> tuple[Path, ...]:
    """Return import directories for the bundle virtualenv."""

    venv = bundle_venv_dir(bundle_dir)
    if os.name == "nt":
        return (venv / "Lib" / "site-packages",)

    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        venv / "lib" / py_tag / "site-packages",
        venv / "lib64" / py_tag / "site-packages",
    ]
    return tuple(path for path in candidates if path.is_dir()) or (candidates[0],)


def activate_bundle_venv_imports(bundle_dir: str | Path) -> tuple[str, ...]:
    """Expose a bundle venv's site-packages to the current Python process.

    This is used by in-process wrapper dispatch. It does not activate scripts
    or mutate ``sys.executable``; it only makes installed SDK dependencies
    importable while keeping the REPL/agent process alive.

    Current limitation: ``sys.path`` and ``sys.modules`` are process-global, so
    one REPL process cannot fully isolate multiple bundles with conflicting
    dependency versions. We warn once when a second bundle is activated.
    """

    normalized_bundle_dir = normalize_runtime_path(bundle_dir)
    _record_active_import_bundle(normalized_bundle_dir)

    added: list[str] = []
    for path in bundle_venv_site_packages(normalized_bundle_dir):
        if not path.is_dir():
            continue
        normalized = str(path)
        before = set(sys.path)
        site.addsitedir(normalized)
        new_entries = [entry for entry in sys.path if entry not in before]
        if not new_entries and normalized in sys.path:
            new_entries = [normalized]
        for entry in new_entries:
            try:
                sys.path.remove(entry)
            except ValueError:
                continue
        for entry in reversed(new_entries):
            sys.path.insert(0, entry)
        added.extend(new_entries)

    return tuple(added)


@contextmanager
def in_process_bundle_venv_reexec() -> Iterator[None]:
    """Mark the current scope as in-process SDK wrapper dispatch.

    While active, :func:`ensure_bundle_venv_and_reexec` will **not** ``os.execv``
    into the bundle interpreter; it only ensures the venv and calls
    :func:`activate_bundle_venv_imports`. Used by
    ``execute_sdk_wrapper_in_process`` so the Agent/REPL process is not replaced.
    """

    previous = getattr(_IN_PROCESS_REEXEC_STATE, "enabled", False)
    _IN_PROCESS_REEXEC_STATE.enabled = True
    try:
        yield
    finally:
        _IN_PROCESS_REEXEC_STATE.enabled = previous


def _is_in_process_reexec() -> bool:
    """True when real ``os.execv`` re-exec must be skipped (soft import activation).

    Set by :func:`in_process_bundle_venv_reexec` or env
    ``CLAWCODEX_IN_PROCESS_SDK_WRAPPER=1``.
    """

    return bool(getattr(_IN_PROCESS_REEXEC_STATE, "enabled", False) or os.environ.get(_IN_PROCESS_REEXEC_ENV) == "1")


def _record_active_import_bundle(bundle_dir: Path) -> None:
    global _ACTIVE_IMPORT_BUNDLE_DIR, _ACTIVE_IMPORT_BUNDLE_WARNED

    with _ACTIVE_IMPORT_BUNDLE_LOCK:
        if _ACTIVE_IMPORT_BUNDLE_DIR is None:
            _ACTIVE_IMPORT_BUNDLE_DIR = bundle_dir
            return
        if _ACTIVE_IMPORT_BUNDLE_DIR == bundle_dir:
            return
        if not _ACTIVE_IMPORT_BUNDLE_WARNED:
            message = (
                "[bundle-venv] Warning: activating SDK dependencies for a second "
                f"bundle in the same Python process ({_ACTIVE_IMPORT_BUNDLE_DIR} -> "
                f"{bundle_dir}). In-process wrapper dispatch shares sys.path and "
                "sys.modules, so conflicting dependency versions are not isolated. "
                "Use one converted bundle per REPL session when dependencies differ."
            )
            logger.warning("%s", message)
            print(message, file=sys.stderr)
            _ACTIVE_IMPORT_BUNDLE_WARNED = True
        _ACTIVE_IMPORT_BUNDLE_DIR = bundle_dir


def _reset_bundle_venv_import_state() -> None:
    """Reset in-process import activation bookkeeping for tests."""

    global _ACTIVE_IMPORT_BUNDLE_DIR, _ACTIVE_IMPORT_BUNDLE_WARNED

    with _ACTIVE_IMPORT_BUNDLE_LOCK:
        _ACTIVE_IMPORT_BUNDLE_DIR = None
        _ACTIVE_IMPORT_BUNDLE_WARNED = False
    _IN_PROCESS_REEXEC_STATE.enabled = False


def is_venv_ready(
    bundle_dir: str | Path,
    requirements: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Return True when the bundle venv exists and matches *requirements*."""

    python_path = bundle_venv_python(bundle_dir)
    marker = bundle_venv_dir(bundle_dir) / _VENV_MARKER
    if not python_path.is_file() or not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    marker_platform = data.get("platform_tag")
    if marker_platform and marker_platform != _platform_tag():
        return False
    if requirements is None:
        return True
    return data.get("requirements_hash") == _requirements_hash(tuple(requirements))


def ensure_bundle_venv(
    bundle_dir: str | Path,
    deps: SdkDependencySpec,
    *,
    force: bool = False,
) -> Path:
    """Create the bundle venv if needed and install SDK dependencies into it."""

    bundle_path = normalize_runtime_path(bundle_dir)
    venv_dir = bundle_venv_dir(bundle_path)
    python_path = bundle_venv_python(bundle_path)
    requirements = tuple(deps.requirements)

    if not force and is_venv_ready(bundle_path, requirements):
        return python_path

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if force and venv_dir.exists():
        _reset_venv_dir(venv_dir, reason="force requested")
    elif venv_dir.exists() and not python_path.is_file():
        _reset_venv_dir(
            venv_dir,
            reason=f"incompatible with current runtime ({_platform_tag()})",
        )

    if not python_path.is_file():
        _create_venv(venv_dir)
    if not python_path.is_file():
        raise RuntimeError(f"Bundle venv Python was not created: {python_path}")

    if requirements:
        install_timeout = _install_timeout_seconds()
        _install_requirements(python_path, list(requirements), timeout=install_timeout)

    marker_payload = {
        "version": 1,
        "python": str(python_path),
        "requirements": list(requirements),
        "requirements_hash": _requirements_hash(requirements),
        "source": deps.source,
        "raw_path": deps.raw_path,
        "platform_tag": _platform_tag(),
    }
    (venv_dir / _VENV_MARKER).write_text(
        json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return python_path


def ensure_bundle_venv_and_reexec(
    bundle_dir: str | Path,
    deps: SdkDependencySpec,
    *,
    argv: list[str] | None = None,
    script_file: str | None = None,
) -> None:
    """Ensure the bundle venv exists; optionally re-exec into its Python.

    When this takes effect as a real process replace (``os.execv``):

    - Caller is **not** under :func:`in_process_bundle_venv_reexec` / not with
      ``CLAWCODEX_IN_PROCESS_SDK_WRAPPER=1`` (typical: wrapper run as a
      standalone script, e.g. bash/subprocess, with
      ``CLAWCODEX_ENABLE_BUNDLE_VENV_REEXEC=1`` so the wrapper invokes us);
    - ``deps.requirements`` is non-empty;
    - Current ``sys.executable`` is not already the bundle venv python.

    When it does **not** really re-exec (REPL / Agent default path):

    - In-process SDK wrapper dispatch sets the in-process flag, so we only
      ``ensure_bundle_venv`` + ``activate_bundle_venv_imports`` and return.
      ``sys.executable`` stays the host interpreter (e.g. clawcodex ``.venv``).
      Ray/multiprocessing workers that spawn from ``sys.executable`` therefore
      still see the host env unless dependencies are also present there or
      another isolation mechanism is used.

    ``CLAWCODEX_ENABLE_BUNDLE_VENV_REEXEC=1`` only opts the generated wrapper
    into *calling* this function; it does not override the in-process short
    circuit above.
    """

    if not deps.requirements:
        return
    normalized_bundle_dir = normalize_runtime_path(bundle_dir)
    if _is_in_process_reexec():
        # Soft path: keep Agent/REPL process; do not os.execv.
        ensure_bundle_venv(normalized_bundle_dir, deps)
        activate_bundle_venv_imports(normalized_bundle_dir)
        return

    python_path = ensure_bundle_venv(normalized_bundle_dir, deps)
    try:
        current = Path(sys.executable).resolve()
        target = python_path.resolve()
    except OSError:
        current = Path(sys.executable)
        target = python_path
    if current == target:
        return

    effective_argv = list(argv or sys.argv)
    executable_script = script_file or effective_argv[0]
    os.execv(str(target), [str(target), executable_script, *effective_argv[1:]])


def _create_venv(venv_dir: Path, *, timeout: float = 120.0) -> None:
    uv = _find_uv()
    python_spec = os.environ.get(
        "CLAWCODEX_BUNDLE_PYTHON",
        f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    if uv:
        returncode, tail = _run_command_streamed(
            [uv, "venv", str(venv_dir), "--python", python_spec],
            f"Creating bundle venv at {venv_dir}",
            timeout=timeout,
        )
        if returncode == 0:
            return
        logger.warning(
            "uv venv failed for %s with python %s; falling back to sys.executable: %s",
            venv_dir,
            python_spec,
            tail,
        )

    returncode, tail = _run_command_streamed(
        [sys.executable, "-m", "venv", str(venv_dir)],
        f"Creating bundle venv with stdlib venv at {venv_dir}",
        timeout=timeout,
    )
    if returncode != 0:
        raise RuntimeError(f"Bundle venv creation failed: {tail}")


# Ponytail: Tsinghua mirror as default (most users are in China),
# falling back to Aliyun then PyPI on timeout.
# User-configured index URLs (PIP_INDEX_URL / UV_INDEX_URL) take precedence
# over the built-in defaults so private registries and regional mirrors
# are respected without code changes.
_USER_INDEX_URL = os.environ.get("UV_INDEX_URL") or os.environ.get("PIP_INDEX_URL") or ""
_DEFAULT_INDEX = _USER_INDEX_URL or "https://pypi.tuna.tsinghua.edu.cn/simple"
_FALLBACK_INDEXES = [
    "https://mirrors.aliyun.com/pypi/simple/",
    "",  # empty = PyPI default (no --index-url)
]
_PYPI_SIMPLE = "https://pypi.org/simple"
_BUILDING_EQ_RE = re.compile(r"Building\s+(\w+)==(\S+)", re.IGNORECASE)
_BUILDING_WHEEL_FOR_RE = re.compile(r"Building wheel for (\w+)", re.IGNORECASE)


def _install_timeout_seconds() -> float:
    """Install timeout for ``uv pip install`` (env override for large wheels)."""

    raw = os.environ.get("CLAWCODEX_BUNDLE_VENV_INSTALL_TIMEOUT", "").strip()
    if raw:
        try:
            return max(60.0, float(raw))
        except ValueError:
            logger.warning(
                "Invalid CLAWCODEX_BUNDLE_VENV_INSTALL_TIMEOUT=%r; using default",
                raw,
            )
    return 1200.0


def _uv_http_timeout_seconds() -> int:
    """Default ``UV_HTTP_TIMEOUT`` for bundle venv ``uv`` subprocesses (seconds)."""

    raw = os.environ.get("CLAWCODEX_BUNDLE_VENV_UV_HTTP_TIMEOUT", "").strip()
    if raw:
        try:
            return max(30, int(float(raw)))
        except ValueError:
            logger.warning(
                "Invalid CLAWCODEX_BUNDLE_VENV_UV_HTTP_TIMEOUT=%r; using default",
                raw,
            )
    return 300


def _uv_concurrent_downloads() -> int:
    """Default ``UV_CONCURRENT_DOWNLOADS`` for bundle venv ``uv`` subprocesses."""

    raw = os.environ.get("CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_DOWNLOADS", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            logger.warning(
                "Invalid CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_DOWNLOADS=%r; using default",
                raw,
            )
    return 16


def _uv_concurrent_builds() -> int:
    """Default ``UV_CONCURRENT_BUILDS`` for bundle venv ``uv`` subprocesses."""

    raw = os.environ.get("CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_BUILDS", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            logger.warning(
                "Invalid CLAWCODEX_BUNDLE_VENV_UV_CONCURRENT_BUILDS=%r; using default",
                raw,
            )
    return 8


def _subprocess_env(cmd: list[str]) -> dict[str, str] | None:
    """Return an env dict for *cmd* when it invokes ``uv``, else inherit default."""

    if not cmd:
        return None
    if Path(cmd[0]).name != "uv":
        return None
    env = os.environ.copy()
    if not env.get("UV_HTTP_TIMEOUT", "").strip():
        env["UV_HTTP_TIMEOUT"] = str(_uv_http_timeout_seconds())
    if not env.get("UV_CONCURRENT_DOWNLOADS", "").strip():
        env["UV_CONCURRENT_DOWNLOADS"] = str(_uv_concurrent_downloads())
    if not env.get("UV_CONCURRENT_BUILDS", "").strip():
        env["UV_CONCURRENT_BUILDS"] = str(_uv_concurrent_builds())
    return env


_WHEEL_FIRST_PACKAGES = frozenset({"pyarrow", "grpcio"})


def _install_wheel_preference_flags(*, use_uv: bool) -> list[str]:
    """Prefer wheels when using pip; uv has no ``--prefer-binary`` (through 0.11.x)."""

    if use_uv:
        return []
    return ["--prefer-binary"]


def _pypi_extra_index_flags(*, use_uv: bool, index_url: str | None) -> list[str]:
    """When using a mirror, also consult PyPI for wheels."""

    if not index_url:
        return []
    normalized = index_url.rstrip("/").lower()
    if normalized in {_PYPI_SIMPLE.lower(), "https://pypi.org/simple/"}:
        return []
    if use_uv:
        return ["--extra-index-url", _PYPI_SIMPLE]
    return ["--extra-index-url", _PYPI_SIMPLE]


def _last_building_package(tail: str) -> tuple[str, str | None] | None:
    """Return the last package seen in pip/uv sdist build output, if any."""

    name: str | None = None
    version: str | None = None
    for line in tail.splitlines():
        match = _BUILDING_EQ_RE.search(line)
        if match:
            name, version = match.group(1), match.group(2)
            continue
        match = _BUILDING_WHEEL_FOR_RE.search(line)
        if match:
            name = match.group(1)
            version = None
    if name:
        return name, version
    return None


def _remediate_sdist_with_pypi_wheel(
    python_path: Path,
    package: str,
    version: str | None,
    *,
    timeout: float,
) -> bool:
    """Install *package* from PyPI using wheels only (best-effort)."""

    uv = _find_uv()
    if not uv:
        return False

    def _wheel_only_install(spec: str, label: str) -> bool:
        cmd = [
            uv,
            "pip",
            "install",
            "--no-progress",
            "--python",
            str(python_path),
            "--index-url",
            _PYPI_SIMPLE,
            "--only-binary",
            package,
            spec,
        ]
        try:
            returncode, remediate_tail = _run_command_streamed(
                cmd,
                label,
                timeout=min(timeout, 600.0),
                heartbeat_interval=30,
            )
        except RuntimeError as exc:
            logger.warning("Wheel remediation failed for %s: %s", spec, exc)
            return False
        if returncode != 0:
            logger.warning("Wheel remediation failed for %s: %s", spec, remediate_tail)
            return False
        return True

    if version:
        spec = f"{package}=={version}"
        if _wheel_only_install(spec, f"Replacing sdist with PyPI wheel for {spec}"):
            return True
        if package in _WHEEL_FIRST_PACKAGES:
            return _wheel_only_install(
                package,
                f"Replacing sdist with PyPI wheel (any version) for {package}",
            )
        return False

    return _wheel_only_install(
        package,
        f"Replacing sdist with PyPI wheel for {package}",
    )


def _is_user_configured_index() -> bool:
    """Return True when the user explicitly set PIP_INDEX_URL / UV_INDEX_URL.

    When True, retry attempts keep using the user's index instead of
    cycling through the built-in fallback mirrors, since a private
    registry failure is unlikely to be resolved by switching to Tsinghua.
    """
    return bool(_USER_INDEX_URL)


def _install_requirements(python_path: Path, requirements: list[str], *, timeout: float = 1200.0) -> None:
    uv = _find_uv()
    base_cmd: list[str]
    if uv:
        base_cmd = [
            uv,
            "pip",
            "install",
            "--no-progress",
            "--python",
            str(python_path),
            *_install_wheel_preference_flags(use_uv=True),
        ]
    else:
        base_cmd = [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "off",
            *_install_wheel_preference_flags(use_uv=False),
        ]

    user_index = _is_user_configured_index()
    mirrors_tried = -1  # -1 = default index, 0+ = fallback mirror
    last_tail = ""
    # Detect dependency-resolution errors (version conflicts) that no mirror
    # switch can fix, so we can stop retrying early instead of wasting 3 cycles.
    resolution_error_markers = (
        "ResolutionImpossible",
        "conflict",
        "Cannot install",
        "ERROR: Cannot resolve",
        "incompatible",
    )

    for attempt in range(1, 4):
        cmd = list(base_cmd)
        index_url: str | None = None
        if user_index:
            # User explicitly configured an index; never override it.
            index_url = _DEFAULT_INDEX
        elif mirrors_tried < 0:
            index_url = _DEFAULT_INDEX
        elif mirrors_tried < len(_FALLBACK_INDEXES):
            index_url = _FALLBACK_INDEXES[mirrors_tried] or None
        if index_url:
            if uv:
                cmd += ["--index-url", index_url]
                cmd += _pypi_extra_index_flags(use_uv=True, index_url=index_url)
            else:
                cmd += ["-i", index_url]
                cmd += _pypi_extra_index_flags(use_uv=False, index_url=index_url)
        cmd += list(requirements)

        label_parts = [f"Installing {len(requirements)} SDK dependencies into {python_path}"]
        if attempt > 1:
            label_parts[0] = f"Retry {attempt - 1}/2: installing {len(requirements)} SDK dependencies"
        if index_url:
            label_parts.append(f"(mirror: {index_url})")
        label = " ".join(label_parts)

        try:
            returncode, tail = _run_command_streamed(
                cmd,
                label,
                timeout=timeout,
                heartbeat_interval=30,
            )
        except RuntimeError as exc:
            returncode = -1
            tail = str(exc)
        last_tail = tail
        if returncode == 0:
            return

        # Dependency-resolution failures cannot be fixed by retrying or
        # switching mirrors — abort early with a clear error.
        is_resolution_error = any(marker in tail for marker in resolution_error_markers)
        if is_resolution_error:
            raise RuntimeError(
                f"Bundle venv dependency installation failed due to a resolution conflict (not retried): {last_tail}"
            )

        if attempt < 3:
            timed_out = "timed out" in tail.lower()
            building = _last_building_package(tail)
            if building is not None:
                package, version = building
                _remediate_sdist_with_pypi_wheel(
                    python_path,
                    package,
                    version,
                    timeout=timeout,
                )
            # Switch primary mirror only on download/install timeout. Sdist build
            # failures are retried on the same index so uv reuses the venv and
            # global wheel cache instead of re-resolving against a new mirror.
            if not user_index and timed_out:
                mirrors_tried += 1
            elif attempt < 3 and building is not None:
                print(
                    "[bundle-venv] Retrying on same mirror (venv + uv cache reuse; "
                    "only missing/failed packages should be fetched)",
                    file=sys.stderr,
                    flush=True,
                )

            wait = 2**attempt
            print(
                f"[bundle-venv] Install attempt {attempt} failed, retrying in {wait}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"Bundle venv dependency installation failed after 3 attempts: {last_tail}")


def _find_uv() -> str | None:
    for candidate in ("uv", os.path.expanduser("~/.local/bin/uv")):
        resolved = shutil.which(candidate) if candidate == "uv" else candidate
        if not resolved:
            continue
        try:
            result = subprocess.run(
                [resolved, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return resolved
    return None


def _run_command_streamed(
    cmd: list[str],
    description: str,
    *,
    timeout: float = 300.0,
    heartbeat_interval: float = 0,
) -> tuple[int, str]:
    """Run *cmd*, streaming combined output to stderr and returning its tail.

    If the process does not exit within *timeout* seconds, it is killed and
    a RuntimeError is raised.  This prevents indefinite hangs caused by
    network stalls (e.g., TCP connections that never close) or slow I/O on
    ``/mnt/`` filesystems in WSL.

    When *heartbeat_interval* > 0, a ``.`` is printed to stderr every
    *heartbeat_interval* seconds while no process output arrives, so the user
    can tell the operation is still in progress.
    """

    print(f"[bundle-venv] {description}", file=sys.stderr, flush=True)
    print(f"[bundle-venv] $ {_format_cmd(cmd)}", file=sys.stderr, flush=True)
    popen_env = _subprocess_env(cmd)
    try:
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=popen_env,
        )
    except OSError as exc:
        raise RuntimeError(f"{description} failed to start: {exc}") from exc

    tail: deque[str] = deque(maxlen=80)
    assert proc.stdout is not None

    # Read output in a daemon thread so the main thread can enforce timeout.
    reader_errors: list[BaseException] = []
    output_arrived = threading.Event()

    def _reader() -> None:
        try:
            for line in proc.stdout:
                tail.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()
                output_arrived.set()
        except BaseException as exc:  # noqa: BLE001
            reader_errors.append(exc)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Heartbeat: print '.' periodically when no output arrives.
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(heartbeat_interval):
            if not output_arrived.is_set():
                sys.stderr.write(".")
                sys.stderr.flush()
            output_arrived.clear()

    heartbeat_thread: threading.Thread | None = None
    if heartbeat_interval > 0:
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        proc.wait(timeout=timeout)
        # Drain a final newline after heartbeat dots.
        if heartbeat_thread is not None:
            sys.stderr.write("\n")
            sys.stderr.flush()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader_thread.join(timeout=5)
        raise RuntimeError(
            f"{description} timed out after {timeout:.0f}s and was killed. Last output:\n{''.join(tail).strip()}"
        ) from None
    finally:
        heartbeat_stop.set()

    reader_thread.join(timeout=10)
    if reader_errors:
        raise RuntimeError(f"Output reader error: {reader_errors[0]}")
    return proc.returncode, "".join(tail).strip()


def _format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def _reset_venv_dir(venv_dir: Path, *, reason: str) -> None:
    # Safety: refuse to delete arbitrary paths. Allow:
    # 1. <bundle_dir>/.venv (standard non-WSL layout)
    # 2. ~/.cache/clawcodex/bundle-venvs/<name>-<hash> (WSL native ext4 layout)
    is_standard = venv_dir.name == ".venv"
    is_wsl_cache = "bundle-venvs" in venv_dir.parts
    if not is_standard and not is_wsl_cache:
        raise RuntimeError(f"Refusing to reset non-standard bundle venv: {venv_dir}")
    print(
        f"[bundle-venv] Resetting bundle venv ({reason}): {venv_dir}",
        file=sys.stderr,
        flush=True,
    )
    if venv_dir.is_symlink() or venv_dir.is_file():
        venv_dir.unlink()
    elif venv_dir.exists():
        shutil.rmtree(venv_dir)


def _platform_tag() -> str:
    if os.name == "nt":
        return "windows"
    if is_wsl_runtime():
        return "wsl"
    return sys.platform


def _requirements_hash(requirements: tuple[str, ...]) -> str:
    payload = "\n".join(requirements).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
