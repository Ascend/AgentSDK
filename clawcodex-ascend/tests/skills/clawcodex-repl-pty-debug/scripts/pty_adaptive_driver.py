#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Run the ClawCodex PTY controller with adaptive callback decisions.

The decider file must define:

    def first_request() -> dict: ...
    def decide_next(response: dict) -> dict | None: ...

The returned dict must be a controller request with an "op" key. Return None
only when no more controller operations are needed; return an explicit
{"op": "observe", ...} when the child needs more time before the next decision.

Use this when a shell-only agent needs same-session adaptive turns without
rewriting controller stdin/stdout plumbing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TextIO


_ALLOWED_OPS = {"start", "send", "key", "raw", "observe", "stop", "exit"}


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_decider(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pty_adaptive_decider", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load decider: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise SystemExit(
            f"decider must define callable {name}(). Minimal contract:\n\n"
            "def first_request():\n"
            '    return {"op": "start"}\n\n'
            "def decide_next(response):\n"
            '    if response.get("event") == "ready":\n'
            '        return {"op": "send", "text": "...", "label": "turn1"}\n'
            "    return None\n"
        )
    return value


def _request_from_decider(value: Any, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{source} must return a controller request dict")
    if "op" not in value:
        if "action" in value:
            raise RuntimeError(f'{source} must use "op", not "action"; example: {{"op": "start"}}')
        raise RuntimeError(f'{source} must include an "op" key')
    op = value["op"]
    if not isinstance(op, str) or op not in _ALLOWED_OPS:
        allowed = ", ".join(sorted(_ALLOWED_OPS))
        raise RuntimeError(f"{source} returned unsupported op {op!r}; expected one of: {allowed}")
    return dict(value)


class _TeeStdout:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        self._primary.write(text)
        self._secondary.write(text)
        self._secondary.flush()
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _response_basis(response: dict[str, object]) -> dict[str, object]:
    from decider_helpers import decision_basis

    keys = (
        "event",
        "op",
        "label",
        "kind",
        "state",
        "error_kind",
        "ok",
        "signals",
        "artifact_dir",
        "step",
        "input_source",
    )
    basis = {key: _json_safe(response[key]) for key in keys if key in response}
    basis.update(_json_safe(decision_basis(response)))
    return basis


def _write_progress(progress: TextIO, payload: dict[str, object]) -> None:
    progress.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    progress.flush()


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        return str(exc.code)
    return str(exc)


def _exit_code(exc: BaseException) -> int:
    if isinstance(exc, SystemExit) and isinstance(exc.code, int):
        return exc.code or 1
    return 1


def _write_driver_error(
    *,
    artifact_dir: Path,
    progress: TextIO,
    stage: str,
    exc: BaseException,
    decider: Path,
) -> None:
    payload: dict[str, object] = {
        "event": "driver_error",
        "stage": stage,
        "error": _error_text(exc),
        "decider": str(decider),
    }
    _write_progress(progress, payload)
    (artifact_dir / "driver-error.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _last_progress_error(progress_path: Path) -> str | None:
    error = None
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("event") == "error":
            value = payload.get("error")
            if value:
                error = str(value)
    return error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    _prepend_sys_path(Path(__file__).resolve().parent)
    _prepend_sys_path(repo_root)

    from clawcodex_ext.debug.repl_pty_session import run_adaptive_jsonl

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.artifact_dir / "adaptive-driver.jsonl"
    with progress_path.open("w", encoding="utf-8") as progress:
        try:
            module = _load_decider(args.decider.resolve())
            first_request = _callable(module, "first_request")
            decide_next = _callable(module, "decide_next")

            first = _request_from_decider(first_request(), "first_request()")
            if first.get("op") != "start":
                raise SystemExit(
                    'first_request() must return {"op": "start", ...}. '
                    "put the first user turn in decide_next(response) after ready."
                )
        except (Exception, SystemExit) as exc:
            _write_driver_error(
                artifact_dir=args.artifact_dir,
                progress=progress,
                stage="preflight",
                exc=exc,
                decider=args.decider.resolve(),
            )
            print(_error_text(exc), file=sys.stderr)
            return _exit_code(exc)

        _write_progress(
            progress,
            {
                "event": "decider_request",
                "source": "first_request()",
                "request": _json_safe(first),
            },
        )

        def wrapped_decide(response: dict[str, object]) -> dict[str, object] | None:
            request = decide_next(response)
            if request is None:
                return None
            checked = _request_from_decider(request, "decide_next(response)")
            _write_progress(
                progress,
                {
                    "event": "decider_request",
                    "source": "decide_next(response)",
                    "basis": _response_basis(response),
                    "request": _json_safe(checked),
                },
            )
            if response.get("event") == "stopped" and checked.get("op") == "stop":
                _write_progress(
                    progress,
                    {
                        "event": "decider_warning",
                        "source": "decide_next(response)",
                        "basis": _response_basis(response),
                        "ignored_request": _json_safe(checked),
                        "message": ("ignored duplicate stop after stopped; return None after a stopped response"),
                    },
                )
                return None
            return checked

        try:
            rc = run_adaptive_jsonl(
                first_request=first,
                decide_next=wrapped_decide,
                stdout=_TeeStdout(sys.stdout, progress),
                artifact_dir=args.artifact_dir,
                timeout=args.timeout,
                max_turns=args.max_turns,
            )
            if rc != 0 and not (args.artifact_dir / "driver-error.json").exists():
                progress.flush()
                error = _last_progress_error(progress_path) or (f"adaptive driver exited with code {rc}")
                _write_driver_error(
                    artifact_dir=args.artifact_dir,
                    progress=progress,
                    stage="run",
                    exc=RuntimeError(error),
                    decider=args.decider.resolve(),
                )
            return rc
        except Exception as exc:
            _write_driver_error(
                artifact_dir=args.artifact_dir,
                progress=progress,
                stage="run",
                exc=exc,
                decider=args.decider.resolve(),
            )
            print(_error_text(exc), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
