#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Pin Spec-Audit inputs and create a new, disposable audit run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import stat
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


MAX_REMOTE_BYTES = 100 * 1024 * 1024
REMOTE_TIMEOUT_SECONDS = 30
SCHEMA_VERSION = 1
VCS_NAMES = {".git", ".hg", ".svn"}


class PrepareError(Exception):
    """An input or preparation error that should produce exit code 2."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _safe_name(value: str, fallback: str) -> str:
    name = Path(value).name
    cleaned = "".join(character if character.isalnum() or character in ".-_" else "_" for character in name)
    return cleaned or fallback


def _pin_file(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise PrepareError(f"specification path must not be a symlink: {source}")
    data = source.read_bytes()
    _write_bytes(destination, data)
    return {
        "id": "",
        "kind": "file",
        "source": str(source.resolve()),
        "pinned_path": str(destination.resolve()),
        "sha256": _sha256(data),
        "size": len(data),
    }


def _pin_directory(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise PrepareError(f"specification directory must not be a symlink: {source}")
    files: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    def reject_walk_error(error: OSError) -> None:
        raise PrepareError(f"cannot enumerate specification directory {source}: {error}") from error

    for current, directories, filenames in os.walk(
        source,
        topdown=True,
        onerror=reject_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in [*directories, *filenames]:
            candidate = current_path / name
            if candidate.is_symlink():
                relative = candidate.relative_to(source).as_posix()
                raise PrepareError(f"specification directory contains a symlink: {relative}")
        directories[:] = sorted(directories)
        relative_directory = current_path.relative_to(source)
        pinned_directory = destination / relative_directory
        pinned_directory.mkdir(parents=True, exist_ok=True)
        for directory_name in directories:
            relative = (relative_directory / directory_name).as_posix()
            pinned = destination / relative
            pinned.mkdir(parents=True, exist_ok=True)
            entries.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": _mode(pinned.lstat()),
                }
            )
        for filename in sorted(filenames):
            candidate = current_path / filename
            relative = candidate.relative_to(source).as_posix()
            if not candidate.is_file():
                raise PrepareError(f"unsupported specification entry: {candidate}")
            data = candidate.read_bytes()
            pinned = destination / relative
            _write_bytes(pinned, data)
            file_record = {
                "path": relative,
                "pinned_path": str(pinned.resolve()),
                "sha256": _sha256(data),
                "size": len(data),
            }
            files.append(file_record)
            entries.append(
                {
                    **file_record,
                    "kind": "file",
                    "mode": _mode(pinned.lstat()),
                }
            )
    if not files:
        raise PrepareError(f"specification directory contains no regular files: {source}")
    entries.sort(key=lambda item: item["path"])
    files.sort(key=lambda item: item["path"])
    content_manifest = [
        {key: item[key] for key in ("path", "kind", "mode", "sha256", "size") if key in item} for item in entries
    ]
    manifest = json.dumps(
        content_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "id": "",
        "kind": "directory",
        "source": str(source.resolve()),
        "pinned_path": str(destination.resolve()),
        "sha256": _sha256(manifest),
        "mode": _mode(destination.lstat()),
        "files": files,
        "entries": entries,
    }


def _fetch_url(source: str) -> tuple[bytes, str]:
    """Fetch once in a daemon worker so the caller has a total wall-clock bound."""

    result: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        request = Request(source, headers={"User-Agent": "spec-audit/1"})
        try:
            with urlopen(request, timeout=REMOTE_TIMEOUT_SECONDS) as response:  # noqa: S310
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as error:
                        raise PrepareError("remote specification has an invalid Content-Length") from error
                    if declared_size < 0 or declared_size > MAX_REMOTE_BYTES:
                        raise PrepareError(f"remote specification exceeds {MAX_REMOTE_BYTES} bytes")
                data = response.read(MAX_REMOTE_BYTES + 1)
                resolved_url = response.geturl()
            if len(data) > MAX_REMOTE_BYTES:
                raise PrepareError(f"remote specification exceeds {MAX_REMOTE_BYTES} bytes")
            if urlparse(resolved_url).scheme.lower() not in {"http", "https"}:
                raise PrepareError(f"redirected specification has unsupported URL scheme: {resolved_url}")
            result.put(("ok", (data, resolved_url)))
        except Exception as error:  # marshal worker errors back to the caller
            result.put(("error", error))

    thread = threading.Thread(target=worker, daemon=True, name="spec-audit-url-pin")
    thread.start()
    thread.join(REMOTE_TIMEOUT_SECONDS)
    if thread.is_alive():
        raise PrepareError(f"specification URL fetch exceeded {REMOTE_TIMEOUT_SECONDS} seconds")
    try:
        status, payload = result.get_nowait()
    except queue.Empty as error:
        raise PrepareError("specification URL fetch ended without a result") from error
    if status == "error":
        if isinstance(payload, PrepareError):
            raise payload
        if isinstance(payload, (HTTPError, URLError, OSError, TimeoutError)):
            raise PrepareError(f"could not fetch specification URL: {payload}") from payload
        if isinstance(payload, Exception):
            raise PrepareError(f"could not fetch specification URL: {payload}") from payload
        raise PrepareError("could not fetch specification URL")
    data, resolved_url = payload  # type: ignore[misc]
    return data, resolved_url


def _pin_url(source: str, destination: Path) -> dict[str, Any]:
    data, resolved_url = _fetch_url(source)
    _write_bytes(destination, data)
    return {
        "id": "",
        "kind": "url",
        "source": source,
        "resolved_url": resolved_url,
        "pinned_path": str(destination.resolve()),
        "sha256": _sha256(data),
        "size": len(data),
    }


def _pin_spec(source: str, destination_root: Path, number: int) -> dict[str, Any]:
    parsed = urlparse(source)
    if parsed.scheme.lower() in {"http", "https"}:
        suffix = Path(parsed.path).suffix or ".txt"
        pinned = _pin_url(
            source,
            destination_root / f"{number:03d}-remote{suffix}",
        )
    else:
        path = Path(source).expanduser()
        if not path.exists():
            if parsed.scheme:
                raise PrepareError(f"unsupported specification URL scheme: {parsed.scheme}")
            raise PrepareError(
                f"specification path does not exist; materialize pasted text as a file before running: {source}"
            )
        if path.is_symlink():
            raise PrepareError(f"specification path must not be a symlink: {path}")
        resolved = path.resolve()
        if resolved.is_file():
            pinned = _pin_file(
                resolved,
                destination_root / f"{number:03d}-{_safe_name(resolved.name, 'spec')}",
            )
        elif resolved.is_dir():
            pinned = _pin_directory(
                resolved,
                destination_root / f"{number:03d}-{_safe_name(resolved.name, 'spec')}",
            )
        else:
            raise PrepareError(f"unsupported specification path: {resolved}")
    pinned["id"] = f"SPEC-{number:03d}"
    return pinned


def _mode(metadata: os.stat_result) -> int:
    return stat.S_IMODE(metadata.st_mode)


def _entry(
    path: str,
    kind: str,
    status: str,
    size: int,
    mode: int,
    **extra: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "status": status,
        "size": size,
        "mode": mode,
    }
    value.update(extra)
    return value


def _scan_repository(
    repository: Path,
    exclusions: dict[str, str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise PrepareError(f"cannot enumerate repository path {directory}: {error}") from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(repository).as_posix()
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise PrepareError(f"cannot stat repository path {relative}: {error}") from error

            if child.is_symlink():
                entries.append(
                    _entry(
                        relative,
                        "symlink",
                        "excluded",
                        metadata.st_size,
                        _mode(metadata),
                        reason="symlink-not-followed",
                        target=os.readlink(path),
                    )
                )
                continue

            reason = exclusions.get(relative)
            if reason is None and child.name in VCS_NAMES:
                reason = "version-control-metadata"
            if reason is not None:
                kind = "directory" if child.is_dir(follow_symlinks=False) else "file"
                entries.append(_entry(relative, kind, "excluded", 0, _mode(metadata), reason=reason))
                continue

            if child.is_dir(follow_symlinks=False):
                entries.append(_entry(relative, "directory", "included", 0, _mode(metadata)))
                visit(path)
            elif child.is_file(follow_symlinks=False):
                entries.append(
                    _entry(
                        relative,
                        "file",
                        "included",
                        metadata.st_size,
                        _mode(metadata),
                        sha256=_file_sha256(path),
                    )
                )
            else:
                entries.append(
                    _entry(
                        relative,
                        "special",
                        "excluded",
                        metadata.st_size,
                        _mode(metadata),
                        reason="unsupported-file-type",
                    )
                )

    visit(repository)
    entries.sort(key=lambda item: item["path"])
    return entries


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _reserve_output(requested: Path) -> Path:
    requested.parent.mkdir(parents=True, exist_ok=True)
    candidate = requested
    timestamped_base: Path | None = None
    counter = 1
    while True:
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            if timestamped_base is None:
                timestamped_base = requested.with_name(f"{requested.name}-{_timestamp()}")
                candidate = timestamped_base
            else:
                counter += 1
                candidate = timestamped_base.with_name(f"{timestamped_base.name}-{counter}")
        except OSError as error:
            raise PrepareError(f"cannot reserve output directory {candidate}: {error}") from error


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _initial_report(
    repository: Path,
    repository_pin_sha256: str,
    specifications: list[dict[str, Any]],
) -> str:
    specification_blocks: list[str] = []
    for item in specifications:
        source_id = item["id"]
        lines = [f"- Specification `{source_id}`: {item['source']}; SHA-256: `{item['sha256']}`"]
        if item["kind"] == "directory":
            raw_members = [(member["path"], member["size"], member["sha256"]) for member in item["files"]]
        else:
            raw_members = [("document", item["size"], item["sha256"])]
        for number, (identity, size, digest) in enumerate(raw_members, start=1):
            encoded_identity = quote(str(identity), safe="/-._~")
            lines.append(
                f"  - Member `{source_id}/M-{number:03d}`: `{encoded_identity}`; Bytes: {size}; SHA-256: `{digest}`"
            )
        specification_blocks.append("\n".join(lines))
    specification_lines = "\n".join(specification_blocks)
    return f"""# Spec-Audit Report

## Status

- Contract: Lean v1
- Result: Partial
- Meaning: Audit preparation completed; semantic discovery has not started.

## Pinned Inputs

- Repository: `{repository}`; Pin SHA-256: `{repository_pin_sha256}`
{specification_lines}

## Execution Mode

- Mode: Not selected
- Scheduling: Serial
- Model policy: Host-configured policy, unchanged

## Probe Preflight

- Command: `None`
- Anchor: `None`
- Anchor verification: Not applicable
- Bound: Not executed
- Execution: Not executed
- Reachability: Not reached
- Reason: Probe Preflight has not started; no command has executed.

## Coverage

- Specifications oriented: 0/{len(specifications)}
- Discovery packages completed: 0/{len(specifications)}
- Repository scope inspected: None.
- Search strategy executed: None.
- Unchecked or bounded scope: All semantic discovery remains.

## Findings (0)

None.

## Specification Conflicts (0)

None.

## Uncertain and Unfinished Work

- Specification orientation, repository discovery, counter-search, and adversarial review have not started.

## Limitations

- Complete means the declared audit procedure finished, not that all possible inconsistencies were exhaustively disproved.
- This initial Partial report does not establish consistency or absence of further problems.

## Validation

- Result: Not completed
- Inventory verification: Not completed
- Report lint: Not completed
"""


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(args.repo).expanduser().resolve()
    if not repository.is_dir():
        raise PrepareError(f"repository is not a directory: {repository}")
    if args.budget_seconds is not None and args.budget_seconds <= 0:
        raise PrepareError("budget must be greater than zero")

    requested_output = (Path(args.output).expanduser() if args.output else Path.cwd() / "spec-audit-report").resolve()
    work = Path(tempfile.mkdtemp(prefix="spec-audit-run-"))
    output: Path | None = None
    try:
        specifications = [
            _pin_spec(source, work / "specifications", number) for number, source in enumerate(args.spec, start=1)
        ]
        specification_pin_file = work / "specification-pin.json"
        _atomic_json(
            specification_pin_file,
            {
                "schema_version": SCHEMA_VERSION,
                "specifications": specifications,
            },
        )
        specification_pin_sha256 = _file_sha256(specification_pin_file)
        output = _reserve_output(requested_output)
        exclusions: list[dict[str, str]] = []
        if _inside(output, repository):
            exclusions.append(
                {
                    "path": output.relative_to(repository).as_posix(),
                    "reason": "audit-output",
                }
            )
        if _inside(work, repository):
            exclusions.append(
                {
                    "path": work.relative_to(repository).as_posix(),
                    "reason": "audit-work",
                }
            )
        exclusion_map = {item["path"]: item["reason"] for item in exclusions}
        pin_file = work / "repository-pin.json"
        pin = {
            "schema_version": SCHEMA_VERSION,
            "repository": {"path": str(repository)},
            "entries": _scan_repository(repository, exclusion_map),
        }
        _atomic_json(pin_file, pin)
        pin_sha256 = _file_sha256(pin_file)

        run_file = work / "run.json"
        candidates_dir = work / "candidates"
        candidates_dir.mkdir()
        run: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "resumable": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": {
                "path": str(repository),
                "pin_file": str(pin_file),
                "pin_sha256": pin_sha256,
            },
            "specifications": specifications,
            "specification_pin": {
                "path": str(specification_pin_file.resolve()),
                "sha256": specification_pin_sha256,
            },
            "output_dir": str(output),
            "report_file": str(output / "report.md"),
            "findings_dir": str(output / "findings"),
            "work_dir": str(work),
            "candidates_dir": str(candidates_dir),
            "run_file": str(run_file),
            "inventory_file": str(work / "inventory.json"),
            "budget_seconds": args.budget_seconds,
            "excluded_paths": exclusions,
        }
        _atomic_json(run_file, run)
        (output / "findings").mkdir()
        (output / "report.md").write_text(
            _initial_report(repository, pin_sha256, specifications),
            encoding="utf-8",
        )
        return {
            "run_file": str(run_file),
            "output_dir": str(output),
            "report_file": str(output / "report.md"),
            "candidates_dir": str(candidates_dir),
            "repository": str(repository),
            "specifications": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "kind": item["kind"],
                    "sha256": item["sha256"],
                }
                for item in specifications
            ],
        }
    except Exception:
        if output is not None:
            shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.getcwd(),
        help="repository path; defaults to the current directory",
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="confirmed existing file/directory or HTTP(S) URL",
    )
    parser.add_argument(
        "--output",
        help="report directory; existing paths are never overwritten",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        help="known external audit budget",
    )
    return parser


def main() -> int:
    try:
        result = _prepare(_parser().parse_args())
    except PrepareError as error:
        print(f"prepare_audit: {error}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"prepare_audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
