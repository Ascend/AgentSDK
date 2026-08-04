#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Publish or verify the immutable repository pin created at audit start."""

# This bundled executable is intentionally self-contained so the skill can copy
# and run it without depending on ClawCodex package internals.
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from functools import partial
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1
VCS_NAMES = {".git", ".hg", ".svn"}


class InventoryError(Exception):
    """An invalid run, pin, or inventory state that produces exit code 2."""


class FinalizationRequired(Exception):
    """A known audit budget has entered its reserved finalization phase."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise InventoryError(f"JSON root must be an object: {path}")
    return value


def _load_run(path: Path) -> tuple[dict[str, Any], Path, Path]:
    path = path.expanduser().resolve()
    run = _load_json(path)
    if run.get("schema_version") != SCHEMA_VERSION or run.get("resumable") is not False:
        raise InventoryError("unsupported or resumable run file")
    repository_value = run.get("repository")
    if not isinstance(repository_value, dict):
        raise InventoryError("run file has no repository object")
    if not all(isinstance(repository_value.get(key), str) for key in ("path", "pin_file", "pin_sha256")):
        raise InventoryError("run file has no complete repository pin identity")
    repository = Path(repository_value["path"]).resolve()
    if not repository.is_dir():
        raise InventoryError(f"repository is not a directory: {repository}")
    return run, path, repository


def _load_pin(run: dict[str, Any], repository: Path) -> tuple[dict[str, Any], str]:
    repository_value = run["repository"]
    pin_file = Path(repository_value["pin_file"]).expanduser().resolve()
    expected_sha256 = repository_value["pin_sha256"]
    try:
        pin_bytes = pin_file.read_bytes()
    except OSError as error:
        raise InventoryError(f"cannot read repository pin {pin_file}: {error}") from error
    actual_sha256 = hashlib.sha256(pin_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise InventoryError("repository pin SHA-256 does not match run.json")
    try:
        pin = json.loads(pin_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"repository pin is not valid UTF-8 JSON: {error}") from error
    if not isinstance(pin, dict) or pin.get("schema_version") != SCHEMA_VERSION:
        raise InventoryError("unsupported repository pin schema")
    pinned_repository = pin.get("repository")
    if not isinstance(pinned_repository, dict) or pinned_repository.get("path") != str(repository):
        raise InventoryError("repository pin path does not match run.json")
    entries = pin.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise InventoryError("repository pin entries must be a list of objects")
    paths = [item.get("path") for item in entries]
    if not all(isinstance(path, str) for path in paths):
        raise InventoryError("repository pin entries require string paths")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise InventoryError("repository pin paths must be unique and sorted")
    return pin, actual_sha256


def _path_ancestor_state(path: Path) -> str:
    for ancestor in reversed(path.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            return "missing"
        except OSError as error:
            raise InventoryError(f"cannot inspect path ancestor {ancestor}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            return "symlink"
        if not stat.S_ISDIR(metadata.st_mode):
            return "wrong-type"
    return "clean"


def _load_specification_pin(run: dict[str, Any]) -> list[dict[str, Any]]:
    identity = run.get("specification_pin")
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("path"), str)
        or not isinstance(identity.get("sha256"), str)
    ):
        raise InventoryError("run file has no complete specification pin identity")
    pin_file = Path(identity["path"]).expanduser()
    if not pin_file.is_absolute():
        raise InventoryError("specification pin path must be absolute")
    if _path_ancestor_state(pin_file) != "clean":
        raise InventoryError("specification pin path has an unavailable or symlink ancestor")
    try:
        metadata = pin_file.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise InventoryError("specification pin is not a regular file")
        pin_bytes = pin_file.read_bytes()
    except OSError as error:
        raise InventoryError(f"cannot read specification pin {pin_file}: {error}") from error
    actual_sha256 = hashlib.sha256(pin_bytes).hexdigest()
    if actual_sha256 != identity["sha256"]:
        raise InventoryError("specification pin SHA-256 does not match run.json")
    try:
        pin = json.loads(pin_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"specification pin is not valid UTF-8 JSON: {error}") from error
    if not isinstance(pin, dict) or pin.get("schema_version") != SCHEMA_VERSION:
        raise InventoryError("unsupported specification pin schema")
    specifications = pin.get("specifications")
    if not isinstance(specifications, list) or not all(isinstance(item, dict) for item in specifications):
        raise InventoryError("specification pin entries must be a list of objects")
    return specifications


def _validate_byte_specification(item: dict[str, Any]) -> tuple[str, Path, str, int]:
    source_id = item.get("id")
    pinned_path = item.get("pinned_path")
    digest = item.get("sha256")
    size = item.get("size")
    invalid_source = not isinstance(source_id, str) or not source_id or not isinstance(pinned_path, str)
    invalid_digest = not isinstance(digest, str) or len(digest) != 64
    invalid_size = not isinstance(size, int) or isinstance(size, bool) or size < 0
    if invalid_source or invalid_digest or invalid_size:
        raise InventoryError("specification pin has an invalid byte-source entry")
    path = Path(pinned_path).expanduser()
    if not path.is_absolute():
        raise InventoryError("pinned specification path must be absolute")
    return source_id, path, digest, size


def _relative_specification_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryError("specification member path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise InventoryError(f"invalid specification member path: {value!r}")
    return value


def _directory_manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in ("path", "kind", "mode", "sha256", "size") if key in entry}


def _validate_directory_specification(
    item: dict[str, Any],
) -> tuple[str, Path, int, list[dict[str, Any]]]:
    source_id = item.get("id")
    pinned_path = item.get("pinned_path")
    expected_sha256 = item.get("sha256")
    expected_mode = item.get("mode")
    raw_entries = item.get("entries")
    raw_files = item.get("files")
    invalid_source = not isinstance(source_id, str) or not source_id or not isinstance(pinned_path, str)
    invalid_digest = not isinstance(expected_sha256, str) or len(expected_sha256) != 64
    invalid_mode = not isinstance(expected_mode, int) or isinstance(expected_mode, bool)
    invalid_entries = not isinstance(raw_entries, list) or not all(isinstance(entry, dict) for entry in raw_entries)
    if invalid_source or invalid_digest or invalid_mode or invalid_entries or not isinstance(raw_files, list):
        raise InventoryError("specification pin has an invalid directory entry")
    root = Path(pinned_path).expanduser()
    if not root.is_absolute():
        raise InventoryError("pinned specification directory path must be absolute")

    entries: list[dict[str, Any]] = []
    expected_files: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        relative = _relative_specification_path(raw_entry.get("path"))
        kind = raw_entry.get("kind")
        mode = raw_entry.get("mode")
        if kind not in {"directory", "file"} or not isinstance(mode, int) or isinstance(mode, bool):
            raise InventoryError("specification directory member has invalid type metadata")
        entry: dict[str, Any] = {"path": relative, "kind": kind, "mode": mode}
        if kind == "file":
            member_path = raw_entry.get("pinned_path")
            digest = raw_entry.get("sha256")
            size = raw_entry.get("size")
            invalid_member = not isinstance(member_path, str) or Path(member_path) != root / relative
            invalid_digest = not isinstance(digest, str) or len(digest) != 64
            invalid_size = not isinstance(size, int) or isinstance(size, bool) or size < 0
            if invalid_member or invalid_digest or invalid_size:
                raise InventoryError("specification file member has invalid pin metadata")
            entry.update(
                {
                    "pinned_path": member_path,
                    "sha256": digest,
                    "size": size,
                }
            )
            expected_files.append(
                {
                    "path": relative,
                    "pinned_path": member_path,
                    "sha256": digest,
                    "size": size,
                }
            )
        entries.append(entry)

    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise InventoryError("specification directory member paths must be unique and sorted")
    if raw_files != expected_files:
        raise InventoryError("specification directory files differ from its member manifest")
    manifest = json.dumps(
        [_directory_manifest_entry(entry) for entry in entries],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(manifest).hexdigest() != expected_sha256:
        raise InventoryError("specification directory manifest SHA-256 is inconsistent")
    return source_id, root, expected_mode, entries


def _scan_specification_directory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise InventoryError(f"cannot enumerate pinned specification directory {directory}: {error}") from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise InventoryError(f"cannot stat pinned specification member {relative}: {error}") from error
            mode = _mode(metadata)
            if child.is_symlink():
                try:
                    target = os.readlink(path)
                except OSError as error:
                    raise InventoryError(f"cannot read pinned specification symlink {relative}: {error}") from error
                entries.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "mode": mode,
                        "size": metadata.st_size,
                        "target": target,
                    }
                )
            elif child.is_dir(follow_symlinks=False):
                entries.append({"path": relative, "kind": "directory", "mode": mode})
                visit(path)
            elif child.is_file(follow_symlinks=False):
                try:
                    digest = _file_sha256(path)
                except OSError as error:
                    raise InventoryError(f"cannot hash pinned specification member {relative}: {error}") from error
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "mode": mode,
                        "size": metadata.st_size,
                        "sha256": digest,
                    }
                )
            else:
                entries.append(
                    {
                        "path": relative,
                        "kind": "special",
                        "mode": mode,
                        "size": metadata.st_size,
                    }
                )

    visit(root)
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _specification_entry_identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("kind"),
        entry.get("mode"),
        entry.get("size"),
        entry.get("sha256"),
    )


def _verify_directory_specification(
    item: dict[str, Any],
) -> tuple[str, list[str], list[str], list[str]]:
    source_id, root, expected_mode, expected_entries = _validate_directory_specification(item)
    ancestor_state = _path_ancestor_state(root)
    if ancestor_state == "missing":
        return source_id, [], [source_id], []
    if ancestor_state != "clean":
        return source_id, [], [], [source_id]
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return source_id, [], [source_id], []
    except OSError as error:
        raise InventoryError(f"cannot stat pinned specification directory {source_id}: {error}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        return source_id, [], [], [source_id]

    current_entries = _scan_specification_directory(root)
    old = {entry["path"]: entry for entry in expected_entries}
    new = {entry["path"]: entry for entry in current_entries}
    added = [f"{source_id}/{path}" for path in sorted(new.keys() - old.keys())]
    removed = [f"{source_id}/{path}" for path in sorted(old.keys() - new.keys())]
    changed = [
        f"{source_id}/{path}"
        for path in sorted(old.keys() & new.keys())
        if _specification_entry_identity(old[path]) != _specification_entry_identity(new[path])
    ]
    if _mode(root_metadata) != expected_mode:
        changed.insert(0, source_id)
    return source_id, added, removed, changed


def _verify_specifications(run: dict[str, Any]) -> dict[str, Any]:
    pinned = _load_specification_pin(run)
    metadata_changed = run.get("specifications") != pinned
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    seen_ids: set[str] = set()

    for item in pinned:
        kind = item.get("kind")
        if kind == "directory":
            source_id, member_added, member_removed, member_changed = _verify_directory_specification(item)
            added.extend(member_added)
            removed.extend(member_removed)
            changed.extend(member_changed)
        elif kind in {"file", "url", "pasted", "text"}:
            source_id, path, expected_sha256, expected_size = _validate_byte_specification(item)
            ancestor_state = _path_ancestor_state(path)
            if ancestor_state == "missing":
                removed.append(source_id)
                metadata = None
            elif ancestor_state != "clean":
                changed.append(source_id)
                metadata = None
            else:
                try:
                    metadata = path.lstat()
                except FileNotFoundError:
                    removed.append(source_id)
                    metadata = None
                except OSError as error:
                    raise InventoryError(f"cannot stat pinned specification {source_id}: {error}") from error
            if metadata is not None:
                if not stat.S_ISREG(metadata.st_mode):
                    changed.append(source_id)
                else:
                    try:
                        actual_sha256 = _file_sha256(path)
                    except OSError as error:
                        raise InventoryError(f"cannot hash pinned specification {source_id}: {error}") from error
                    if metadata.st_size != expected_size or actual_sha256 != expected_sha256:
                        changed.append(source_id)
        else:
            raise InventoryError(f"unsupported pinned specification kind: {kind!r}")
        if source_id in seen_ids:
            raise InventoryError("specification pin IDs must be unique")
        seen_ids.add(source_id)

    return {
        "metadata_changed": metadata_changed,
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
    }


def _relative_if_inside(path: Path, repository: Path) -> str | None:
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None


def _exclusions(run: dict[str, Any], repository: Path) -> dict[str, str]:
    exclusions: dict[str, str] = {}
    configured = run.get("excluded_paths", [])
    if not isinstance(configured, list):
        raise InventoryError("excluded_paths must be a list")
    for item in configured:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("reason"), str)
        ):
            raise InventoryError("each excluded path requires path and reason strings")
        relative = Path(item["path"]).as_posix().strip("/")
        reason = item["reason"].strip()
        if not relative or not reason:
            raise InventoryError("excluded path and reason must not be empty")
        exclusions[relative] = reason

    for key, reason in (("output_dir", "audit-output"), ("work_dir", "audit-work")):
        value = run.get(key)
        if isinstance(value, str):
            relative = _relative_if_inside(Path(value), repository)
            if relative:
                exclusions[relative] = reason
    return exclusions


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


def _scan(repository: Path, exclusions: dict[str, str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise InventoryError(f"cannot enumerate {directory}: {error}") from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(repository).as_posix()
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise InventoryError(f"cannot stat {relative}: {error}") from error

            if child.is_symlink():
                try:
                    target = os.readlink(path)
                except OSError as error:
                    raise InventoryError(f"cannot read symlink {relative}: {error}") from error
                entries.append(
                    _entry(
                        relative,
                        "symlink",
                        "excluded",
                        metadata.st_size,
                        _mode(metadata),
                        reason="symlink-not-followed",
                        target=target,
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
                try:
                    digest = _file_sha256(path)
                except OSError as error:
                    raise InventoryError(f"cannot hash {relative}: {error}") from error
                entries.append(
                    _entry(
                        relative,
                        "file",
                        "included",
                        metadata.st_size,
                        _mode(metadata),
                        sha256=digest,
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


def _inventory_path(run: dict[str, Any], run_file: Path) -> Path:
    configured = run.get("inventory_file")
    if isinstance(configured, str):
        return Path(configured).expanduser().resolve()
    return run_file.parent / "inventory.json"


def _summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_entries": len(entries),
        "included_entries": sum(entry["status"] == "included" for entry in entries),
        "included_files": sum(entry["status"] == "included" and entry["kind"] == "file" for entry in entries),
        "included_directories": sum(
            entry["status"] == "included" and entry["kind"] == "directory" for entry in entries
        ),
        "excluded_entries": sum(entry["status"] == "excluded" for entry in entries),
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise InventoryError(f"inventory already exists and cannot be rebased: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _build(run: dict[str, Any], run_file: Path, repository: Path) -> dict[str, Any]:
    pin, pin_sha256 = _load_pin(run, repository)
    inventory_file = _inventory_path(run, run_file)
    entries = pin["entries"]
    document = {
        "schema_version": SCHEMA_VERSION,
        "repository": {"path": str(repository)},
        "repository_pin_sha256": pin_sha256,
        "summary": _summary(entries),
        "entries": entries,
    }
    _write_json_exclusive(inventory_file, document)
    return {
        "status": "built",
        "inventory_file": str(inventory_file),
        "repository": str(repository),
        "summary": document["summary"],
        "specification_ids": [
            item.get("id")
            for item in run.get("specifications", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ],
    }


def _identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("kind"),
        entry.get("size"),
        entry.get("mode"),
        entry.get("sha256"),
    )


def _verify(
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> tuple[int, dict[str, Any]]:
    pin, pin_sha256 = _load_pin(run, repository)
    inventory_file = _inventory_path(run, run_file)
    inventory = _load_json(inventory_file)
    if inventory.get("repository_pin_sha256") != pin_sha256:
        raise InventoryError("inventory does not identify this repository pin")
    if inventory.get("entries") != pin.get("entries"):
        raise InventoryError("inventory contents differ from the immutable repository pin")

    specification_drift = _verify_specifications(run)
    current_entries = _scan(repository, _exclusions(run, repository))
    old = {item["path"]: item for item in pin["entries"] if item.get("status") == "included"}
    new = {item["path"]: item for item in current_entries if item.get("status") == "included"}
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    changed = sorted(path for path in old.keys() & new.keys() if _identity(old[path]) != _identity(new[path]))
    drift = bool(
        added
        or removed
        or changed
        or specification_drift["metadata_changed"]
        or specification_drift["added"]
        or specification_drift["removed"]
        or specification_drift["changed"]
    )
    return (
        1 if drift else 0,
        {
            "status": "drift" if drift else "clean",
            "inventory_file": str(inventory_file),
            "repository_pin_sha256": pin_sha256,
            "added": added,
            "removed": removed,
            "changed": changed,
            "specifications": specification_drift,
        },
    )


def _compile_pattern(value: str, label: str) -> re.Pattern[str]:
    try:
        return re.compile(value, re.IGNORECASE)
    except re.error as error:
        raise InventoryError(f"invalid {label} regex: {error}") from error


def _repository_files(run: dict[str, Any], run_file: Path, repository: Path) -> list[tuple[str, Path]]:
    inventory = _load_json(_inventory_path(run, run_file))
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise InventoryError("inventory entries are unavailable; run build first")
    result: list[tuple[str, Path]] = []
    for entry in entries:
        if (
            isinstance(entry, dict)
            and entry.get("status") == "included"
            and entry.get("kind") == "file"
            and isinstance(entry.get("path"), str)
        ):
            relative = entry["path"]
            result.append((relative, repository / relative))
    return result


def _specification_files(run: dict[str, Any]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for item in _load_specification_pin(run):
        source_id = item.get("id")
        if not isinstance(source_id, str):
            raise InventoryError("specification source has no id")
        if item.get("kind") == "directory":
            files = item.get("files")
            if not isinstance(files, list):
                raise InventoryError("directory specification has no files")
            for number, member in enumerate(files, start=1):
                if not isinstance(member, dict) or not isinstance(member.get("pinned_path"), str):
                    raise InventoryError("directory specification member is invalid")
                result.append((f"{source_id}/M-{number:03d}", Path(member["pinned_path"])))
        else:
            pinned_path = item.get("pinned_path")
            if not isinstance(pinned_path, str):
                raise InventoryError("specification source has no pinned path")
            result.append((f"{source_id}/M-001", Path(pinned_path)))
    return result


def _query_files(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> list[tuple[str, Path]]:
    if args.scope == "repository":
        files = _repository_files(run, run_file, repository)
    else:
        files = _specification_files(run)
    path_pattern = _compile_pattern(args.path_pattern, "path")
    return [(identity, path) for identity, path in files if path_pattern.search(identity)]


def _paths(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> dict[str, Any]:
    _require_discovery_phase(run)
    files = _query_files(args, run, run_file, repository)
    start = args.cursor
    page = files[start : start + args.limit]
    next_cursor = start + len(page)
    return {
        "scope": args.scope,
        "paths": [identity for identity, _path in page],
        "next_cursor": None if next_cursor >= len(files) else next_cursor,
        "complete": next_cursor >= len(files),
        "matched_paths": len(files),
    }


def _iter_text_lines(path: Path):
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8192)
            if b"\x00" in prefix:
                return
            stream.seek(0)
            for number, raw in enumerate(stream, start=1):
                yield number, raw.decode("utf-8", errors="replace").rstrip("\r\n")
    except OSError as error:
        raise InventoryError(f"cannot read query source {path}: {error}") from error


def _display_text(text: str, limit: int = 600) -> tuple[str, bool]:
    """Bound a single displayed source line without changing its line identity."""

    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _search(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> dict[str, Any]:
    _require_discovery_phase(run)
    files = _query_files(args, run, run_file, repository)
    pattern = _compile_pattern(args.pattern, "search")
    skipped = args.cursor
    seen = 0
    matches: list[dict[str, Any]] = []
    more = False
    for identity, path in files:
        for line_number, text in _iter_text_lines(path):
            if not pattern.search(text):
                continue
            if seen < skipped:
                seen += 1
                continue
            if len(matches) >= args.limit:
                more = True
                break
            displayed, truncated = _display_text(text)
            record: dict[str, Any] = {
                "path": identity,
                "line": line_number,
                "text": displayed,
            }
            if truncated:
                record["text_truncated"] = True
            matches.append(record)
            seen += 1
        if more:
            break
    return {
        "scope": args.scope,
        "matches": matches,
        "next_cursor": skipped + len(matches) if more else None,
        "complete": not more,
    }


def _read(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> dict[str, Any]:
    _require_discovery_phase(run)
    files = dict(_query_files(args, run, run_file, repository))
    path = files.get(args.path)
    if path is None:
        raise InventoryError(f"query path is not pinned in {args.scope}: {args.path}")
    selected: list[dict[str, Any]] = []
    end = args.start_line + args.lines
    for line_number, text in _iter_text_lines(path):
        if line_number < args.start_line:
            continue
        if line_number >= end:
            break
        displayed, truncated = _display_text(text)
        record: dict[str, Any] = {"line": line_number, "text": displayed}
        if truncated:
            record["text_truncated"] = True
        selected.append(record)
    return {"scope": args.scope, "path": args.path, "lines": selected}


NORMATIVE_PATTERN = re.compile(
    r"\b(?:MUST(?:\s+NOT)?|SHALL(?:\s+NOT)?|REQUIRED|SHOULD(?:\s+NOT)?|"
    r"RECOMMENDED|MAY|OPTIONAL)\b|"
    r"(?:必须|不得|应当|应该|可以|可选|禁止|必須|してはならない|"
    r"推奨|任意)",
    re.IGNORECASE,
)
REQUIREMENT_RISK_PATTERNS = {
    "boundary": re.compile(
        r"\b(?:all|any|each|every|more\s+than|less\s+than|at\s+least|at\s+most|"
        r"maximum|minimum|limit|length|count|multiple|remaining|truncate)\b|\d+"
        r"|(?:全部|每个|每一|至少|至多|最大|最小|限制|长度|数量|剩余|截断)",
        re.IGNORECASE,
    ),
    "state_timing": re.compile(
        r"\b(?:state|transition|timer|timeout|retry|retransmit|delay|random|"
        r"unsolicited|periodic|before|after|until|ordering)\b|"
        r"(?:状态|转换|定时|超时|重试|延迟|随机|周期|之前|之后|直到|顺序|"
        r"状態|遷移|タイマー|再試行)",
        re.IGNORECASE,
    ),
    "routing_traversal": re.compile(
        r"\b(?:dispatch|route|forward|filter|drop|bypass|handler|registration|"
        r"extension|chain|header|parser|input|output)\b|"
        r"(?:分发|路由|转发|过滤|丢弃|处理器|注册|扩展|链|头部|解析|输入|输出)",
        re.IGNORECASE,
    ),
    "capability": re.compile(
        r"\b(?:support|capability|optional|mode|feature|implement|provide|accept|"
        r"generate|send|receive)\b|"
        r"(?:支持|能力|可选|模式|功能|实现|提供|接受|生成|发送|接收|対応|実装|送信|受信)",
        re.IGNORECASE,
    ),
}

REQUIREMENT_RISK_LANES = (
    "boundary",
    "state_timing",
    "routing_traversal",
    "capability",
    "other",
)


def _generic_heading(text: str) -> str | None:
    """Return a format-neutral section label for common specification shapes."""

    stripped = text.strip()
    if not stripped or len(stripped) > 180:
        return None
    markdown = re.fullmatch(r"#{1,6}\s+(\S.*)", stripped)
    if markdown is not None:
        return markdown.group(1).strip()
    numbered = re.fullmatch(
        r"(?:\d+(?:\.\d+)*(?:[.)])?|[A-Z][.)])\s+(\S.*)",
        stripped,
    )
    if numbered is not None and not NORMATIVE_PATTERN.search(stripped) and not stripped.endswith((";", ".")):
        return numbered.group(1).strip()
    json_key = re.fullmatch(r'"([^"\\]{1,100})"\s*:\s*.*', stripped)
    if json_key is not None:
        return json_key.group(1).strip()
    yaml_key = re.fullmatch(
        r"([\w\- .\u0080-\uffff]{1,100})\s*:\s*.*",
        stripped,
    )
    if yaml_key is not None:
        return yaml_key.group(1).strip()
    return None


def _section_contexts(
    lines: list[tuple[int, str]],
) -> dict[int, tuple[int, str]]:
    """Map each line to its nearest preceding generic section heading."""

    current = (0, "<document>")
    contexts: dict[int, tuple[int, str]] = {}
    for number, text in lines:
        heading = _generic_heading(text)
        if heading is not None:
            current = (number, heading)
        contexts[number] = current
    return contexts


def _requirements(
    args: argparse.Namespace,
    run: dict[str, Any],
) -> dict[str, Any]:
    """Rank bounded normative specification leads for one oriented package."""

    _require_discovery_phase(run)
    members = [
        (identity, path) for identity, path in _specification_files(run) if identity.startswith(f"{args.source}/")
    ]
    if not members:
        raise InventoryError(f"unknown pinned specification source: {args.source}")
    for term in args.term or []:
        _reject_bare_short_term_alternatives(term, "requirement term")
    term_pattern = (
        _compile_pattern(
            "|".join(f"(?:{term})" for term in args.term),
            "requirement term",
        )
        if args.term
        else None
    )
    ranked: list[tuple[int, str, int, dict[str, Any]]] = []
    sections: dict[tuple[str, int], tuple[int, str]] = {}
    for identity, path in members:
        lines = list(_iter_text_lines(path))
        contexts = _section_contexts(lines)
        term_lines = [number for number, text in lines if term_pattern.search(text)] if term_pattern is not None else []
        if term_pattern is not None and not term_lines:
            continue
        for number, source_text in lines:
            if not NORMATIVE_PATTERN.search(source_text):
                continue
            distance = min(abs(number - term_line) for term_line in term_lines) if term_lines else None
            if distance is not None and distance > 20:
                continue
            kinds = [name for name, pattern in REQUIREMENT_RISK_PATTERNS.items() if pattern.search(source_text)]
            score = 50
            score += sum(
                {
                    "boundary": 45,
                    "state_timing": 15,
                    "routing_traversal": 25,
                    "capability": 20,
                }[kind]
                for kind in kinds
            )
            if (
                "boundary" in kinds
                and distance is not None
                and re.search(
                    r"\b(?:limit|maximum|minimum|count|length|at\s+least|at\s+most)\b|\d+|"
                    r"(?:限制|最大|最小|数量|长度|至少|至多)",
                    source_text,
                    re.IGNORECASE,
                )
            ):
                # An oriented finite-domain obligation is a stronger defect lead
                # than nearby generic lifecycle prose.
                score += 35
            if re.search(
                r"\b(?:MUST(?:\s+NOT)?|SHALL(?:\s+NOT)?|REQUIRED)\b",
                source_text,
                re.IGNORECASE,
            ):
                score += 30
            if re.search(r"\b(?:MAY|OPTIONAL)\b", source_text, re.IGNORECASE):
                score += 20
            if distance is not None:
                score += 30 if distance == 0 else 20 if distance <= 5 else 10
            displayed, truncated = _display_text(source_text)
            record: dict[str, Any] = {
                "path": identity,
                "line": number,
                "kinds": kinds,
                "text": displayed,
            }
            if re.search(
                r"\b(?:MUST(?:\s+NOT)?|SHALL(?:\s+NOT)?|REQUIRED)\b",
                source_text,
                re.IGNORECASE,
            ):
                record["strength"] = "mandatory"
            elif re.search(
                r"\b(?:SHOULD(?:\s+NOT)?|RECOMMENDED)\b",
                source_text,
                re.IGNORECASE,
            ):
                record["strength"] = "recommended"
            elif re.search(
                r"\b(?:MAY|OPTIONAL)\b",
                source_text,
                re.IGNORECASE,
            ):
                record["strength"] = "optional"
            else:
                record["strength"] = "normative"
            if distance is not None:
                record["term_distance"] = distance
            if truncated:
                record["text_truncated"] = True
            ranked.append((-score, identity, number, record))
            sections[(identity, number)] = contexts[number]
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    matches = [item[3] for item in ranked[: args.limit]]
    risk_lanes: dict[str, dict[str, Any]] = {}
    for lane in REQUIREMENT_RISK_LANES:
        lane_items = [
            item for item in ranked if (lane in item[3]["kinds"]) or (lane == "other" and not item[3]["kinds"])
        ]
        representatives: list[dict[str, Any]] = []
        represented_sections: set[tuple[str, int, str]] = set()
        for _score, identity, number, _record in lane_items:
            heading_line, heading = sections[(identity, number)]
            section_key = (identity, heading_line, heading)
            if section_key in represented_sections:
                continue
            represented_sections.add(section_key)
            representative: dict[str, Any] = {
                "anchor": f"{identity}:{number}",
                "heading": heading,
            }
            if heading_line:
                representative["heading_anchor"] = f"{identity}:{heading_line}"
            representatives.append(representative)
            if len(representatives) == 3:
                break
        risk_lanes[lane] = {
            "matched": len(lane_items),
            "representatives": representatives,
        }
    return {
        "source": args.source,
        "terms": args.term or [],
        "matches": matches,
        "matched_signals": len(ranked),
        "selection_limit": args.limit,
        "risk_lanes": risk_lanes,
        "semantic_verdict": "none; normative lexical leads only",
    }


HOTSPOT_PATTERNS = {
    "boundary": re.compile(
        r"(?:\b(?:max(?:imum)?|min(?:imum)?|limit|count|capacity|length|size|truncate|remaining)\b|"
        r"(?:<=|>=|==|!=|<|>)\s*\d+|\d+\s*(?:<=|>=|==|!=|<|>))",
        re.IGNORECASE,
    ),
    "dispatch": re.compile(
        r"\b(?:switch|case|match|dispatch|route|routing|forward|bridge|offload|filter|drop|bypass|handler|callback|register)\b",
        re.IGNORECASE,
    ),
    "state": re.compile(
        r"\b(?:state|transition|timer|timeout|retry|retransmit|delay|random|unsolicited|pending|expired?)\b",
        re.IGNORECASE,
    ),
    "capability": re.compile(
        r"\b(?:unsupported|not\s+supported|not\s+implemented|unimplemented|disabled|todo|fixme|feature|capability)\b",
        re.IGNORECASE,
    ),
}

HOTSPOT_SIGNAL_MARKERS = {
    "explicit-omission": re.compile(
        r"\b(?:unsupported|not\s+supported|not\s+implemented|unimplemented|"
        r"todo|fixme)\b|\b(?:feature|capability|operation|support)\b[^\n]{0,32}"
        r"\bdisabled\b",
        re.IGNORECASE,
    ),
    "finite-boundary": re.compile(
        r"(?:\b(?:max(?:imum)?|limit|capacity|truncate|remaining)\b|"
        r"(?:<=|>=|==|!=|<|>)\s*\d+|\[[ \t]*\d+[ \t]*\])",
        re.IGNORECASE,
    ),
    "early-exit": re.compile(
        r"\b(?:break|continue|return|goto|drop|discard|truncate)\b",
        re.IGNORECASE,
    ),
    "dispatch-diversion": re.compile(
        r"\b(?:route|routing|forward|filter|drop|bypass|offload|fallback)\b",
        re.IGNORECASE,
    ),
    "state-timing": re.compile(
        r"\b(?:state|transition|timer|timeout|retry|retransmit|delay|random|"
        r"unsolicited|periodic)\b",
        re.IGNORECASE,
    ),
}

MAX_FULL_SCOPE_HOTSPOT_FILES = 5000
MAX_PRIORITY_OMISSIONS = 8
MAX_DIVERSE_PRIORITY_LEADS = 8
MAX_PRIORITY_LEADS = 12
PRIORITY_DIVERSITY_MARKERS = (
    "finite-boundary",
    "dispatch-diversion",
    "early-exit",
    "state-timing",
)


def _priority_lead_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, compact identity needed to revisit a lexical lead."""

    return {
        "lead_id": record["lead_id"],
        "markers": list(record.get("markers", [])),
        "path": record["path"],
        "line": record["line"],
    }


def _select_priority_leads(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep omissions plus a small marker- and file-diverse high-signal set."""

    unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        lead_id = record["lead_id"]
        if lead_id in seen_ids:
            continue
        seen_ids.add(lead_id)
        unique.append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    omissions = [record for record in unique if "explicit-omission" in record.get("markers", [])][
        :MAX_PRIORITY_OMISSIONS
    ]
    for record in omissions:
        selected.append(record)
        selected_ids.add(record["lead_id"])

    diverse: list[dict[str, Any]] = []
    for marker in PRIORITY_DIVERSITY_MARKERS:
        marker_files: set[str] = set()
        for record in unique:
            if len(diverse) >= MAX_DIVERSE_PRIORITY_LEADS:
                break
            if record["lead_id"] in selected_ids:
                continue
            if marker not in record.get("markers", []):
                continue
            if record["path"] in marker_files:
                continue
            marker_files.add(record["path"])
            diverse.append(record)
            selected_ids.add(record["lead_id"])
        if len(diverse) >= MAX_DIVERSE_PRIORITY_LEADS:
            break

    if len(diverse) < MAX_DIVERSE_PRIORITY_LEADS:
        per_file: dict[str, int] = {}
        for record in diverse:
            per_file[record["path"]] = per_file.get(record["path"], 0) + 1
        for record in unique:
            if len(diverse) >= MAX_DIVERSE_PRIORITY_LEADS:
                break
            if record["lead_id"] in selected_ids or not record.get("markers"):
                continue
            if per_file.get(record["path"], 0) >= 2:
                continue
            diverse.append(record)
            selected_ids.add(record["lead_id"])
            per_file[record["path"]] = per_file.get(record["path"], 0) + 1

    return [_priority_lead_summary(record) for record in (selected + diverse)[:MAX_PRIORITY_LEADS]]


def _reject_bare_short_term_alternatives(expression: str, label: str) -> None:
    """Reject noisy bare acronyms while still allowing explicitly bounded regexes."""

    if "\\b" in expression:
        return
    for alternative in expression.split("|"):
        simplified = alternative.strip().strip("()?:")
        if re.fullmatch(r"[A-Za-z0-9_]{1,3}", simplified):
            raise InventoryError(
                f"{label} contains noisy bare short term {simplified!r}; "
                "use regex word boundaries or a longer neighboring phrase"
            )


def _hotspots(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> dict[str, Any]:
    """Return bounded lexical leads; callers must establish semantic relevance."""

    _require_discovery_phase(run)
    if args.path_pattern == ".*" and not args.term:
        raise InventoryError("hotspots requires --term or a narrowed --path-pattern")
    files = _query_files(args, run, run_file, repository)
    if len(files) > MAX_FULL_SCOPE_HOTSPOT_FILES:
        raise InventoryError(
            "hotspots scope is too broad for this inventory; first identify a "
            "smaller project-owned seam, then pass a narrower --path-pattern"
        )
    kinds = args.kind or list(HOTSPOT_PATTERNS)
    if args.term and not getattr(args, "_term_validated", False):
        _reject_bare_short_term_alternatives(args.term, "hotspot term")
    term_pattern = _compile_pattern(args.term, "hotspot term") if args.term else None
    ranked: list[tuple[int, str, int, dict[str, Any]]] = []
    for identity, path in files:
        source_lines = list(_iter_text_lines(path))
        term_lines = (
            [line_number for line_number, source_text in source_lines if term_pattern.search(source_text)]
            if term_pattern is not None
            else []
        )
        if term_pattern is not None and not term_lines:
            continue
        for line_number, source_text in source_lines:
            matched = [kind for kind in kinds if HOTSPOT_PATTERNS[kind].search(source_text)]
            if not matched:
                continue
            term_distance = min(abs(line_number - term_line) for term_line in term_lines) if term_lines else None
            # A concept present somewhere in a large file is too weak a signal.
            # Keep the bounded result local to the oriented term so callers see
            # the relevant guard/dispatch/state marker instead of unrelated TODOs.
            if term_distance is not None and term_distance > 40:
                continue
            displayed, truncated = _display_text(source_text)
            record: dict[str, Any] = {
                "path": identity,
                "line": line_number,
                "kinds": matched,
                "text": displayed,
            }
            markers = [marker for marker, pattern in HOTSPOT_SIGNAL_MARKERS.items() if pattern.search(source_text)]
            record["lead_id"] = (
                "L-"
                + hashlib.sha256(
                    f"{identity}\0{line_number}\0{','.join(matched)}\0{source_text}".encode("utf-8")
                ).hexdigest()[:12]
            )
            if markers:
                record["markers"] = markers
            if term_distance is not None:
                record["term_distance"] = term_distance
            if truncated:
                record["text_truncated"] = True
            score = sum({"capability": 50, "boundary": 30, "dispatch": 20, "state": 15}[kind] for kind in matched)
            if len(matched) > 1:
                score += 20
            if re.search(r"(?:<=|>=|==|!=|<|>)\s*\d+|\d+\s*(?:<=|>=|==|!=|<|>)", source_text):
                score += 10
            if term_distance is not None:
                # Relevance breaks ties, but must not bury a nearby TODO or
                # missing-capability marker beneath every line that repeats a
                # common oriented term.
                score += 30 if term_distance == 0 else 25 if term_distance <= 3 else 20 if term_distance <= 12 else 10
            score += 80 if "explicit-omission" in markers else 0
            score += 25 if "finite-boundary" in markers else 0
            score += 20 if "dispatch-diversion" in markers else 0
            score += 15 if "early-exit" in markers else 0
            ranked.append((-score, identity, line_number, record))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    per_file: dict[str, int] = {}
    diverse: list[tuple[int, str, int, dict[str, Any]]] = []
    overflow: list[tuple[int, str, int, dict[str, Any]]] = []
    for item in ranked:
        identity = item[1]
        if per_file.get(identity, 0) < 2:
            diverse.append(item)
            per_file[identity] = per_file.get(identity, 0) + 1
        else:
            overflow.append(item)
    ordered = diverse + overflow
    selected = ordered[args.cursor : args.cursor + args.limit]
    matches = [item[3] for item in selected]
    marker_counts = {
        marker: sum(marker in match.get("markers", []) for match in matches) for marker in HOTSPOT_SIGNAL_MARKERS
    }
    next_cursor = args.cursor + len(matches)
    return {
        "scope": args.scope,
        "matches": matches,
        "next_cursor": None if next_cursor >= len(ranked) else next_cursor,
        "complete": next_cursor >= len(ranked),
        "matched_hotspots": len(ranked),
        "selected_marker_counts": marker_counts,
        "priority_leads": _select_priority_leads([item[3] for item in ranked]),
        "semantic_verdict": "none; lexical leads only",
    }


def _merged_hotspot_sort_key(
    key: tuple[str, int, str],
    *,
    merged: dict[tuple[str, int, str], dict[str, Any]],
    marker_weight: dict[str, int],
    origins: dict[tuple[str, int, str], set[str]],
    order: dict[tuple[str, int, str], int],
) -> tuple[int, int, str, int]:
    """Rank a merged hotspot without capturing loop-local state."""

    record = merged[key]
    strength = sum(marker_weight.get(marker, 0) for marker in record.get("markers", []))
    if origins[key] == {"oriented", "structural"}:
        strength += 15
    if "oriented" in origins[key] and "finite-boundary" in record.get("markers", []):
        strength += 30
    return (-strength, order[key], key[0], key[1])


def _triad(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_file: Path,
    repository: Path,
) -> dict[str, Any]:
    """Run the three required risk-signature passes over two explicit seams."""

    all_groups = {
        "boundary": ["boundary"],
        "dispatch_state": ["dispatch", "state"],
        "capability": ["capability"],
    }
    requested_kinds = set(args.kind or HOTSPOT_PATTERNS)
    groups = {name: [kind for kind in kinds if kind in requested_kinds] for name, kinds in all_groups.items()}
    groups = {name: kinds for name, kinds in groups.items() if kinds}

    def literal_scope(values: list[str] | None, label: str) -> str | None:
        if not values:
            return None
        parts: list[str] = []
        for raw in values:
            if not raw or raw.startswith("/"):
                raise InventoryError(f"{label} paths must be non-empty repository-relative paths")
            pure = PurePosixPath(raw)
            if ".." in pure.parts or str(pure) in {"", "."}:
                raise InventoryError(f"unsafe {label} path: {raw!r}")
            normalized = pure.as_posix().rstrip("/")
            candidate = repository / normalized
            if not candidate.exists():
                raise InventoryError(f"{label} path does not exist: {normalized}")
            escaped = re.escape(normalized)
            parts.append(rf"^{escaped}(?:/.*)?$" if candidate.is_dir() else rf"^{escaped}$")
        return "(?:" + "|".join(parts) + ")"

    integration_pattern = literal_scope(args.integration_path, "integration")
    core_pattern = literal_scope(args.core_path, "core")
    integration_pattern = integration_pattern or args.integration_pattern
    core_pattern = core_pattern or args.core_pattern
    terms = args.term
    for term in terms:
        _reject_bare_short_term_alternatives(term, "triad term")
    term_pattern = "|".join(f"(?:{term})" for term in terms)
    result: dict[str, Any] = {
        "scope": "repository",
        "terms": terms,
        "selected_kinds": sorted(requested_kinds),
        "semantic_verdict": "none; lexical leads only",
    }
    for seam, path_pattern in (
        ("integration", integration_pattern),
        ("core", core_pattern),
    ):
        seam_result: dict[str, Any] = {"path_pattern": path_pattern}
        for group, kinds in groups.items():
            oriented_query = argparse.Namespace(
                run=args.run,
                scope="repository",
                path_pattern=path_pattern,
                cursor=0,
                limit=args.limit,
                term=term_pattern,
                kind=kinds,
                _term_validated=True,
            )
            structural_query = argparse.Namespace(
                run=args.run,
                scope="repository",
                path_pattern=path_pattern,
                cursor=0,
                limit=args.limit,
                term=None,
                kind=kinds,
                _term_validated=True,
            )
            oriented = _hotspots(oriented_query, run, run_file, repository)
            structural = _hotspots(structural_query, run, run_file, repository)
            merged: dict[tuple[str, int, str], dict[str, Any]] = {}
            order: dict[tuple[str, int, str], int] = {}
            origins: dict[tuple[str, int, str], set[str]] = {}
            for channel_name, channel in (
                ("oriented", oriented),
                ("structural", structural),
            ):
                for index, record in enumerate(channel["matches"]):
                    key = (record["path"], record["line"], record["text"])
                    merged.setdefault(key, dict(record))
                    origins.setdefault(key, set()).add(channel_name)
                    order.setdefault(
                        key,
                        index + (0 if channel_name == "oriented" else args.limit),
                    )

            marker_weight = {
                "explicit-omission": 100,
                "finite-boundary": 40,
                "dispatch-diversion": 35,
                "early-exit": 25,
                "state-timing": 20,
            }

            selected_keys = sorted(
                merged,
                key=partial(
                    _merged_hotspot_sort_key,
                    merged=merged,
                    marker_weight=marker_weight,
                    origins=origins,
                    order=order,
                ),
            )[: args.limit]
            selected_matches: list[dict[str, Any]] = []
            for key in selected_keys:
                record = merged[key]
                record["channels"] = sorted(origins[key])
                selected_matches.append(record)
            priority_pool = [
                *selected_matches,
                *oriented["priority_leads"],
                *structural["priority_leads"],
            ]
            # The structural channel scans the same seam/kinds without the
            # oriented term filter, so it is the unique superset count rather
            # than a channel sum that would double-count oriented leads.
            matched_total = structural["matched_hotspots"]
            priority_leads = _select_priority_leads(priority_pool)
            seam_result[group] = {
                "matches": selected_matches,
                "returned_leads": len(selected_matches),
                "matched_total": matched_total,
                "returned_total": len(priority_leads),
                "priority_leads": priority_leads,
                "channels": {
                    "oriented": {
                        "selected": len(oriented["matches"]),
                        "matched": oriented["matched_hotspots"],
                    },
                    "structural": {
                        "selected": len(structural["matches"]),
                        "matched": structural["matched_hotspots"],
                    },
                },
                "selected_marker_counts": {
                    marker: sum(marker in record.get("markers", []) for record in selected_matches)
                    for marker in HOTSPOT_SIGNAL_MARKERS
                },
                "complete": oriented["complete"] and structural["complete"],
                "semantic_verdict": "none; merged lexical leads only",
            }
        result[seam] = seam_result
    return result


def _clock(run: dict[str, Any]) -> dict[str, Any]:
    created_at = run.get("created_at")
    if not isinstance(created_at, str):
        raise InventoryError("run file has no created_at timestamp")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise InventoryError("run created_at timestamp is invalid") from error
    if created.tzinfo is None:
        raise InventoryError("run created_at timestamp has no timezone")
    elapsed = max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
    budget = run.get("budget_seconds")
    if budget is None:
        return {"elapsed_seconds": elapsed, "budget_seconds": None, "phase": "unbounded"}
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise InventoryError("run budget is invalid")
    remaining = max(0, budget - elapsed)
    ratio = elapsed / budget
    phase = "breadth" if ratio < 0.2 else "evidence" if ratio < 0.8 else "finalization"
    return {
        "elapsed_seconds": elapsed,
        "budget_seconds": budget,
        "remaining_seconds": remaining,
        "phase": phase,
        "finalization_required": phase == "finalization",
    }


def _require_discovery_phase(run: dict[str, Any]) -> None:
    state = _clock(run)
    if state.get("finalization_required") is True:
        raise FinalizationRequired(
            "FINALIZATION_REQUIRED: bounded input queries are closed; finish candidate review, report, verify, and lint"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify", "clock"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--run",
            required=True,
            help="run.json emitted by prepare_audit.py",
        )
    for command in ("paths", "search", "read", "hotspots"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run", required=True)
        subparser.add_argument("--scope", choices=("repository", "specification"), required=True)
        subparser.add_argument("--path-pattern", default=".*")
        if command in {"paths", "search", "hotspots"}:
            subparser.add_argument("--cursor", type=int, default=0)
            subparser.add_argument(
                "--limit",
                type=int,
                default=30 if command == "paths" else 20,
                choices=range(1, 31) if command == "paths" else range(1, 21),
            )
        if command == "search":
            subparser.add_argument("--pattern", required=True)
        if command == "read":
            subparser.add_argument("--path", required=True)
            subparser.add_argument("--start-line", type=int, default=1)
            subparser.add_argument("--lines", type=int, default=60, choices=range(1, 81))
        if command == "hotspots":
            subparser.add_argument(
                "--term",
                help="optional oriented-concept regex; only files containing it are ranked",
            )
            subparser.add_argument(
                "--kind",
                action="append",
                choices=tuple(HOTSPOT_PATTERNS),
                help="repeat to select lexical lead classes; defaults to all",
            )
    requirements = subparsers.add_parser("requirements")
    requirements.add_argument("--run", required=True)
    requirements.add_argument(
        "--source",
        required=True,
        help="one pinned SPEC-xxx source",
    )
    requirements.add_argument(
        "--term",
        action="append",
        help="repeatable oriented mechanism regex; omit for a source-wide sweep",
    )
    requirements.add_argument(
        "--limit",
        type=int,
        default=20,
        choices=range(1, 31),
    )
    triad = subparsers.add_parser("triad")
    triad.add_argument("--run", required=True)
    triad.add_argument(
        "--term",
        action="append",
        required=True,
        help="repeatable oriented-concept regex grounded in one discovery package",
    )
    integration = triad.add_mutually_exclusive_group(required=True)
    integration.add_argument(
        "--integration-pattern",
        help="narrow path regex for project-owned boundary and integration code",
    )
    integration.add_argument(
        "--integration-path",
        action="append",
        help="repeatable repository-relative integration file or directory (preferred)",
    )
    core = triad.add_mutually_exclusive_group(required=True)
    core.add_argument(
        "--core-pattern",
        help="narrow path regex for the connected implementation core",
    )
    core.add_argument(
        "--core-path",
        action="append",
        help="repeatable repository-relative core file or directory (preferred)",
    )
    triad.add_argument(
        "--kind",
        action="append",
        choices=tuple(HOTSPOT_PATTERNS),
        help="optional compatibility filter; omit to run the complete triad",
    )
    triad.add_argument("--limit", type=int, default=10, choices=range(1, 21))
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run, run_file, repository = _load_run(Path(args.run))
        if args.command == "build":
            result = _build(run, run_file, repository)
            exit_code = 0
        elif args.command == "verify":
            exit_code, result = _verify(run, run_file, repository)
        elif args.command == "clock":
            result = _clock(run)
            exit_code = 0
        elif args.command == "paths":
            result = _paths(args, run, run_file, repository)
            exit_code = 0
        elif args.command == "search":
            result = _search(args, run, run_file, repository)
            exit_code = 0
        elif args.command == "read":
            result = _read(args, run, run_file, repository)
            exit_code = 0
        elif args.command == "hotspots":
            result = _hotspots(args, run, run_file, repository)
            exit_code = 0
        elif args.command == "requirements":
            result = _requirements(args, run)
            exit_code = 0
        else:
            result = _triad(args, run, run_file, repository)
            exit_code = 0
    except FinalizationRequired as error:
        print(str(error), file=sys.stderr)
        return 1
    except (InventoryError, OSError, UnicodeError, ValueError) as error:
        print(f"inventory: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
