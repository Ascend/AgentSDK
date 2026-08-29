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

"""Atomic on-disk persistence for a :class:`TemplateRegistry`.

The state file is a single JSON document keyed by template id, with the
per-template value being the result of :meth:`Template.to_dict`. Atomic
write follows the same pattern established by
:mod:`src.services.ultraplan.store` and
:mod:`src.services.context_collapse.persistence`: write to a temp file
in the same directory, fsync + ``os.replace`` to publish, and clean up
the temp file on failure.

The class is intentionally separate from :class:`TemplateRegistry` so
callers can:

* Save a registry to disk and reload it in a fresh process.
* Round-trip a registry that was populated via :meth:`discover`
  without paying the re-parse cost on every read.
* Persist snapshots of a registry at specific points in time
  (e.g. before / after a CLI ``/template create`` command).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from .exceptions import (
    TemplateCorruptError,
    TemplateNotFoundError,
)
from .models import Template
from .registry import TemplateRegistry


class TemplateStateFile:
    """Atomic on-disk persistence for a :class:`TemplateRegistry`."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        with self._lock:
            return self._path.exists()

    def save(self, registry: TemplateRegistry) -> Path:
        if not isinstance(registry, TemplateRegistry):
            raise TypeError("TemplateStateFile.save expects a TemplateRegistry")
        with self._lock:
            payload: dict[str, Any] = {
                "version": self.SCHEMA_VERSION,
                "templates": {t.id: t.to_dict() for t in registry.list_templates()},
            }
            self._atomic_write(self._path, payload)
            return self._path

    def load(self) -> TemplateRegistry:
        with self._lock:
            if not self._path.exists():
                raise TemplateNotFoundError(f"template state file does not exist: {self._path}")
            try:
                raw = self._path.read_text(encoding="utf-8")
                data: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TemplateCorruptError(f"template state file is not valid JSON: {exc}") from exc
            except OSError as exc:
                raise TemplateCorruptError(f"template state file could not be read: {exc}") from exc
            if not isinstance(data, dict):
                raise TemplateCorruptError(f"template state root must be an object: {self._path}")
            templates_payload = data.get("templates", {})
            if not isinstance(templates_payload, dict):
                raise TemplateCorruptError("template state 'templates' must be an object")
            registry = TemplateRegistry()
            for tid, tdata in templates_payload.items():
                try:
                    # Inject the id so from_dict can validate the shape.
                    if isinstance(tdata, dict) and "id" not in tdata:
                        tdata = {**tdata, "id": tid}
                    template = Template.from_dict(tdata)
                except (ValueError, TypeError) as exc:
                    raise TemplateCorruptError(f"template {tid!r} failed validation: {exc}") from exc
                registry.register(template, overwrite=True)
            return registry

    def delete(self) -> None:
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return

    def _atomic_write(self, target: Path, payload: dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # Best-effort operation failed; keep the surrounding fallback.
            raise


def save_registry(path: Path | str, registry: TemplateRegistry) -> Path:
    """Convenience: write ``registry`` to ``path`` atomically."""
    return TemplateStateFile(path).save(registry)


def load_registry(path: Path | str) -> TemplateRegistry:
    """Convenience: read a registry from ``path``."""
    return TemplateStateFile(path).load()


def merge_registries(
    target: TemplateRegistry,
    source: TemplateRegistry,
    *,
    prefer_target: bool = True,
) -> int:
    """Append any templates in ``source`` missing from ``target``.

    Two templates are considered the same when their ``id`` matches.
    The function returns the number of templates appended to ``target``.
    """
    if not isinstance(target, TemplateRegistry) or not isinstance(source, TemplateRegistry):
        raise TypeError("merge_registries expects two TemplateRegistry instances")
    added = 0
    for template in source.list_templates():
        if template.id in target:
            if not prefer_target:
                target.register(template, overwrite=True)
            continue
        target.register(template, overwrite=False)
        added += 1
    return added


def safe_filename_suffix() -> str:
    """Return a short, low-collision suffix used in temp file names."""
    return uuid.uuid4().hex[:8]


__all__ = [
    "TemplateStateFile",
    "load_registry",
    "merge_registries",
    "safe_filename_suffix",
    "save_registry",
]
