# pylint: disable=relative-beyond-top-level
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

"""SOP Resource Catalog.

Persistent runtime resource handles for SOP-converted SDK tools.

**Write path:** create tools upsert :class:`ResourceRecord` into
``resource-catalog.json`` via :func:`build_resource_record_from_create`.

**Read path:** resolve only from the resource catalog (bundle/user/
session locations). Legacy ``agent-catalog.json`` is no longer read or written.
"""

# pylint: disable=too-many-lines,too-many-nested-blocks
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

try:
    import fcntl
except Exception:  # pragma: no cover - not available on Windows
    fcntl = None

HOME_ROOT_ENV = "CLAWCODEX_HOME"
HOME_ONLY_ENV = "CLAWCODEX_CATALOG_HOME_ONLY"

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_HOME = Path.home() / ".clawcodex"
RESOURCE_CATALOG_FILENAME = "resource-catalog.json"
SESSION_ID_ENV = "CLAWCODEX_SESSION_ID"
DUAL_WRITE_ENV = "CLAWCODEX_CATALOG_DUAL_WRITE"
PAYLOAD_REF_ENV = "CLAWCODEX_CATALOG_PAYLOAD_REF"
PAYLOAD_SPILL_THRESHOLD = 65536

RESOURCE_CATALOG_MISSING = "resource_catalog_missing"
RESOURCE_CATALOG_AMBIGUOUS = "resource_catalog_ambiguous"
RESOURCE_CATALOG_WRITE_FAILED = "resource_catalog_write_failed"
RESOURCE_PAYLOAD_INVALID = "resource_payload_invalid"
RESOURCE_PAYLOAD_REF_MISSING = "resource_payload_ref_missing"
RESOURCE_MATERIALIZE_FAILED = "resource_materialize_failed"
RESOURCE_VERSION_UNSUPPORTED = "resource_version_unsupported"
RESOURCE_SECRET_MISSING = "resource_secret_missing"  # nosec

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|token|secret|password|passwd|pwd)",
    re.IGNORECASE,
)
_REDACTION_PLACEHOLDER = "<redacted:env:{env_var}>"
_AGENT_REFERENCE_FIELDS = frozenset({"name", "agent_name", "display_name", "alias", "aliases"})
_PAYLOAD_STUB_FIELDS = frozenset({"kind", "ref", "handle_field", "name_index", "media_type"})


class ResourceCatalogError(RuntimeError):
    """Catalog lookup failure with a machine-readable error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_resource_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def normalize_resource_type(value: str) -> str:
    """Return the canonical key used by catalogs and handler registries."""
    return _normalise_resource_type(value)


def _resource_key(resource_type: str, resource_id: str) -> str:
    return f"{_normalise_resource_type(resource_type)}:{str(resource_id).strip()}"


def _normalise_agent_reference(value: Any) -> str:
    return str(value or "").strip().casefold()


def _collect_agent_reference_names(value: Any) -> set[str]:
    """Extract explicit human-facing agent names from catalog payloads.

    This deliberately follows only name/alias fields. It avoids treating a
    model name, SDK class, or arbitrary string in a DSL as an agent reference.
    """
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _AGENT_REFERENCE_FIELDS:
                if isinstance(child, str):
                    normalised = _normalise_agent_reference(child)
                    if normalised:
                        names.add(normalised)
                elif isinstance(child, (list, tuple, set)):
                    for item in child:
                        if isinstance(item, str):
                            normalised = _normalise_agent_reference(item)
                            if normalised:
                                names.add(normalised)
            elif isinstance(child, (dict, list, tuple)):
                names.update(_collect_agent_reference_names(child))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.update(_collect_agent_reference_names(item))
    return names


def _agent_reference_names(record: ResourceRecord) -> set[str]:
    names = _collect_agent_reference_names({"payload": record.payload, "metadata": record.metadata})
    payload = record.payload or {}
    name_index = payload.get("name_index")
    if isinstance(name_index, (list, tuple, set)):
        for item in name_index:
            if isinstance(item, str):
                normalised = _normalise_agent_reference(item)
                if normalised:
                    names.add(normalised)
    return names


def payload_ref_filename(resource_type: str, resource_id: str) -> str:
    key = _resource_key(resource_type, resource_id)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"payloads/{digest}.json"


def safe_payload_path(catalog_dir: Path, ref: str) -> Path:
    catalog_dir = Path(catalog_dir).expanduser().resolve()
    ref_str = str(ref or "").strip()
    if not ref_str:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            "payload ref is empty",
        )
    ref_path = Path(ref_str)
    if ref_path.is_absolute() or ref_path.drive:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload ref must be relative: {ref_str!r}",
        )
    if ".." in ref_path.parts:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload ref must not contain ..: {ref_str!r}",
        )
    resolved = (catalog_dir / ref_path).resolve()
    try:
        resolved.relative_to(catalog_dir)
    except ValueError as exc:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload ref escapes catalog directory: {ref_str!r}",
        ) from exc
    return resolved


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write *payload* as indented JSON to *path*.

    The payload is dumped to a temporary file in the same directory and
    *path* is replaced only after the dump succeeds; on any failure the
    temporary file is removed and the original *path* is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # nosec
            pass
        raise


def _write_payload_file(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(path, payload)


def spill_payload_if_needed(
    record: ResourceRecord,
    catalog_dir: Path,
    *,
    force_ref: bool = False,
    force_inline: bool = False,
) -> ResourceRecord:
    payload = dict(record.payload or {})
    if payload.get("kind") == "payload_ref":
        return record
    if force_inline:
        return record

    spill_body = {key: value for key, value in payload.items() if key not in _PAYLOAD_STUB_FIELDS}
    spill_bytes = len(json.dumps(spill_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    should_spill = force_ref or _env_flag(PAYLOAD_REF_ENV) or spill_bytes > PAYLOAD_SPILL_THRESHOLD
    if not should_spill:
        return record

    name_index = sorted(_collect_agent_reference_names(payload))
    env_refs: list[str] = list((record.secrets or {}).get("env_refs") or [])
    redacted_body = _redact_tree(
        spill_body,
        bundle_id=record.bundle_id,
        env_refs=env_refs,
    )
    ref = payload_ref_filename(record.resource_type, record.resource_id)
    _write_payload_file(safe_payload_path(catalog_dir, ref), redacted_body)

    stub_payload: dict[str, Any] = {
        key: payload[key] for key in _PAYLOAD_STUB_FIELDS if key in payload and key not in {"kind", "ref"}
    }
    stub_payload["kind"] = "payload_ref"
    stub_payload["ref"] = ref
    stub_payload["name_index"] = name_index

    secrets = dict(record.secrets or {})
    if env_refs:
        secrets["env_refs"] = sorted(set(secrets.get("env_refs") or []) | set(env_refs))

    return ResourceRecord(
        resource_type=record.resource_type,
        resource_id=record.resource_id,
        bundle_id=record.bundle_id,
        source_tool=record.source_tool,
        materializer=dict(record.materializer or {}),
        invoker=dict(record.invoker or {}),
        payload=stub_payload,
        sdk=dict(record.sdk or {}),
        secrets=secrets,
        status=record.status,
        schema_version=record.schema_version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=dict(record.metadata or {}),
    )


def resolve_payload(
    record: ResourceRecord,
    catalog_dir: Path,
    *,
    restore: bool = True,
) -> ResourceRecord:
    """Inline a ``payload_ref`` spill file into the record.

    Runtime callers keep ``restore=True`` (default) so secrets match
    :meth:`ResourceCatalog.get`. CLI storage views pass ``restore=False``.
    """
    payload = dict(record.payload or {})
    if payload.get("kind") != "payload_ref":
        return record

    ref = payload.get("ref")
    if not ref:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            "payload_ref record is missing ref",
        )

    try:
        path = safe_payload_path(catalog_dir, str(ref))
    except ResourceCatalogError:
        raise
    except Exception as exc:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"invalid payload ref {ref!r}: {exc}",
        ) from exc

    if not path.is_file():
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload file missing: {ref}",
        )

    try:
        external = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload file unreadable: {ref} ({exc})",
        ) from exc

    if not isinstance(external, dict):
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            f"payload file must contain an object: {ref}",
        )

    inline_payload = {key: value for key, value in payload.items() if key not in {"ref"}}
    inline_payload["kind"] = "inline"
    inline_payload.update(external)

    materializer = dict(record.materializer or {})
    invoker = dict(record.invoker or {})
    sdk = dict(record.sdk or {})
    if restore:
        materializer = _restore_tree(materializer)
        invoker = _restore_tree(invoker)
        inline_payload = _restore_tree(inline_payload)
        sdk = _restore_tree(sdk)

    return ResourceRecord(
        resource_type=record.resource_type,
        resource_id=record.resource_id,
        bundle_id=record.bundle_id,
        source_tool=record.source_tool,
        materializer=materializer,
        invoker=invoker,
        payload=inline_payload,
        sdk=sdk,
        secrets=dict(record.secrets or {}),
        status=record.status,
        schema_version=record.schema_version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=dict(record.metadata or {}),
    )


def _is_agent_record(record: ResourceRecord) -> bool:
    return "agent" in record.resource_type or isinstance((record.payload or {}).get("agent_catalog_entry"), dict)


def _clawcodex_home() -> Path:
    raw = os.environ.get(HOME_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_HOME.resolve()


def _is_home_only_forced() -> bool:
    raw = os.environ.get(HOME_ONLY_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _bundle_id_from_path(bundle: Path | str | None, bundle_id: str | None) -> str:
    if bundle_id and bundle_id.strip():
        return bundle_id.strip()
    if bundle is not None:
        return Path(bundle).expanduser().resolve().name
    return "default"


def _redact_value(key: str, value: Any, *, bundle_id: str | None) -> tuple[Any, str | None]:
    if not _SENSITIVE_KEY_RE.search(key):
        return value, None
    if isinstance(value, str) and value.startswith("env:"):
        return value, None
    bundle_prefix = re.sub(r"[^A-Z0-9]+", "_", (bundle_id or "BUNDLE").upper())
    field_suffix = re.sub(r"[^A-Z0-9]+", "_", key.upper())
    env_var = f"CLAWCODEX_{bundle_prefix}_{field_suffix}"
    return _REDACTION_PLACEHOLDER.format(env_var=env_var), env_var


def _redact_tree(value: Any, *, bundle_id: str | None, env_refs: list[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                if _SENSITIVE_KEY_RE.search(str(key)):
                    # A sensitive key holding a container would otherwise be
                    # persisted in plaintext; redact the whole container as an
                    # opaque value (restored via JSON when the env var is set).
                    redacted, env_var = _redact_value(
                        str(key), json.dumps(child, ensure_ascii=False), bundle_id=bundle_id
                    )
                    out[key] = redacted
                    if env_var:
                        env_refs.append(env_var)
                else:
                    out[key] = _redact_tree(child, bundle_id=bundle_id, env_refs=env_refs)
                continue
            redacted, env_var = _redact_value(str(key), child, bundle_id=bundle_id)
            out[key] = redacted
            if env_var:
                env_refs.append(env_var)
        return out
    if isinstance(value, list):
        return [_redact_tree(item, bundle_id=bundle_id, env_refs=env_refs) for item in value]
    return value


def _restore_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _restore_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore_tree(item) for item in value]
    if isinstance(value, str):
        match = re.match(r"^<redacted:env:([A-Z0-9_]+)>$", value)
        if match:
            env_var = match.group(1)
            restored = os.environ.get(env_var, value)
            if isinstance(restored, str) and restored.startswith(("{", "[")):
                try:
                    return json.loads(restored)
                except (ValueError, TypeError):
                    return restored
            return restored
    return value


@dataclass(frozen=True)
class ResourceCatalogLocation:
    """Resolved catalog path and provenance."""

    path: Path
    reason: str
    writable: bool | None = None

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CatalogExecutionContext:
    """Canonical catalog identity shared by create and workflow read paths."""

    bundle_path: Path | None = None
    bundle_id: str = ""
    home_only: bool = False
    session_id: str = ""
    dual_write: bool = False

    def __post_init__(self) -> None:
        bundle_path = self.bundle_path
        if bundle_path is not None:
            bundle_path = Path(bundle_path).expanduser().resolve()
            object.__setattr__(self, "bundle_path", bundle_path)
        bundle_id = str(self.bundle_id or "").strip()
        if not bundle_id and bundle_path is not None:
            bundle_id = bundle_path.name
        object.__setattr__(self, "bundle_id", bundle_id or "default")


def context_from_env(
    *,
    bundle_path: Path | str | None = None,
    bundle_id: str = "",
    home_only: bool | None = None,
    session_id: str | None = None,
    dual_write: bool | None = None,
) -> CatalogExecutionContext:
    if home_only is None:
        home_only = _is_home_only_forced()
    if session_id is None:
        session_id = os.environ.get(SESSION_ID_ENV, "").strip()
    if dual_write is None:
        dual_write = _env_flag(DUAL_WRITE_ENV)
    return CatalogExecutionContext(
        bundle_path=Path(bundle_path) if bundle_path else None,
        bundle_id=bundle_id,
        home_only=bool(home_only),
        session_id=str(session_id or ""),
        dual_write=bool(dual_write),
    )


def format_resource_catalog_locations_block(
    bundle_path: Path | str | None = None,
    *,
    bundle_id: str = "",
) -> str:
    """Prompt block listing catalog locations for SOP agents.

    Session path is a **template** (no concrete session id at startup).
    Instructs agents to use ``resource-catalog`` + Read — not workspace Grep.
    """
    ctx = context_from_env(bundle_path=bundle_path, bundle_id=bundle_id)
    home = _clawcodex_home()
    bid = ctx.bundle_id or "default"

    bundle_loc = resolve_resource_catalog_path(ctx.bundle_path, bundle_id=bid, scope="bundle")
    user_loc = resolve_resource_catalog_path(ctx.bundle_path, bundle_id=bid, scope="user", home_only=False)

    def _status(path: Path) -> str:
        try:
            return "exists" if path.is_file() else "missing"
        except OSError:
            return "missing"

    session_template = f"{home}/sessions/<session_id>/sop-resources.json"
    lines = [
        "## Resource catalogs (read order: session → bundle → user)",
        f"- session: {session_template}",
        f"- bundle:  {bundle_loc.path}  ({_status(bundle_loc.path)})",
        f"- user:    {user_loc.path}  ({_status(user_loc.path)})",
        f"CLAWCODEX_HOME: {home}",
        "",
        "Inspect persisted agents/resources with tool `resource-catalog` "
        "(action=list|get|latest) — it uses the current session automatically.",
        "If you need the raw JSON file, Read only catalog files: "
        "sop-resources.json, catalog.json, or resource-catalog.json under the roots above.",
        "Do not search the workspace for agent ids (no Grep/Glob/Bash find for verify-bot).",
        "To invoke an existing agent: resume-resource / invoke-existing-agent.",
    ]
    return "\n".join(lines)


@dataclass
class ResourceRecord:
    """Generic runtime resource row."""

    resource_type: str
    resource_id: str
    source_tool: str
    materializer: dict[str, Any]
    invoker: dict[str, Any]
    payload: dict[str, Any]
    bundle_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    sdk: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resource_type = _normalise_resource_type(self.resource_type)
        self.resource_id = str(self.resource_id)

    def key(self) -> str:
        return _resource_key(self.resource_type, self.resource_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ResourceCatalog:
    """On-disk ResourceRecord index.

    Storage format::

        {
          "version": 1,
          "records": {"agent:<id>": {...ResourceRecord...}}
        }
    """

    _save_lock: ClassVar[threading.Lock] = threading.Lock()

    version: int = SCHEMA_VERSION
    records: dict[str, ResourceRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ResourceCatalog:
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("resource-catalog: %s is invalid JSON (%s)", path, exc)
            return cls()
        if not isinstance(raw, dict):
            logger.warning("resource-catalog: %s top-level must be an object", path)
            return cls()
        version = raw.get("version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ResourceCatalogError(
                RESOURCE_VERSION_UNSUPPORTED,
                f"resource catalog {path} version={version} is unsupported; expected {SCHEMA_VERSION}",
            )
        records_raw = raw.get("records", {})
        if not isinstance(records_raw, dict):
            logger.warning("resource-catalog: %s records must be an object", path)
            return cls()
        records: dict[str, ResourceRecord] = {}
        for key, payload in records_raw.items():
            if not isinstance(payload, dict):
                logger.warning("resource-catalog: skipping non-dict record %r", key)
                continue
            try:
                record = ResourceRecord.from_dict(payload)
            except Exception as exc:
                logger.warning("resource-catalog: skipping bad record %r (%s)", key, exc)
                continue
            records[record.key()] = record
        return cls(version=SCHEMA_VERSION, records=records)

    def _save_unlocked(self, path: Path, *, merge: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = dict(self.records)
        if merge:
            disk = self.load(path)
            for key, record in disk.records.items():
                if key not in records:
                    records[key] = record
        payload = {
            "version": self.version,
            "records": {key: record.to_dict() for key, record in sorted(records.items(), key=lambda item: item[0])},
        }
        _atomic_write_json(path, payload)

    def save(self, path: Path, *, merge: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / f".{path.name}.lock"
        with open(lock_path, "a+", encoding="utf-8") as lock_fd:
            if fcntl is not None:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                try:
                    self._save_unlocked(path, merge=merge)
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            else:
                with self._save_lock:
                    self._save_unlocked(path, merge=merge)

    def upsert(self, record: ResourceRecord) -> ResourceRecord:
        if not record.resource_type or not record.resource_id:
            raise ValueError("resource_type and resource_id are required")
        if record.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported resource schema_version={record.schema_version}")

        env_refs: list[str] = []
        existing = self.records.get(record.key())
        secrets = dict(record.secrets or {})
        existing_env_refs = []
        if existing and isinstance(existing.secrets, dict):
            existing_env_refs = list(existing.secrets.get("env_refs") or [])
        secrets["env_refs"] = sorted(set(existing_env_refs) | set(secrets.get("env_refs") or []) | set(env_refs))

        metadata = dict(existing.metadata) if existing else {}
        metadata.update(record.metadata or {})
        redacted = ResourceRecord(
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            bundle_id=record.bundle_id,
            source_tool=record.source_tool,
            materializer=_redact_tree(
                record.materializer or {},
                bundle_id=record.bundle_id,
                env_refs=env_refs,
            ),
            invoker=_redact_tree(
                record.invoker or {},
                bundle_id=record.bundle_id,
                env_refs=env_refs,
            ),
            payload=_redact_tree(
                record.payload or {},
                bundle_id=record.bundle_id,
                env_refs=env_refs,
            ),
            sdk=_redact_tree(record.sdk or {}, bundle_id=record.bundle_id, env_refs=env_refs),
            secrets=secrets,
            status=record.status or "active",
            schema_version=record.schema_version,
            created_at=existing.created_at if existing else record.created_at,
            updated_at=_now(),
            metadata=metadata,
        )
        if env_refs:
            redacted.secrets["env_refs"] = sorted(set(redacted.secrets.get("env_refs") or []) | set(env_refs))
        self.records[redacted.key()] = redacted
        return redacted

    def get(self, resource_type: str, resource_id: str) -> ResourceRecord | None:
        record = self.records.get(_resource_key(resource_type, resource_id))
        if record is None:
            return None
        return ResourceRecord(
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            bundle_id=record.bundle_id,
            source_tool=record.source_tool,
            materializer=_restore_tree(record.materializer),
            invoker=_restore_tree(record.invoker),
            payload=_restore_tree(record.payload),
            sdk=_restore_tree(record.sdk),
            secrets=dict(record.secrets),
            status=record.status,
            schema_version=record.schema_version,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=dict(record.metadata),
        )

    def get_stored(self, resource_type: str, resource_id: str) -> ResourceRecord | None:
        record = self.records.get(_resource_key(resource_type, resource_id))
        if record is None:
            return None
        return ResourceRecord(
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            bundle_id=record.bundle_id,
            source_tool=record.source_tool,
            materializer=dict(record.materializer or {}),
            invoker=dict(record.invoker or {}),
            payload=dict(record.payload or {}),
            sdk=dict(record.sdk or {}),
            secrets=dict(record.secrets),
            status=record.status,
            schema_version=record.schema_version,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=dict(record.metadata),
        )

    def find_by_resource_id(
        self,
        resource_id: str,
        *,
        resource_type: str | None = None,
    ) -> list[ResourceRecord]:
        target_type = _normalise_resource_type(resource_type or "")
        matches: list[ResourceRecord] = []
        for record in self.records.values():
            if str(record.resource_id) != str(resource_id):
                continue
            if target_type and record.resource_type != target_type:
                continue
            restored = self.get(record.resource_type, record.resource_id)
            if restored is not None:
                matches.append(restored)
        return sorted(matches, key=lambda item: item.created_at, reverse=True)

    def find_by_agent_reference(
        self,
        agent_ref: str,
        *,
        resource_type: str | None = None,
    ) -> list[ResourceRecord]:
        """Resolve an agent's stable ID or persisted human-facing name.

        Stable IDs retain the existing exact-match semantics.  A name or alias
        is resolved only from explicit catalog fields such as ``dsl.name`` and
        ``metadata.aliases``.  Callers must reject multiple name matches rather
        than guessing which saved agent a user meant.
        """
        reference = str(agent_ref or "").strip()
        if not reference:
            return []

        exact = self.find_by_resource_id(reference, resource_type=resource_type)
        if exact:
            return exact

        target_type = _normalise_resource_type(resource_type or "")
        target_name = _normalise_agent_reference(reference)
        matches: list[ResourceRecord] = []
        for record in self.records.values():
            if target_type and record.resource_type != target_type:
                continue
            if not _is_agent_record(record):
                continue
            if target_name not in _agent_reference_names(record):
                continue
            restored = self.get(record.resource_type, record.resource_id)
            if restored is not None:
                matches.append(restored)
        return sorted(matches, key=lambda item: item.created_at, reverse=True)

    def put_prepared(self, record: ResourceRecord) -> ResourceRecord:
        """Insert an already-redacted record without re-stamping updated_at."""
        self.records[record.key()] = record
        return record

    def latest(self, resource_type: str) -> ResourceRecord | None:
        nt = _normalise_resource_type(resource_type)
        candidates = [r for r in self.records.values() if r.resource_type == nt and (r.status or "active") == "active"]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.updated_at or "", r.created_at or ""))

    def find_by_source_tool(self, source_tool: str) -> list[ResourceRecord]:
        target = str(source_tool or "")
        return sorted(
            [r for r in self.records.values() if r.source_tool == target],
            key=lambda r: (r.updated_at or "", r.created_at or ""),
            reverse=True,
        )

    def mark_failed(
        self,
        resource_type: str,
        resource_id: str,
        *,
        reason: str = "",
    ) -> None:
        key = _resource_key(resource_type, resource_id)
        rec = self.records.get(key)
        if rec is None:
            raise ResourceCatalogError(RESOURCE_CATALOG_MISSING, f"missing {key}")
        rec.status = "failed"
        rec.metadata = dict(rec.metadata or {})
        rec.metadata["failure_reason"] = reason
        rec.updated_at = _now()

    def delete(self, resource_type: str, resource_id: str) -> bool:
        key = _resource_key(resource_type, resource_id)
        return self.records.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        return sorted(self.records.keys())


def mutate_catalog(
    path: Path,
    mutator: Callable[[ResourceCatalog], None],
    *,
    merge: bool,
) -> None:
    """Load, mutate, and persist a catalog under an exclusive file lock.

    The mutator must not call ``cat.save()`` while the lock is held.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_fd:
        if fcntl is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                cat = ResourceCatalog.load(path)
                mutator(cat)
                cat._save_unlocked(path, merge=merge)
            finally:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        else:
            with ResourceCatalog._save_lock:
                cat = ResourceCatalog.load(path)
                mutator(cat)
                cat._save_unlocked(path, merge=merge)


def delete_resource_at(path: Path, resource_type: str, resource_id: str) -> bool:
    catalog_dir = Path(path).parent
    deleted = False
    payload_ref: str | None = None

    def mutator(cat: ResourceCatalog) -> None:
        nonlocal deleted, payload_ref
        key = _resource_key(resource_type, resource_id)
        record = cat.records.get(key)
        if record is None:
            return
        payload = record.payload or {}
        if payload.get("kind") == "payload_ref":
            ref = payload.get("ref")
            if isinstance(ref, str) and ref.strip():
                payload_ref = ref.strip()
        deleted = cat.delete(resource_type, resource_id)

    mutate_catalog(path, mutator, merge=False)

    if deleted and payload_ref:
        try:
            safe_payload_path(catalog_dir, payload_ref).unlink(missing_ok=True)
        except ResourceCatalogError:
            logger.warning(
                "resource-catalog: skipped unlink for unsafe payload ref %r at %s",
                payload_ref,
                path,
            )

    return deleted


def resolve_resource_catalog_path(
    bundle: Path | str | None = None,
    *,
    bundle_id: str | None = None,
    session_id: str | None = None,
    scope: str | None = None,
    home_only: bool | None = None,
) -> ResourceCatalogLocation:
    """Resolve one catalog path.

    ``scope`` may be ``"bundle"``, ``"user"``, ``"session"``, or ``None``.
    With ``None`` we prefer bundle-local when a bundle is available, otherwise
    user-local.  ``CLAWCODEX_CATALOG_HOME_ONLY=1`` forces user-local.
    """
    if home_only is None:
        home_only = _is_home_only_forced()
    effective_bundle_id = _bundle_id_from_path(bundle, bundle_id)
    home = _clawcodex_home()

    if scope == "session":
        sid = (session_id or "default").strip() or "default"
        path = home / "sessions" / sid / "sop-resources.json"
        return ResourceCatalogLocation(path=path, reason="session-local", writable=_probe_writable(path))

    if scope == "user" or home_only or bundle is None:
        path = home / "sop-resources" / effective_bundle_id / "catalog.json"
        reason = "user-local" if scope == "user" else ("home-forced" if home_only else "no-bundle")
        return ResourceCatalogLocation(path=path, reason=reason, writable=_probe_writable(path))

    bundle_path = Path(bundle).expanduser().resolve()
    path = bundle_path / ".clawcodex" / RESOURCE_CATALOG_FILENAME
    return ResourceCatalogLocation(path=path, reason="bundle-local", writable=_probe_writable(path))


def plan_write_targets(ctx: CatalogExecutionContext) -> list[ResourceCatalogLocation]:
    out: list[ResourceCatalogLocation] = []
    if ctx.session_id:
        out.append(
            resolve_resource_catalog_path(
                ctx.bundle_path,
                bundle_id=ctx.bundle_id,
                session_id=ctx.session_id,
                scope="session",
            )
        )
    if ctx.dual_write and not ctx.home_only and ctx.bundle_path is not None:
        out.append(resolve_resource_catalog_path(ctx.bundle_path, bundle_id=ctx.bundle_id, scope="bundle"))
        out.append(
            resolve_resource_catalog_path(ctx.bundle_path, bundle_id=ctx.bundle_id, scope="user", home_only=False)
        )
    elif ctx.home_only or ctx.bundle_path is None:
        out.append(
            resolve_resource_catalog_path(
                ctx.bundle_path,
                bundle_id=ctx.bundle_id,
                scope="user",
                home_only=True,
            )
        )
    else:
        out.append(resolve_resource_catalog_path(ctx.bundle_path, bundle_id=ctx.bundle_id, scope="bundle"))
    return out


_REASON_TO_LAYER = {
    "session-local": "session",
    "bundle-local": "bundle",
    "user-local": "user",
    "home-forced": "user",
    "no-bundle": "user",
}


def _layer_name(location: ResourceCatalogLocation) -> str:
    return _REASON_TO_LAYER.get(location.reason, location.reason)


def _prepare_redacted_record(record: ResourceRecord, *, stamp: str) -> ResourceRecord:
    """Redact once and apply a shared timestamp for multi-layer writes."""
    prep = ResourceCatalog()
    prepared = prep.upsert(record)
    prepared.updated_at = stamp
    return prepared


@dataclass
class WriteResult:
    """Envelope returned by multi-layer catalog writes."""

    written_layers: list[str]
    failed_layer: str | None
    catalog_paths: dict[str, str]
    resource_catalog_path: str
    retryable: bool
    error: str = ""


def write_record(record: ResourceRecord, ctx: CatalogExecutionContext) -> WriteResult:
    """Write a resource record to all targets from ``plan_write_targets``."""
    stamp = _now()
    prepared = _prepare_redacted_record(record, stamp=stamp)
    payload_kind = str((record.metadata or {}).get("payload_kind") or "").strip()
    force_ref = payload_kind == "payload_ref"
    force_inline = payload_kind == "inline"

    written_layers: list[str] = []
    catalog_paths: dict[str, str] = {}
    failed_layer: str | None = None
    error = ""

    for target in plan_write_targets(ctx):
        layer = _layer_name(target)
        try:
            target.ensure_parent()
            layer_record = spill_payload_if_needed(
                prepared,
                target.path.parent,
                force_ref=force_ref,
                force_inline=force_inline,
            )

            def mutator(cat: ResourceCatalog, rec: ResourceRecord = layer_record) -> None:
                existing = cat.records.get(rec.key())
                if existing is not None:
                    secrets = dict(rec.secrets or {})
                    existing_refs = list((existing.secrets or {}).get("env_refs") or [])
                    new_refs = list(secrets.get("env_refs") or [])
                    if existing_refs or new_refs:
                        secrets["env_refs"] = sorted(set(existing_refs) | set(new_refs))
                    rec = replace(
                        rec,
                        created_at=existing.created_at,
                        secrets=secrets,
                    )
                cat.put_prepared(rec)

            mutate_catalog(target.path, mutator, merge=True)
        except Exception as exc:
            if failed_layer is None:
                failed_layer = layer
            if not error:
                error = RESOURCE_CATALOG_WRITE_FAILED
            logger.warning(
                "resource-catalog: write failed for layer %s at %s: %s",
                layer,
                target.path,
                exc,
            )
            continue

        written_layers.append(layer)
        catalog_paths[layer] = str(target.path)

    resource_catalog_path = ""
    for base in ("bundle", "user", "session"):
        if base in catalog_paths:
            resource_catalog_path = catalog_paths[base]
            break

    return WriteResult(
        written_layers=written_layers,
        failed_layer=failed_layer,
        catalog_paths=catalog_paths,
        resource_catalog_path=resource_catalog_path,
        retryable=failed_layer is not None,
        error=error,
    )


@dataclass(frozen=True)
class ResolvedResource:
    """Cross-layer resolve result with provenance."""

    record: ResourceRecord
    location: ResourceCatalogLocation


_LAYER_RANK = {
    "session-local": 0,
    "bundle-local": 1,
    "user-local": 2,
    "home-forced": 2,
    "no-bundle": 2,
}


def _find_matches_at_location(
    location: ResourceCatalogLocation,
    reference: str,
    *,
    handler: Any,
    resource_type: str,
    normalized_type: str,
    agent_family: bool,
) -> list[ResourceRecord]:
    from .resource_handlers import get_resource_handler

    if not location.path.exists():
        return []
    catalog = ResourceCatalog.load(location.path)
    exact_type = None if agent_family and normalized_type in {"agent", "agentconfig"} else resource_type
    matches = catalog.find_by_resource_id(reference, resource_type=exact_type)
    if agent_family and not matches:
        matches = catalog.find_by_agent_reference(
            reference,
            resource_type=exact_type,
        )
    return [record for record in matches if get_resource_handler(record.resource_type) is handler]


def _pick_cross_layer_winner(
    left: tuple[ResourceRecord, ResourceCatalogLocation],
    right: tuple[ResourceRecord, ResourceCatalogLocation],
) -> tuple[ResourceRecord, ResourceCatalogLocation]:
    left_record, left_location = left
    right_record, right_location = right
    if (left_record.updated_at or "") != (right_record.updated_at or ""):
        return left if (left_record.updated_at or "") > (right_record.updated_at or "") else right
    if (left_record.created_at or "") != (right_record.created_at or ""):
        return left if (left_record.created_at or "") > (right_record.created_at or "") else right
    left_rank = _LAYER_RANK.get(left_location.reason, 99)
    right_rank = _LAYER_RANK.get(right_location.reason, 99)
    return left if left_rank <= right_rank else right


def _resolve_record_payload_if_needed(
    record: ResourceRecord,
    location: ResourceCatalogLocation,
) -> ResourceRecord:
    payload = record.payload if isinstance(record.payload, dict) else {}
    if payload.get("kind") != "payload_ref":
        return record
    return resolve_payload(record, location.path.parent)


def resolve_record(
    resource_ref: str,
    *,
    resource_type: str,
    catalog_context: CatalogExecutionContext,
) -> ResolvedResource:
    """Resolve a resource across session, bundle, and user catalog layers."""
    from .resource_handlers import require_resource_handler

    handler = require_resource_handler(resource_type)
    normalized_type = _normalise_resource_type(resource_type)
    agent_family = handler.resource_type == "agent"
    reference = str(resource_ref or "").strip()
    if not reference:
        raise ResourceCatalogError(
            RESOURCE_CATALOG_MISSING,
            "resource_ref is required to load a resource catalog record",
        )

    bundle_path = catalog_context.bundle_path
    if isinstance(bundle_path, str) and not bundle_path.strip():
        bundle_path = None
    bundle_id = catalog_context.bundle_id
    session_id = catalog_context.session_id
    home_only = catalog_context.home_only

    checked: list[str] = []

    if session_id:
        session_location = resolve_resource_catalog_path(
            bundle_path,
            bundle_id=bundle_id or None,
            session_id=session_id,
            scope="session",
        )
        checked.append(str(session_location.path))
        try:
            session_matches = _find_matches_at_location(
                session_location,
                reference,
                handler=handler,
                resource_type=resource_type,
                normalized_type=normalized_type,
                agent_family=agent_family,
            )
        except ResourceCatalogError as exc:
            if getattr(exc, "error_code", None) == RESOURCE_VERSION_UNSUPPORTED:
                logger.warning(
                    "resource-catalog: skipping unsupported session catalog at %s (%s)",
                    session_location.path,
                    exc,
                )
                session_matches = []
            else:
                raise
        if len(session_matches) == 1:
            record = _resolve_record_payload_if_needed(session_matches[0], session_location)
            return ResolvedResource(record=record, location=session_location)
        if len(session_matches) > 1:
            ids = ", ".join(record.resource_id for record in session_matches)
            raise ResourceCatalogError(
                RESOURCE_CATALOG_AMBIGUOUS,
                f"resource reference {reference!r} matches multiple records: {ids}",
            )

    base_matches: list[tuple[ResourceRecord, ResourceCatalogLocation]] = []
    version_errors: list[ResourceCatalogError] = []
    for location in iter_resource_catalog_locations(
        bundle_path,
        bundle_id=bundle_id or None,
        session_id=None,
        home_only=home_only,
    ):
        checked.append(str(location.path))
        try:
            matches = _find_matches_at_location(
                location,
                reference,
                handler=handler,
                resource_type=resource_type,
                normalized_type=normalized_type,
                agent_family=agent_family,
            )
        except ResourceCatalogError as exc:
            # A stale/unsupported secondary catalog (e.g. user home) must not
            # block resolution when another layer still has a valid match.
            if getattr(exc, "error_code", None) == RESOURCE_VERSION_UNSUPPORTED:
                logger.warning(
                    "resource-catalog: skipping unsupported catalog at %s (%s)",
                    location.path,
                    exc,
                )
                version_errors.append(exc)
                continue
            raise
        for record in matches:
            base_matches.append((record, location))

    deduped: dict[str, tuple[ResourceRecord, ResourceCatalogLocation]] = {}
    for record, location in base_matches:
        key = record.key()
        if key not in deduped:
            deduped[key] = (record, location)
        else:
            deduped[key] = _pick_cross_layer_winner(deduped[key], (record, location))

    winners = list(deduped.values())
    distinct_ids = {record.resource_id for record, _ in winners}
    if len(distinct_ids) > 1:
        ids = ", ".join(sorted(distinct_ids))
        raise ResourceCatalogError(
            RESOURCE_CATALOG_AMBIGUOUS,
            f"resource reference {reference!r} matches multiple records: {ids}",
        )
    if len(winners) == 1:
        record, location = winners[0]
        record = _resolve_record_payload_if_needed(record, location)
        return ResolvedResource(record=record, location=location)

    if version_errors:
        raise version_errors[0]

    raise ResourceCatalogError(
        RESOURCE_CATALOG_MISSING,
        f"resource reference {reference!r} was not found; checked: {', '.join(checked)}",
    )


def iter_resource_catalog_locations(
    bundle: Path | str | None = None,
    *,
    bundle_id: str | None = None,
    session_id: str | None = None,
    home_only: bool | None = None,
) -> Iterable[ResourceCatalogLocation]:
    """Yield read locations in priority order."""
    # Normalise: empty string → None so that the "no bundle" path uses the
    # home-directory fallback rather than resolving against CWD.
    if isinstance(bundle, str) and not bundle.strip():
        bundle = None
    effective_home_only = _is_home_only_forced() or bool(home_only)
    if session_id:
        yield resolve_resource_catalog_path(
            bundle,
            bundle_id=bundle_id,
            session_id=session_id,
            scope="session",
        )
    if bundle is not None and not effective_home_only:
        yield resolve_resource_catalog_path(bundle, bundle_id=bundle_id, scope="bundle")
    yield resolve_resource_catalog_path(
        bundle,
        bundle_id=bundle_id,
        scope="user",
        home_only=effective_home_only,
    )


def get_resource_record(
    resource_ref: str,
    *,
    resource_type: str,
    bundle_path: str | Path | None = None,
    bundle_id: str = "",
    catalog_context: CatalogExecutionContext | None = None,
) -> ResourceRecord:
    """Load one registered resource without assuming Agent semantics.

    Generic resources resolve by their stable ID. The built-in Agent handler
    additionally supports persisted names in the resource catalog.
    """
    if catalog_context is None:
        if isinstance(bundle_path, str) and not bundle_path.strip():
            bundle_path = None
        catalog_context = context_from_env(
            bundle_path=bundle_path,
            bundle_id=bundle_id,
        )
    return resolve_record(
        resource_ref,
        resource_type=resource_type,
        catalog_context=catalog_context,
    ).record


def get_agent_record(
    agent_id: str = "",
    *,
    agent_ref: str = "",
    bundle_path: str | Path | None = None,
    bundle_id: str = "",
    catalog_context: CatalogExecutionContext | None = None,
    resource_type: str = "",
) -> ResourceRecord:
    """Compatibility alias for the built-in Agent resource handler."""
    return get_resource_record(
        str(agent_ref or agent_id or ""),
        resource_type=resource_type or "agent",
        bundle_path=bundle_path,
        bundle_id=bundle_id,
        catalog_context=catalog_context,
    )


def resolve_agent_record(
    agent_id: str = "",
    *,
    agent_ref: str = "",
    bundle_path: str | Path | None = None,
    bundle_id: str = "",
    catalog_context: CatalogExecutionContext | None = None,
    resource_type: str = "",
) -> ResolvedResource:
    """Resolve an agent record with its catalog location for materialization."""
    if catalog_context is None:
        if isinstance(bundle_path, str) and not bundle_path.strip():
            bundle_path = None
        catalog_context = context_from_env(
            bundle_path=bundle_path,
            bundle_id=bundle_id,
        )
    return resolve_record(
        str(agent_ref or agent_id or ""),
        resource_type=resource_type or "agent",
        catalog_context=catalog_context,
    )


def _probe_writable(path: Path) -> bool | None:
    parent = path.parent
    if not parent.exists():
        return None
    return os.access(parent, os.W_OK)


def resource_error(
    code: str,
    message: str,
    *,
    resource_type: str = "",
    resource_id: str = "",
    retryable: bool = False,
    recovery: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "error_code": code,
        "message": message,
        "retryable": retryable,
    }
    if resource_type:
        payload["resource_type"] = _normalise_resource_type(resource_type)
    if resource_id:
        payload["resource_id"] = str(resource_id)
    if recovery:
        payload["recovery"] = recovery
    payload.update(extra)
    return payload


def build_resource_record_from_create(
    *,
    resource_id: str,
    resource_type: str = "agent",
    handle_field: str = "agent_id",
    bundle_id: str | None = None,
    source_tool: str = "",
    snapshot: Any = None,
    init_kwargs: Mapping[str, Any] | None = None,
    model: str = "",
    provider: str = "",
    class_name: str = "",
    module_name: str = "",
    factory: Mapping[str, Any] | None = None,
    invoke_method: str = "invoke",
    query_arg: str = "query",
    sdk_source_dir: str = "",
    sdk_version: str = "",
    metadata: Mapping[str, Any] | None = None,
    env_refs: list[str] | None = None,
    agent_catalog_entry: Mapping[str, Any] | None = None,
) -> ResourceRecord:
    """Build a :class:`ResourceRecord` directly from create-tool outputs.

    This is the canonical write-path constructor for persistence.
    """
    meta = dict(metadata or {})
    init = dict(init_kwargs or {})
    factory_meta = dict(factory or {})
    if not factory_meta and isinstance(meta.get("factory"), Mapping):
        factory_meta = dict(meta.get("factory") or {})
    dsl = snapshot if isinstance(snapshot, dict) else ({"value": snapshot} if snapshot is not None else {})
    materializer: dict[str, Any] = {
        "kind": "python_function" if factory_meta else "python_class",
        "module": str(factory_meta.get("module") or module_name or ""),
        "init_kwargs": init,
    }
    if factory_meta:
        materializer["name"] = str(factory_meta.get("name") or "")
    else:
        materializer["class_name"] = str(class_name or "")
    payload: dict[str, Any] = {
        "kind": "inline",
        "handle_field": handle_field or "agent_id",
        "dsl": dsl,
        "model": str(model or ""),
        "provider": str(provider or ""),
        "init_kwargs": init,
    }
    if agent_catalog_entry is not None:
        # Legacy bridge only — new writes omit this blob.
        payload["agent_catalog_entry"] = dict(agent_catalog_entry)
    return ResourceRecord(
        resource_type=resource_type or "agent",
        resource_id=str(resource_id),
        bundle_id=bundle_id,
        source_tool=source_tool or str(meta.get("source_tool") or ""),
        materializer=materializer,
        invoker={
            "kind": "python_method",
            "method": str(invoke_method or "invoke"),
            "input_param": str(query_arg or "query"),
        },
        payload=payload,
        sdk={
            "source_dir": str(sdk_source_dir or ""),
            "version": str(sdk_version or ""),
        },
        secrets={
            "policy": "env_refs_only",
            "env_refs": list(env_refs or meta.get("env_vars") or []),
        },
        metadata=meta,
    )


def agent_entry_to_resource_record(
    entry: Any,
    *,
    bundle_id: str | None = None,
    source_tool: str | None = None,
) -> ResourceRecord:
    """Convert an AgentCatalogEntry-like object to a ResourceRecord.

    Kept for tests and one-off migration scripts. Runtime create/read paths
    use :func:`build_resource_record_from_create` / :class:`ResourceCatalog`
    only.
    """
    metadata = dict(getattr(entry, "metadata", {}) or {})
    factory = metadata.get("factory") if isinstance(metadata.get("factory"), dict) else {}
    entry_dict = entry.to_dict() if hasattr(entry, "to_dict") else dict(metadata)
    return build_resource_record_from_create(
        resource_id=str(getattr(entry, "agent_id", "") or ""),
        resource_type=str(getattr(entry, "resource_type", "") or "agent"),
        handle_field=str(getattr(entry, "handle_field", "") or "agent_id"),
        bundle_id=bundle_id,
        source_tool=source_tool or str(metadata.get("source_tool") or ""),
        snapshot=dict(getattr(entry, "dsl", {}) or {}),
        init_kwargs=dict(getattr(entry, "init_kwargs", {}) or {}),
        model=str(getattr(entry, "model", "") or ""),
        provider=str(getattr(entry, "provider", "") or ""),
        class_name=str(getattr(entry, "class_name", "") or ""),
        module_name=str(getattr(entry, "module_name", "") or ""),
        factory=factory,
        invoke_method=str(getattr(entry, "invoke_method", "") or "invoke"),
        query_arg=str(getattr(entry, "query_arg", "") or "query"),
        sdk_source_dir=str(getattr(entry, "sdk_source_dir", "") or ""),
        sdk_version=str(getattr(entry, "sdk_version", "") or ""),
        metadata=metadata,
        env_refs=list(metadata.get("env_vars") or []),
        agent_catalog_entry=entry_dict,
    )


__all__ = [
    "DUAL_WRITE_ENV",
    "PAYLOAD_REF_ENV",
    "PAYLOAD_SPILL_THRESHOLD",
    "RESOURCE_CATALOG_MISSING",
    "RESOURCE_CATALOG_AMBIGUOUS",
    "RESOURCE_CATALOG_WRITE_FAILED",
    "RESOURCE_MATERIALIZE_FAILED",
    "RESOURCE_PAYLOAD_INVALID",
    "RESOURCE_PAYLOAD_REF_MISSING",
    "RESOURCE_SECRET_MISSING",
    "RESOURCE_VERSION_UNSUPPORTED",
    "SCHEMA_VERSION",
    "SESSION_ID_ENV",
    "CatalogExecutionContext",
    "ResourceCatalog",
    "ResourceCatalogError",
    "ResourceCatalogLocation",
    "ResourceRecord",
    "WriteResult",
    "agent_entry_to_resource_record",
    "build_resource_record_from_create",
    "context_from_env",
    "delete_resource_at",
    "format_resource_catalog_locations_block",
    "get_agent_record",
    "mutate_catalog",
    "get_resource_record",
    "iter_resource_catalog_locations",
    "normalize_resource_type",
    "payload_ref_filename",
    "plan_write_targets",
    "resolve_agent_record",
    "resolve_payload",
    "resolve_record",
    "ResolvedResource",
    "resolve_resource_catalog_path",
    "resource_error",
    "safe_payload_path",
    "spill_payload_if_needed",
    "write_record",
]
