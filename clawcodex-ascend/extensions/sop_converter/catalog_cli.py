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

"""Diagnostic CLI for resource catalogs.

Reads always use ``get_stored`` / on-disk records — never ``get()`` /
``_restore_tree`` — so stdout cannot leak restored plaintext secrets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .resource_catalog import (
    CatalogExecutionContext,
    ResourceCatalog,
    ResourceCatalogLocation,
    ResourceRecord,
    _layer_name,
    _pick_cross_layer_winner,
    context_from_env,
    delete_resource_at,
    iter_resource_catalog_locations,
    mutate_catalog,
    normalize_resource_type,
    resolve_payload,
    resolve_resource_catalog_path,
)


_VALID_SCOPES = ("effective", "session", "bundle", "user", "all")
_MUTATE_SCOPES = ("session", "bundle", "user", "all")


def run_catalog_command(args: list[str]) -> int:
    parser = _build_parser()
    try:
        ns = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code or 2)

    ctx = context_from_env(
        bundle_path=ns.bundle or None,
        session_id=ns.session if ns.session is not None else None,
    )
    if ns.session is not None:
        ctx = CatalogExecutionContext(
            bundle_path=ctx.bundle_path,
            bundle_id=ctx.bundle_id,
            home_only=ctx.home_only,
            session_id=str(ns.session or ""),
            dual_write=ctx.dual_write,
        )

    try:
        if ns.subcommand == "list":
            rows = _cmd_list(ctx, scope=ns.scope, resource_type=ns.type)
            return _emit(rows, as_json=ns.json)
        if ns.subcommand == "get":
            row = _cmd_get(
                ctx,
                scope=ns.scope,
                resource_type=ns.type,
                resource_id=ns.id,
                resolve_payload_flag=ns.resolve_payload,
            )
            if row is None:
                print("error: resource not found", file=sys.stderr)
                return 1
            return _emit(row, as_json=ns.json)
        if ns.subcommand == "latest":
            row = _cmd_latest(ctx, scope=ns.scope, resource_type=ns.type)
            if row is None:
                print("error: no active resource found", file=sys.stderr)
                return 1
            return _emit(row, as_json=ns.json)
        if ns.subcommand == "delete":
            if ns.scope not in _MUTATE_SCOPES:
                print(
                    "error: delete requires --scope session|bundle|user|all",
                    file=sys.stderr,
                )
                return 2
            result = _cmd_delete(
                ctx,
                scope=ns.scope,
                resource_type=ns.type,
                resource_id=ns.id,
            )
            return _emit(result, as_json=ns.json)
        if ns.subcommand == "mark-failed":
            if ns.scope not in _MUTATE_SCOPES:
                print(
                    "error: mark-failed requires --scope session|bundle|user|all",
                    file=sys.stderr,
                )
                return 2
            result = _cmd_mark_failed(
                ctx,
                scope=ns.scope,
                resource_type=ns.type,
                resource_id=ns.id,
                reason=ns.reason or "",
            )
            return _emit(result, as_json=ns.json)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected I/O
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"error: unknown subcommand {ns.subcommand!r}", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawcodex sop catalog",
        description="Inspect and manage resource catalogs (storage/redacted view).",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bundle", default="", help="Bundle path for catalog resolution")
        p.add_argument("--session", default=None, help="Session id (overrides CLAWCODEX_SESSION_ID)")
        p.add_argument(
            "--scope",
            default="effective",
            choices=_VALID_SCOPES,
            help="Catalog layer scope (default: effective)",
        )
        p.add_argument(
            "--type",
            default="",
            help="Resource type filter (optional for get: omit to look up by --id alone)",
        )
        p.add_argument("--id", default="", help="Resource id")
        p.add_argument("--json", action="store_true", help="Emit JSON")

    list_p = sub.add_parser("list", help="List stored catalog records")
    add_common(list_p)

    get_p = sub.add_parser("get", help="Get one stored catalog record")
    add_common(get_p)
    get_p.add_argument(
        "--resolve-payload",
        action="store_true",
        help="Inline payload_ref file contents (still redacted; no secret restore)",
    )

    latest_p = sub.add_parser("latest", help="Latest active record for a type")
    add_common(latest_p)

    delete_p = sub.add_parser("delete", help="Delete a catalog record")
    add_common(delete_p)

    fail_p = sub.add_parser("mark-failed", help="Mark a catalog record failed")
    add_common(fail_p)
    fail_p.add_argument("--reason", default="", help="Failure reason")

    return parser


def _emit(payload: Any, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _locations_for_scope(
    ctx: CatalogExecutionContext,
    scope: str,
) -> list[tuple[str, ResourceCatalogLocation]]:
    if scope == "all":
        locs = list(
            iter_resource_catalog_locations(
                ctx.bundle_path,
                bundle_id=ctx.bundle_id or None,
                session_id=ctx.session_id or None,
                home_only=ctx.home_only,
            )
        )
        return [(_layer_name(loc), loc) for loc in locs]

    if scope == "session":
        if not ctx.session_id:
            raise ValueError("--scope session requires --session or CLAWCODEX_SESSION_ID")
        loc = resolve_resource_catalog_path(
            ctx.bundle_path,
            bundle_id=ctx.bundle_id or None,
            session_id=ctx.session_id,
            scope="session",
        )
        return [("session", loc)]

    if scope == "bundle":
        loc = resolve_resource_catalog_path(
            ctx.bundle_path,
            bundle_id=ctx.bundle_id or None,
            scope="bundle",
        )
        return [("bundle", loc)]

    if scope == "user":
        loc = resolve_resource_catalog_path(
            ctx.bundle_path,
            bundle_id=ctx.bundle_id or None,
            scope="user",
            home_only=False,
        )
        return [("user", loc)]

    # effective: session → bundle → user (iteration order for later resolve)
    locs = list(
        iter_resource_catalog_locations(
            ctx.bundle_path,
            bundle_id=ctx.bundle_id or None,
            session_id=ctx.session_id or None,
            home_only=ctx.home_only,
        )
    )
    return [(_layer_name(loc), loc) for loc in locs]


def _load_stored(
    location: ResourceCatalogLocation,
) -> ResourceCatalog:
    return ResourceCatalog.load(location.path)


def _record_row(
    record: ResourceRecord,
    layer: str,
    *,
    catalog_dir: Path | None = None,
    resolve_payload_flag: bool = False,
) -> dict[str, Any]:
    view = record
    if resolve_payload_flag and catalog_dir is not None:
        view = resolve_payload(record, catalog_dir, restore=False)
    data = view.to_dict()
    data["layer"] = layer
    return data


def _type_matches(type_filter: str, stored_type: str) -> bool:
    """Match catalog type filters, including Agent-family CLI aliases.

    ``--type agent`` / ``agentconfig`` resolve the same family that
    ``resolve_record`` / ``get_resource_handler`` already treat as Agent
    (exact key, ``*agent`` / ``*agentconfig`` suffix).
    """
    filt = normalize_resource_type(type_filter or "")
    if not filt:
        return True
    stored = normalize_resource_type(stored_type or "")
    if stored == filt:
        return True
    if filt in {"agent", "agentconfig"} and stored.endswith(("agent", "agentconfig")):
        return True
    return False


def _iter_layer_records(
    ctx: CatalogExecutionContext,
    scope: str,
    *,
    resource_type: str = "",
) -> list[tuple[str, ResourceCatalogLocation, ResourceRecord]]:
    """Yield stored records for the given scope (no secret restore)."""
    out: list[tuple[str, ResourceCatalogLocation, ResourceRecord]] = []
    for layer, loc in _locations_for_scope(ctx, scope if scope != "effective" else "all"):
        # For explicit scopes use that layer only; for effective we still scan all
        # then reduce — handled by callers via _effective_winners.
        if scope not in ("effective", "all") and layer != scope:
            continue
        cat = _load_stored(loc)
        for key in cat.list_keys():
            rec = cat.records[key]
            stored = cat.get_stored(rec.resource_type, rec.resource_id)
            if stored is None:
                continue
            if not _type_matches(resource_type, stored.resource_type):
                continue
            out.append((layer, loc, stored))
    return out


def _effective_winners(
    rows: list[tuple[str, ResourceCatalogLocation, ResourceRecord]],
    *,
    session_id: str,
) -> list[tuple[str, ResourceCatalogLocation, ResourceRecord]]:
    """Apply §2 cross-layer rules per resource key (storage view)."""
    by_key: dict[str, list[tuple[ResourceRecord, ResourceCatalogLocation]]] = {}
    for _layer, loc, rec in rows:
        by_key.setdefault(rec.key(), []).append((rec, loc))

    winners: list[tuple[str, ResourceCatalogLocation, ResourceRecord]] = []
    for _key, candidates in by_key.items():
        session_hits = [(rec, loc) for rec, loc in candidates if _layer_name(loc) == "session"]
        if session_id and session_hits:
            # Session shadows base layers for this key.
            rec, loc = session_hits[0]
            winners.append((_layer_name(loc), loc, rec))
            continue
        base = [(rec, loc) for rec, loc in candidates if _layer_name(loc) != "session"]
        if not base:
            continue
        winner = base[0]
        for item in base[1:]:
            winner = _pick_cross_layer_winner(winner, item)
        rec, loc = winner
        winners.append((_layer_name(loc), loc, rec))
    return winners


def _cmd_list(
    ctx: CatalogExecutionContext,
    *,
    scope: str,
    resource_type: str,
) -> list[dict[str, Any]]:
    scan_scope = "all" if scope == "effective" else scope
    rows = _iter_layer_records(ctx, scan_scope, resource_type=resource_type)
    if scope == "effective":
        rows = _effective_winners(rows, session_id=ctx.session_id)
    return [
        _record_row(rec, layer)
        for layer, _loc, rec in sorted(rows, key=lambda t: (t[2].resource_type, t[2].resource_id, t[0]))
    ]


def _cmd_get(
    ctx: CatalogExecutionContext,
    *,
    scope: str,
    resource_type: str,
    resource_id: str,
    resolve_payload_flag: bool,
) -> dict[str, Any] | None:
    if not resource_id:
        raise ValueError("--id is required for get")
    scan_scope = "all" if scope == "effective" else scope
    rows = [
        (layer, loc, rec)
        for layer, loc, rec in _iter_layer_records(ctx, scan_scope, resource_type=resource_type)
        if rec.resource_id == str(resource_id)
    ]
    if scope == "effective":
        rows = _effective_winners(rows, session_id=ctx.session_id)
    if not rows:
        return None
    # Without --type, the same resource_id may exist under multiple types.
    if not str(resource_type or "").strip():
        distinct_types = sorted({normalize_resource_type(rec.resource_type) for _l, _loc, rec in rows})
        if len(distinct_types) > 1:
            raise ValueError(f"ambiguous resource id {resource_id!r}; pass --type one of: " + ", ".join(distinct_types))
    if scope == "all":
        # Return first match as primary plus note — keep simple: one object if
        # single, else list is unusual for get; pick first by layer priority.
        rows = sorted(rows, key=lambda t: {"session": 0, "bundle": 1, "user": 2}.get(t[0], 9))
    layer, loc, rec = rows[0]
    return _record_row(
        rec,
        layer,
        catalog_dir=loc.path.parent,
        resolve_payload_flag=resolve_payload_flag,
    )


def _cmd_latest(
    ctx: CatalogExecutionContext,
    *,
    scope: str,
    resource_type: str,
) -> dict[str, Any] | None:
    if not resource_type:
        raise ValueError("--type is required for latest")
    scan_scope = "all" if scope == "effective" else scope
    rows = _iter_layer_records(ctx, scan_scope, resource_type=resource_type)
    if scope == "effective":
        rows = _effective_winners(rows, session_id=ctx.session_id)
    active = [(layer, loc, rec) for layer, loc, rec in rows if (rec.status or "active") == "active"]
    if not active:
        return None
    layer, loc, rec = max(
        active,
        key=lambda t: (t[2].updated_at or "", t[2].created_at or ""),
    )
    return _record_row(rec, layer)


def _mutate_locations(
    ctx: CatalogExecutionContext,
    scope: str,
) -> list[tuple[str, ResourceCatalogLocation]]:
    return _locations_for_scope(ctx, scope)


def _cmd_delete(
    ctx: CatalogExecutionContext,
    *,
    scope: str,
    resource_type: str,
    resource_id: str,
) -> dict[str, Any]:
    if not resource_type or not resource_id:
        raise ValueError("--type and --id are required for delete")
    deleted_layers: list[str] = []
    for layer, loc in _mutate_locations(ctx, scope):
        if delete_resource_at(loc.path, resource_type, resource_id):
            deleted_layers.append(layer)
    return {"deleted": bool(deleted_layers), "layers": deleted_layers}


def _cmd_mark_failed(
    ctx: CatalogExecutionContext,
    *,
    scope: str,
    resource_type: str,
    resource_id: str,
    reason: str,
) -> dict[str, Any]:
    if not resource_type or not resource_id:
        raise ValueError("--type and --id are required for mark-failed")
    updated_layers: list[str] = []
    last: ResourceRecord | None = None
    for layer, loc in _mutate_locations(ctx, scope):
        found = False

        def mutator(cat: ResourceCatalog) -> None:
            nonlocal found, last
            key_rec = cat.get_stored(resource_type, resource_id)
            if key_rec is None:
                return
            cat.mark_failed(resource_type, resource_id, reason=reason)
            found = True
            last = cat.get_stored(resource_type, resource_id)

        if not loc.path.exists():
            continue
        mutate_catalog(loc.path, mutator, merge=True)
        if found:
            updated_layers.append(layer)
    if not updated_layers:
        raise ValueError(f"resource {resource_type}:{resource_id} not found in scope={scope}")
    assert last is not None
    row = _record_row(last, updated_layers[-1])
    row["layers"] = updated_layers
    return row


__all__ = ["run_catalog_command"]
