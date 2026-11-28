# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Shared persistence validation and snapshot-bound query contracts."""

import base64
import copy
from dataclasses import asdict, dataclass, replace

from .codec import ReplayCodec, canonical, digest, strict_json
from .schema import (ReplayConflictError, ReplayGroup, ReplayIntegrityError, ReplayPage,
                     ReplayQuery, ReplayReceipt, identifier, integer)

MAX_BATCH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Entry:
    sample: object
    checksum: str
    encoded: bytes
    seq: int


def stored_sample(entry, complete):
    """Attach store-owned metadata to an already validated immutable snapshot.

    Only internal Entry objects produced by boundary validation or checksum-
    verified decode reach this helper. Do not rewalk every token just to change
    two metadata fields; never change the stored snapshot or its payload.
    """
    integer(entry.seq, "created_seq", 1)
    if type(complete) is not bool:
        raise ValueError("group_complete must be bool")
    sample = copy.copy(entry.sample)
    object.__setattr__(sample, "created_seq", entry.seq)
    object.__setattr__(sample, "group_complete", complete)
    return sample


def prepare_batch(samples, store_id, max_batch_size=256, max_batch_bytes=MAX_BATCH_BYTES):
    if not isinstance(samples, (list, tuple)) or not 1 <= len(samples) <= max_batch_size:
        raise ValueError("Batch must be a nonempty bounded list/tuple")
    result, total, ids = [], 0, set()
    for sample in samples:
        sample, encoded, checksum = ReplayCodec.prepare(sample)
        total += len(encoded)
        if total > max_batch_bytes:
            raise ValueError("Batch byte limit exceeded")
        if sample.source_store_id != store_id:
            raise ReplayConflictError("Sample source_store_id does not match the destination")
        if sample.trajectory_id in ids:
            raise ReplayConflictError("Duplicate trajectory_id in one request")
        ids.add(sample.trajectory_id)
        result.append(Entry(sample, checksum, encoded, 0))
    return result, digest([(e.sample.trajectory_id, e.checksum) for e in result])


def plan_write(entries, existing, store_id, request_id, request_checksum, next_seq):
    merged = dict(existing)
    new, refs = [], []
    affected = {e.sample.group_id for e in entries}
    for e in entries:
        key = e.sample.trajectory_id
        if key in merged:
            saved = merged[key]
            if e.checksum != saved.checksum or e.encoded != saved.encoded:
                raise ReplayConflictError(f"Immutable trajectory conflict: {key}")
        else:
            saved = replace(e, seq=next_seq)
            merged[key] = saved
            new.append(saved)
            next_seq += 1
        refs.append((key, saved.checksum, saved.seq))
    groups = {}
    grouped_members = {group_id: [] for group_id in affected}
    for entry in merged.values():
        if entry.sample.group_id in grouped_members:
            grouped_members[entry.sample.group_id].append(entry)
    for group_id, members in grouped_members.items():
        group = ReplayGroup(tuple(e.sample for e in members))
        complete_seq = max(e.seq for e in members) if group.complete else None
        groups[group_id] = (group.samples[0].group_size, group.samples[0].train_eligible, complete_seq)
    receipt = ReplayReceipt(store_id, request_id, request_checksum, tuple(refs), len(new), len(entries) - len(new))
    return new, groups, receipt


def receipt_dump(receipt):
    body = asdict(receipt)
    return canonical({"receipt": body, "checksum": digest(body)})


def receipt_load(data):
    value = strict_json(data, 1024 * 1024)
    if not isinstance(value, dict) or set(value) != {"receipt", "checksum"} or digest(value["receipt"]) != value["checksum"]:
        raise ReplayIntegrityError("Invalid receipt checksum")
    try:
        body = dict(value["receipt"])
        body["sample_refs"] = tuple(tuple(ref) for ref in body["sample_refs"])
        receipt = ReplayReceipt(**body)
        identifier(receipt.store_id, "store_id")
        identifier(receipt.request_id, "request_id")
        if (not isinstance(receipt.request_checksum, str) or len(receipt.request_checksum) != 64
                or any(c not in "0123456789abcdef" for c in receipt.request_checksum)):
            raise ValueError("Invalid request checksum")
        integer(receipt.inserted_count, "inserted_count")
        integer(receipt.duplicate_count, "duplicate_count")
        if receipt.inserted_count + receipt.duplicate_count != len(receipt.sample_refs) or type(receipt.replayed) is not bool:
            raise ValueError("Invalid receipt counts")
        for key, checksum, seq in receipt.sample_refs:
            identifier(key, "trajectory_id")
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("Invalid checksum")
            integer(seq, "created_seq", 1)
        return receipt
    except (ValueError, TypeError, KeyError) as error:
        raise ReplayIntegrityError("Invalid receipt") from error


def retry_receipt(receipt, checksum, store_id, request_id):
    if receipt.request_checksum != checksum:
        raise ReplayConflictError("request_id reused with different content or ordering")
    if receipt.store_id != store_id or receipt.request_id != request_id:
        raise ReplayIntegrityError("Receipt identity mismatch")
    return replace(receipt, inserted_count=0, duplicate_count=len(receipt.sample_refs), replayed=True)


def query_value(query=None, **filters):
    if query is not None and filters:
        raise ValueError("Use ReplayQuery or keyword filters, not both")
    if query is None:
        return ReplayQuery(**filters)
    if not isinstance(query, ReplayQuery):
        raise TypeError("query must be ReplayQuery")
    return replace(query)


def page_bounds(query, cursor, store_id, current_seq):
    if cursor is None:
        return 0, current_seq
    if not isinstance(cursor, str) or len(cursor) > 8192:
        raise ValueError("Invalid query cursor")
    try:
        value = strict_json(base64.b64decode(cursor, altchars=b"-_", validate=True), 4096)
        if set(value) != {"version", "store_id", "query", "snapshot", "after"} or type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("Unknown cursor schema")
        if value["store_id"] != store_id or value["query"] != digest(asdict(query)):
            raise ValueError("Cursor is bound to another store or query")
        integer(value["snapshot"], "snapshot", maximum=current_seq)
        integer(value["after"], "after", maximum=value["snapshot"])
        return value["after"], value["snapshot"]
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Invalid or incompatible query cursor") from error


def page_result(samples, query, snapshot, store_id):
    selected = tuple(samples[:query.limit])
    cursor = None
    if selected and len(samples) > query.limit:
        cursor = base64.urlsafe_b64encode(canonical({"version": 1, "store_id": store_id,
            "query": digest(asdict(query)), "snapshot": snapshot, "after": selected[-1].created_seq})).decode("ascii")
    return ReplayPage(selected, cursor, snapshot)


def matches(sample, query):
    for key, value in asdict(query).items():
        if value is None or key == "limit":
            continue
        if key.endswith("_min"):
            if getattr(sample, key[:-4]) < value:
                return False
        elif key.endswith("_max"):
            if getattr(sample, key[:-4]) > value:
                return False
        elif getattr(sample, key) != value:
            return False
    return True


def trajectory_ids(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 256:
        raise ValueError("Expected one to 256 trajectory IDs")
    values = tuple(identifier(value, "trajectory_id") for value in values)
    if len(set(values)) != len(values):
        raise ValueError("Duplicate requested trajectory IDs")
    return values
