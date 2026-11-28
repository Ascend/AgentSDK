# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Canonical checksummed JSON/base64; never pickle or torch.load."""

import base64
from dataclasses import fields, replace
import hashlib
import hmac
import json

from .schema import ReplayIntegrityError, ReplaySample, ReplayTensor, SCHEMA_VERSION

MAX_SAMPLE_BYTES = 24 * 1024 * 1024
CODEC = "aura.replay.json-base64.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayIntegrityError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(data, max_bytes=MAX_SAMPLE_BYTES):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, bytes) or len(data) > max_bytes:
        raise ReplayIntegrityError("Invalid or oversized JSON input")
    def reject(value):
        raise ReplayIntegrityError(f"Nonfinite JSON constant: {value}")
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=reject)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReplayIntegrityError("Invalid replay JSON") from error


def tensor_document(tensor):
    return {"dtype": tensor.dtype, "shape": list(tensor.shape), "byteorder": "little",
            "data": base64.b64encode(tensor.data).decode("ascii")}


def decode_tensor(value):
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "byteorder", "data"}:
        raise ReplayIntegrityError("Invalid tensor specification")
    if value["byteorder"] != "little" or not isinstance(value["data"], str):
        raise ReplayIntegrityError("Unsupported tensor encoding")
    try:
        return ReplayTensor(value["dtype"], value["shape"], base64.b64decode(value["data"], validate=True))
    except (ValueError, TypeError) as error:
        raise ReplayIntegrityError("Invalid replay tensor") from error


class ReplayCodec:
    @staticmethod
    def prepare(sample):
        """Validate a fresh immutable snapshot once and return its canonical bytes.

        Rebuild every tensor too: a deserialized/forged frozen dataclass must not
        bypass byte-shape, dtype or finite-value validation at the actor boundary.
        Avoid an encode->JSON parse->base64 decode->encode round trip.
        """
        if type(sample) is not ReplaySample:
            raise TypeError("Expected ReplaySample")
        from collections.abc import Mapping
        if not isinstance(sample.payload, Mapping) or not 3 <= len(sample.payload) <= 64:
            raise ValueError("payload must be a bounded ReplayTensor mapping")
        def snapshot_tensor(tensor):
            if not isinstance(tensor, ReplayTensor):
                raise TypeError("Expected ReplayTensor")
            return ReplayTensor(tensor.dtype, tensor.shape, tensor.data)
        snapshot = replace(sample, payload={k: snapshot_tensor(v) for k, v in sample.payload.items()},
                           old_log_probs=None if sample.old_log_probs is None else snapshot_tensor(sample.old_log_probs))
        # This is a new, exclusively owned validated object, never the caller's
        # sample. Store metadata is not part of the immutable payload document.
        object.__setattr__(snapshot, "created_seq", None)
        object.__setattr__(snapshot, "group_complete", None)
        body = {f.name: getattr(snapshot, f.name) for f in fields(snapshot)
                if f.name not in ("payload", "old_log_probs", "created_seq", "group_complete")}
        body["payload"] = {k: tensor_document(v) for k, v in snapshot.payload.items()}
        body["old_log_probs"] = None if snapshot.old_log_probs is None else tensor_document(snapshot.old_log_probs)
        # The exclusively owned body is validated/canonicalized once. Embed its
        # exact bytes in canonical key order instead of serializing it twice.
        body_bytes = canonical(body)
        checksum = hashlib.sha256(body_bytes).hexdigest()
        encoded = (b'{"checksum":"' + checksum.encode("ascii") + b'","codec":' + canonical(CODEC)
                   + b',"sample":' + body_bytes + b',"schema_version":' + canonical(SCHEMA_VERSION) + b'}')
        if len(encoded) > MAX_SAMPLE_BYTES:
            raise ValueError("ReplaySample exceeds encoded size limit")
        return snapshot, encoded, checksum

    @staticmethod
    def document(sample):
        if not isinstance(sample, ReplaySample):
            raise TypeError("Expected ReplaySample")
        sample = replace(sample)
        body = {f.name: getattr(sample, f.name) for f in fields(sample)
                if f.name not in ("payload", "old_log_probs", "created_seq", "group_complete")}
        body["payload"] = {k: tensor_document(v) for k, v in sample.payload.items()}
        body["old_log_probs"] = None if sample.old_log_probs is None else tensor_document(sample.old_log_probs)
        return {"codec": CODEC, "schema_version": SCHEMA_VERSION, "checksum": digest(body), "sample": body}

    @classmethod
    def encode(cls, sample):
        data = canonical(cls.document(sample))
        if len(data) > MAX_SAMPLE_BYTES:
            raise ValueError("ReplaySample exceeds encoded size limit")
        return data

    @staticmethod
    def decode(data):
        return ReplayCodec.decode_with_checksum(data)[0]

    @staticmethod
    def decode_with_checksum(data):
        """Return the validated sample and envelope checksum from one parse."""
        value = strict_json(data)
        if not isinstance(value, dict) or set(value) != {"codec", "schema_version", "checksum", "sample"}:
            raise ReplayIntegrityError("Invalid replay envelope")
        if value["codec"] != CODEC or type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ReplayIntegrityError("Unsupported replay codec/schema")
        body = value["sample"]
        checksum = value["checksum"]
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)
                or not hmac.compare_digest(checksum, digest(body))):
            raise ReplayIntegrityError("Replay payload checksum mismatch")
        expected = {f.name for f in fields(ReplaySample)} - {"created_seq", "group_complete"}
        if not isinstance(body, dict) or set(body) != expected or not isinstance(body["payload"], dict):
            raise ReplayIntegrityError("Invalid ReplaySample fields")
        try:
            body = dict(body)
            body["payload"] = {k: decode_tensor(v) for k, v in body["payload"].items()}
            if body["old_log_probs"] is not None:
                body["old_log_probs"] = decode_tensor(body["old_log_probs"])
            return ReplaySample(**body), checksum
        except (TypeError, ValueError, AttributeError) as error:
            raise ReplayIntegrityError("Invalid ReplaySample") from error
