# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Single-writer replay JSONL journal; a checksummed newline is a commit marker."""

from dataclasses import replace
from pathlib import Path
import hashlib
import os
import threading
import uuid
import warnings

from .codec import ReplayCodec, canonical, digest, strict_json
from .schema import ReplayIntegrityError, SCHEMA_VERSION, identifier, integer
from .store_common import (MAX_BATCH_BYTES, matches, page_bounds, page_result, plan_write,
                           prepare_batch, query_value, receipt_dump, receipt_load, retry_receipt, trajectory_ids, stored_sample)

MAX_JOURNAL_RECORD = 96 * 1024 * 1024


class JsonReplayStore:
    def __init__(self, path, *, read_only=False, store_id=None, max_batch_size=256, max_batch_bytes=MAX_BATCH_BYTES):
        if type(read_only) is not bool:
            raise ValueError("read_only must be bool")
        if store_id is not None:
            identifier(store_id, "store_id")
        integer(max_batch_size, "max_batch_size", 1, 256)
        integer(max_batch_bytes, "max_batch_bytes", 1, MAX_BATCH_BYTES)
        self.path = Path(path)
        self.read_only, self.max_batch_size, self.max_batch_bytes = read_only, max_batch_size, max_batch_bytes
        self._mutex = threading.RLock()
        self._writer_lock = None
        self._file = None
        self._closed = False
        self._failed = False
        self.ignored_tail_bytes = 0
        self._entries, self._requests = {}, {}
        self._group_members, self._group_complete_seq = {}, {}
        self._seq = 0
        try:
            if not read_only:
                self._acquire_writer()
            if self.path.exists():
                self._load()
                if store_id is not None and store_id != self.store_id:
                    raise ReplayIntegrityError("store_id mismatch")
            elif read_only:
                raise FileNotFoundError(self.path)
            else:
                self.store_id = store_id or uuid.uuid4().hex
                self._file = self.path.open("ab")
                self._append({"kind": "manifest", "schema_version": SCHEMA_VERSION, "store_id": self.store_id})
                if os.name != "nt":
                    fd = os.open(str(self.path.parent), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            if not read_only and self._file is None:
                self._file = self.path.open("ab")
        except BaseException:
            self.close()
            raise

    def _acquire_writer(self):
        lock = self.path.with_name(self.path.name + ".lock").open("a+b")
        self._writer_lock = lock
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("JSON replay already has a writer") from error
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("JSON replay already has a writer") from error

    def _append(self, body):
        # Identical canonical envelope/checksum and durable newline, with one
        # encoding of the potentially large body. Flush/fsync remain below.
        body_bytes = canonical(body)
        checksum = hashlib.sha256(body_bytes).hexdigest().encode("ascii")
        line = b'{"checksum":"' + checksum + b'","record":' + body_bytes + b'}\n'
        if len(line) > MAX_JOURNAL_RECORD:
            raise ValueError("Replay journal record limit exceeded")
        try:
            if self._file.write(line) != len(line):
                raise OSError("Short replay journal write")
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._failed = True
            raise

    def _load(self):
        with self.path.open("rb") as file:
            number = 0
            while True:
                line = file.readline(MAX_JOURNAL_RECORD + 1)
                if not line:
                    break
                if len(line) > MAX_JOURNAL_RECORD:
                    raise ReplayIntegrityError("Oversized replay journal record")
                if not line.endswith(b"\n"):
                    self.ignored_tail_bytes = len(line)
                    warnings.warn("Ignoring uncommitted JSON replay tail; original file preserved", RuntimeWarning)
                    break
                value = strict_json(line, MAX_JOURNAL_RECORD)
                if not isinstance(value, dict) or set(value) != {"record", "checksum"} or digest(value["record"]) != value["checksum"]:
                    raise ReplayIntegrityError("Replay journal checksum mismatch")
                body = value["record"]
                if number == 0:
                    if (not isinstance(body, dict) or set(body) != {"kind", "schema_version", "store_id"}
                            or body["kind"] != "manifest" or type(body["schema_version"]) is not int
                            or body["schema_version"] != SCHEMA_VERSION):
                        raise ReplayIntegrityError("Unsupported JSON replay schema")
                    self.store_id = identifier(body["store_id"], "store_id")
                else:
                    self._load_commit(body)
                number += 1
            if not number:
                raise ReplayIntegrityError("Missing committed replay manifest")

    def _load_commit(self, body):
        if not isinstance(body, dict) or set(body) != {"kind", "request_id", "samples", "receipt"} or body["kind"] != "commit":
            raise ReplayIntegrityError("Invalid JSON commit record")
        if not isinstance(body["samples"], list) or not 1 <= len(body["samples"]) <= 256:
            raise ReplayIntegrityError("Invalid journal sample count")
        samples = [ReplayCodec.decode(s) for s in body["samples"]]
        entries, checksum = prepare_batch(samples, self.store_id, 256, MAX_BATCH_BYTES)
        request_id = identifier(body["request_id"], "request_id")
        if request_id in self._requests:
            raise ReplayIntegrityError("Repeated committed request in journal")
        new, _, receipt = plan_write(entries, self._entries, self.store_id, request_id, checksum, self._seq + 1)
        if receipt != receipt_load(body["receipt"]):
            raise ReplayIntegrityError("Persisted receipt does not match journal sequence/content")
        self._apply(new, receipt)

    def _apply(self, entries, receipt):
        for entry in entries:
            self._entries[entry.sample.trajectory_id] = entry
            group = entry.sample.group_id
            members = self._group_members.setdefault(group, [])
            members.append(entry.sample.trajectory_id)
            if len(members) == entry.sample.group_size:
                self._group_complete_seq[group] = entry.seq
            self._seq = max(self._seq, entry.seq)
        self._requests[receipt.request_id] = receipt

    def _open(self):
        if self._closed:
            raise RuntimeError("Replay store is closed")

    def get_replay_manifest(self):
        self._open()
        return {"store_id": self.store_id, "schema_version": SCHEMA_VERSION, "backend": "json",
                "read_only": self.read_only, "ignored_tail_bytes": self.ignored_tail_bytes}

    def append_trajectories(self, samples, request_id):
        identifier(request_id, "request_id")
        with self._mutex:
            self._open()
            if self.read_only:
                raise PermissionError("Read-only replay store")
            if self.ignored_tail_bytes or self._failed:
                raise ReplayIntegrityError("Uncertain/torn journal write; preserve file and repair a copy before appending")
            entries, checksum = prepare_batch(samples, self.store_id, self.max_batch_size, self.max_batch_bytes)
            if request_id in self._requests:
                return retry_receipt(self._requests[request_id], checksum, self.store_id, request_id)
            new, _, receipt = plan_write(entries, self._entries, self.store_id, request_id, checksum, self._seq + 1)
            self._append({"kind": "commit", "request_id": request_id,
                          "samples": [e.encoded.decode("utf-8") for e in entries],
                          "receipt": receipt_dump(receipt).decode("utf-8")})
            self._apply(new, receipt)
            return receipt

    def query_page(self, query=None, *, cursor=None, **filters):
        q = query_value(query, **filters)
        with self._mutex:
            self._open()
            after, snapshot = page_bounds(q, cursor, self.store_id, self._seq)
            if q.trajectory_id is not None:
                candidate = self._entries.get(q.trajectory_id)
                entries = () if candidate is None else (candidate,)
            elif q.group_id is not None:
                entries = (self._entries[key] for key in self._group_members.get(q.group_id, ()))
            else:
                entries = self._entries.values()
            selected = []
            for entry in entries:
                if not after < entry.seq <= snapshot:
                    continue
                complete_seq = self._group_complete_seq.get(entry.sample.group_id)
                if q.train_eligible is not None:
                    if entry.sample.train_eligible != q.train_eligible:
                        continue
                    if q.train_eligible and not (complete_seq is not None and complete_seq <= snapshot):
                        continue
                if not matches(entry.sample, replace(q, train_eligible=None)):
                    continue
                selected.append(stored_sample(entry, complete_seq is not None and complete_seq <= snapshot))
                if len(selected) > q.limit:
                    break
            return page_result(selected, q, snapshot, self.store_id)

    def query(self, query=None, **filters):
        return list(self.query_page(query, **filters).samples)

    def get_trajectory(self, trajectory_id, source_store_id=None):
        return self.get_trajectories([trajectory_id], source_store_id)[0]

    def get_trajectories(self, ids, source_store_id=None):
        ids = trajectory_ids(ids)
        if source_store_id is not None:
            identifier(source_store_id, "source_store_id")
        with self._mutex:
            self._open()
            if source_store_id is not None and source_store_id != self.store_id:
                return [None] * len(ids)
            result = []
            for key in ids:
                entry = self._entries.get(key)
                result.append(None if entry is None else stored_sample(entry,
                    entry.sample.group_id in self._group_complete_seq))
            return result

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._writer_lock is not None:
            self._writer_lock.close()  # Native lock released even after process death.
            self._writer_lock = None
        self._closed = True
