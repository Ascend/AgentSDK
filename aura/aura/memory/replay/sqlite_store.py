# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Transactional replay tables, independent of legacy episode object serialization."""

from dataclasses import replace
from collections import OrderedDict
from pathlib import Path
import sqlite3
import threading
import uuid

from .codec import ReplayCodec
from .schema import ReplayIntegrityError, SCHEMA_VERSION, identifier, integer
from .store_common import (Entry, MAX_BATCH_BYTES, page_bounds, page_result, plan_write,
                           prepare_batch, query_value, receipt_dump, receipt_load, retry_receipt, trajectory_ids, stored_sample)


class SQLiteReplayStore:
    def __init__(self, path, *, read_only=False, store_id=None, max_batch_size=256, max_batch_bytes=MAX_BATCH_BYTES,
                 immutable=False):
        if str(path) == ":memory:":
            raise ValueError("Durable replay receipts require a persistent file, not :memory:")
        if type(read_only) is not bool:
            raise ValueError("read_only must be bool")
        if type(immutable) is not bool or (immutable and not read_only):
            raise ValueError('Immutable SQLite requires read_only=True')
        if immutable and any(Path(str(path)+x).exists() for x in ('-wal', '-shm', '-journal')):
            raise ReplayIntegrityError('Frozen SQLite reuse must not have WAL/SHM/journal sidecars')
        if store_id is not None:
            identifier(store_id, "store_id")
        integer(max_batch_size, "max_batch_size", 1, 256)
        integer(max_batch_bytes, "max_batch_bytes", 1, MAX_BATCH_BYTES)
        self.read_only, self.max_batch_size, self.max_batch_bytes = read_only, max_batch_size, max_batch_bytes
        self._lock = threading.RLock()
        self._requested_store_id = store_id
        # Memoize decoding only: every read still fetches the current database
        # row and checks its indexed metadata. Keys are complete payload bytes,
        # so any payload change forces the original checksum/schema validation.
        self._decoded = OrderedDict()
        self._decoded_bytes = 0
        self.conn = None
        uri = Path(path).resolve().as_uri() + ("?mode=ro" if read_only else "?mode=rwc")
        if immutable:
            uri += '&immutable=1'
        try:
            self.conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False, timeout=30)
            self.conn.row_factory = sqlite3.Row
            present = self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='replay_store_manifest'").fetchone()
            if present:
                row = self.conn.execute("SELECT schema_version,store_id FROM replay_store_manifest WHERE id=1").fetchone()
                if row is None or row[0] != SCHEMA_VERSION:
                    raise ReplayIntegrityError("Unsupported replay store schema")
                identifier(row[1], "persisted store_id")
                if store_id is not None and store_id != row[1]:
                    raise ReplayIntegrityError("store_id mismatch")
                self.store_id = row[1]
            elif read_only:
                raise ReplayIntegrityError("Read-only source has no replay schema; migration is forbidden")
            else:
                self.store_id = store_id or uuid.uuid4().hex
            if not read_only:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=FULL")
                self.conn.execute("PRAGMA foreign_keys=ON")
                self._init_schema()
        except BaseException:
            self.close()
            raise

    def _init_schema(self):
        statements = [
            "CREATE TABLE IF NOT EXISTS replay_store_manifest(id INTEGER PRIMARY KEY CHECK(id=1),schema_version INTEGER NOT NULL,store_id TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS replay_groups(group_id TEXT PRIMARY KEY,group_size INTEGER NOT NULL,train_eligible INTEGER NOT NULL,complete_seq INTEGER)",
            """CREATE TABLE IF NOT EXISTS replay_samples(created_seq INTEGER PRIMARY KEY,trajectory_id TEXT NOT NULL UNIQUE,
                group_id TEXT NOT NULL REFERENCES replay_groups(group_id),sample_index INTEGER NOT NULL,prompt_id TEXT NOT NULL,
                reward REAL NOT NULL,weight_version INTEGER NOT NULL,policy_lineage_id TEXT NOT NULL,
                source_experiment_id TEXT NOT NULL,source_run_id TEXT NOT NULL,train_eligible INTEGER NOT NULL,
                checksum TEXT NOT NULL,payload BLOB NOT NULL,UNIQUE(group_id,sample_index))""",
            "CREATE TABLE IF NOT EXISTS replay_requests(request_id TEXT PRIMARY KEY,receipt BLOB NOT NULL)",
            "CREATE INDEX IF NOT EXISTS idx_replay_prompt ON replay_samples(prompt_id,created_seq)",
            "CREATE INDEX IF NOT EXISTS idx_replay_reward ON replay_samples(reward,created_seq)",
            "CREATE INDEX IF NOT EXISTS idx_replay_policy ON replay_samples(policy_lineage_id,weight_version,created_seq)",
            "CREATE INDEX IF NOT EXISTS idx_replay_source ON replay_samples(source_experiment_id,source_run_id,created_seq)",
        ]
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                self.conn.execute(statement)
            self.conn.execute("INSERT OR IGNORE INTO replay_store_manifest VALUES(1,?,?)", (SCHEMA_VERSION, self.store_id))
            row = self.conn.execute("SELECT schema_version,store_id FROM replay_store_manifest WHERE id=1").fetchone()
            if row[0] != SCHEMA_VERSION or (self._requested_store_id is not None and row[1] != self._requested_store_id):
                raise ReplayIntegrityError("Concurrent manifest/schema conflict")
            self.store_id = row[1]
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def get_replay_manifest(self):
        self._open()
        return {"store_id": self.store_id, "schema_version": SCHEMA_VERSION, "backend": "sqlite", "read_only": self.read_only}

    def _open(self):
        if self.conn is None:
            raise RuntimeError("Replay store is closed")

    def _entry(self, row):
        payload = row['payload']
        encoded = payload.encode('utf-8') if isinstance(payload, str) else payload
        decoded = self._decoded.get(encoded)
        if decoded is None:
            decoded = ReplayCodec.decode_with_checksum(encoded)
            self._decoded[encoded] = decoded
            self._decoded_bytes += len(encoded)
            while self._decoded_bytes > MAX_BATCH_BYTES:
                old, _ = self._decoded.popitem(last=False)
                self._decoded_bytes -= len(old)
        else:
            self._decoded.move_to_end(encoded)
        sample, checksum = decoded
        if sample.source_store_id != self.store_id or checksum != row["checksum"]:
            raise ReplayIntegrityError("Persisted sample identity/checksum mismatch")
        for key in ("trajectory_id", "group_id", "sample_index", "prompt_id", "reward", "weight_version",
                    "policy_lineage_id", "source_experiment_id", "source_run_id", "train_eligible"):
            if getattr(sample, key) != row[key]:
                raise ReplayIntegrityError("Indexed metadata does not match the immutable payload")
        return Entry(sample, row["checksum"], row["payload"], row["created_seq"])

    def append_trajectories(self, samples, request_id):
        identifier(request_id, "request_id")
        with self._lock:
            self._open()
            if self.read_only:
                raise PermissionError("Read-only replay store")
            entries, checksum = prepare_batch(samples, self.store_id, self.max_batch_size, self.max_batch_bytes)
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT receipt FROM replay_requests WHERE request_id=?", (request_id,)).fetchone()
                if row:
                    receipt = retry_receipt(receipt_load(row[0]), checksum, self.store_id, request_id)
                    self.conn.commit()
                    return receipt
                groups = tuple({e.sample.group_id for e in entries})
                ids = tuple(e.sample.trajectory_id for e in entries)
                sql = "SELECT * FROM replay_samples WHERE group_id IN (%s) OR trajectory_id IN (%s)" % (
                    ",".join("?" for _ in groups), ",".join("?" for _ in ids))
                existing = {row["trajectory_id"]: self._entry(row) for row in self.conn.execute(sql, groups + ids)}
                seq = self.conn.execute("SELECT COALESCE(MAX(created_seq),0)+1 FROM replay_samples").fetchone()[0]
                new, grouped, receipt = plan_write(entries, existing, self.store_id, request_id, checksum, seq)
                for group_id, (size, eligible, complete_seq) in grouped.items():
                    self.conn.execute("""INSERT INTO replay_groups VALUES(?,?,?,?) ON CONFLICT(group_id)
                        DO UPDATE SET complete_seq=COALESCE(replay_groups.complete_seq,excluded.complete_seq)""",
                        (group_id, size, int(eligible), complete_seq))
                self.conn.executemany("INSERT INTO replay_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                    (e.seq, e.sample.trajectory_id, e.sample.group_id, e.sample.sample_index, e.sample.prompt_id,
                     e.sample.reward, e.sample.weight_version, e.sample.policy_lineage_id, e.sample.source_experiment_id,
                     e.sample.source_run_id, int(e.sample.train_eligible), e.checksum, e.encoded) for e in new])
                self.conn.execute("INSERT INTO replay_requests VALUES(?,?)", (request_id, receipt_dump(receipt)))
                self.conn.commit()
                return receipt
            except BaseException:
                self.conn.rollback()
                raise

    def query_page(self, query=None, *, cursor=None, **filters):
        q = query_value(query, **filters)
        with self._lock:
            self._open()
            current = self.conn.execute("SELECT COALESCE(MAX(created_seq),0) FROM replay_samples").fetchone()[0]
            after, snapshot = page_bounds(q, cursor, self.store_id, current)
            where, args = ["s.created_seq>?", "s.created_seq<=?"], [after, snapshot]
            for key in ("trajectory_id", "prompt_id", "group_id", "policy_lineage_id", "source_experiment_id", "source_run_id", "weight_version"):
                value = getattr(q, key)
                if value is not None:
                    where.append(f"s.{key}=?")
                    args.append(value)
            for key, op in (("reward_min", ">="), ("reward_max", "<="), ("weight_version_min", ">="), ("weight_version_max", "<=")):
                value = getattr(q, key)
                if value is not None:
                    where.append(f"s.{key[:-4]}{op}?")
                    args.append(value)
            if q.train_eligible is not None:
                where.append("s.train_eligible=?")
                args.append(int(q.train_eligible))
                if q.train_eligible:
                    where.append("g.complete_seq IS NOT NULL AND g.complete_seq<=?")
                    args.append(snapshot)
            if q.source_store_id is not None and q.source_store_id != self.store_id:
                where.append("0")
            sql = "SELECT s.*,g.complete_seq FROM replay_samples s JOIN replay_groups g USING(group_id) WHERE " + " AND ".join(where) + " ORDER BY s.created_seq,s.trajectory_id LIMIT ?"
            rows = self.conn.execute(sql, args + [q.limit + 1]).fetchall()
            samples = [stored_sample(self._entry(row), row["complete_seq"] is not None
                       and row["complete_seq"] <= snapshot) for row in rows]
            return page_result(samples, q, snapshot, self.store_id)

    def query(self, query=None, **filters):
        return list(self.query_page(query, **filters).samples)

    def get_trajectory(self, trajectory_id, source_store_id=None):
        return self.get_trajectories([trajectory_id], source_store_id)[0]

    def get_trajectories(self, ids, source_store_id=None):
        ids = trajectory_ids(ids)
        if source_store_id is not None:
            identifier(source_store_id, "source_store_id")
        with self._lock:
            self._open()
            if source_store_id is not None and source_store_id != self.store_id:
                return [None] * len(ids)
            sql = ("SELECT s.*,g.complete_seq FROM replay_samples s JOIN replay_groups g USING(group_id) "
                   "WHERE s.trajectory_id IN (%s)" % ",".join("?" for _ in ids))
            result = {}
            for row in self.conn.execute(sql, ids):
                sample = stored_sample(self._entry(row), row["complete_seq"] is not None)
                result[sample.trajectory_id] = sample
            return [result.get(key) for key in ids]

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        self._decoded.clear()
        self._decoded_bytes = 0
