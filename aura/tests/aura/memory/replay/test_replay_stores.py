from dataclasses import replace
import hashlib
import json
import os
import sqlite3

import pytest

from aura.memory.replay.codec import canonical, digest
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.schema import ReplayConflictError, ReplayIntegrityError, ReplayQuery
from aura.memory.replay.sqlite_store import SQLiteReplayStore


@pytest.fixture(params=[SQLiteReplayStore, JsonReplayStore], ids=["sqlite", "json"])
def store_factory(request, tmp_path):
    opened = []
    def factory(name="replay", **options):
        s = request.param(tmp_path / name, **options)
        opened.append(s)
        return s
    yield factory
    for store in reversed(opened):
        store.close()


def test_durable_roundtrip_retry_and_new_request_dedup(store_factory, make_group):
    s = store_factory(store_id="store")
    samples = make_group()
    receipt = s.append_trajectories(samples, "request")
    assert receipt.inserted_count == 2 and receipt.duplicate_count == 0
    retry = s.append_trajectories(samples, "request")
    assert retry.replayed and retry.inserted_count == 0 and retry.duplicate_count == 2
    assert retry.sample_refs == receipt.sample_refs
    duplicate = s.append_trajectories(samples, "new-request")
    assert duplicate.inserted_count == 0 and duplicate.sample_refs == receipt.sample_refs
    assert [v.created_seq for v in s.query()] == [1, 2]
    s.close()
    reopened = store_factory(store_id="store")
    assert reopened.append_trajectories(samples, "request").sample_refs == receipt.sample_refs
    assert reopened.get_trajectory(samples[0].trajectory_id).old_log_probs.data == samples[0].old_log_probs.data


def test_any_conflict_rolls_back_entire_batch(store_factory, make_group):
    s = store_factory(store_id="store")
    old = make_group()
    s.append_trajectories(old, "first")
    changed = [replace(t, reward=0.1) for t in old]
    with pytest.raises(ReplayConflictError):
        s.append_trajectories(changed, "first")
    with pytest.raises(ReplayConflictError):
        s.append_trajectories(make_group(group_id="new") + changed, "conflict")
    assert len(s.query()) == 2 and s.query()[0].reward == 0.9
    with pytest.raises(ReplayConflictError):
        s.append_trajectories(list(reversed(old)), "first")
    slot_conflict = [replace(t, trajectory_id=t.trajectory_id + "other") for t in old]
    with pytest.raises(ReplayConflictError):
        s.append_trajectories(slot_conflict, "slots")
    assert len(s.query()) == 2


def test_group_failure_is_closed_and_archival_explicit(store_factory, make_group):
    s = store_factory(store_id="store")
    with pytest.raises(ValueError):
        s.append_trajectories(make_group()[:1], "partial")
    assert s.query() == []
    archived = make_group(eligible=False)
    s.append_trajectories(archived[:1], "archive-1")
    assert not s.query()[0].group_complete and s.query(train_eligible=True) == []
    s.append_trajectories(archived[1:], "archive-2")
    assert all(v.group_complete for v in s.query())
    assert len(s.query(train_eligible=False)) == 2 and s.query(train_eligible=True) == []


def test_intersection_queries_and_inclusive_bounds(store_factory, make_group):
    s = store_factory(store_id="store")
    s.append_trajectories(make_group(group_id="a", reward=0.8, version=3) + make_group(group_id="b", reward=0.1, version=4), "mixed")
    assert len(s.query(reward_min=0.8, limit=100)) == 2
    assert len(s.query(reward_min=0.8, reward_max=0.8, weight_version_min=3, weight_version_max=3, train_eligible=True)) == 2
    assert s.query(prompt_id="prompt-a", weight_version=4) == []
    assert s.query(prompt_id="' OR 1=1 --") == []
    assert s.get_trajectory("a/trajectory/0", source_store_id="other") is None
    assert s.get_trajectory("unknown") is None
    assert s.query(limit=0) == []


def test_snapshot_pagination_ignores_new_appends_and_binds_query(store_factory, make_group):
    s = store_factory(store_id="store")
    s.append_trajectories(sum([make_group(group_id=k) for k in ("a", "b", "c")], []), "first")
    q = ReplayQuery(limit=2)
    first = s.query_page(q)
    assert first.snapshot_seq == 6 and first.next_cursor
    s.append_trajectories(make_group(group_id="d"), "new")
    result, page = list(first.samples), first
    while page.next_cursor:
        page = s.query_page(q, cursor=page.next_cursor)
        result.extend(page.samples)
    assert [v.created_seq for v in result] == list(range(1, 7))
    assert len(s.query()) == 8
    with pytest.raises(ValueError):
        s.query_page(ReplayQuery(limit=3), cursor=first.next_cursor)
    other = store_factory("other", store_id="another-store")
    with pytest.raises(ValueError):
        other.query_page(q, cursor=first.next_cursor)
    with pytest.raises(ValueError):
        s.query_page(q, cursor="not-a-cursor")


def test_read_only_refuses_writes_and_never_migrates(store_factory, make_group, tmp_path):
    s = store_factory(store_id="store")
    s.append_trajectories(make_group(), "first")
    s.close()
    path = tmp_path / "replay"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ro = store_factory(read_only=True)
    assert len(ro.query()) == 2 and ro.get_replay_manifest()["store_id"] == "store"
    with pytest.raises(PermissionError):
        ro.append_trajectories(make_group(group_id="b"), "write")
    if isinstance(ro, SQLiteReplayStore):
        with pytest.raises(sqlite3.OperationalError):
            ro.conn.execute("DELETE FROM replay_samples")
    ro.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        store_factory("missing", read_only=True)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("filters", [dict(limit=True), dict(limit=-1), dict(limit=1001), dict(reward_min=True),
    dict(reward_min=float("nan")), dict(reward_min=2, reward_max=1), dict(weight_version=True)])
def test_invalid_queries_fail(store_factory, filters):
    s = store_factory(store_id="store")
    with pytest.raises((ValueError, TypeError)):
        s.query(**filters)


def test_manifest_namespace_and_schema_are_checked(store_factory, make_group):
    s = store_factory(store_id="store")
    with pytest.raises(ReplayConflictError):
        s.append_trajectories(make_group(store_id="foreign"), "wrong-source")
    s.close()
    with pytest.raises(ReplayIntegrityError):
        store_factory(store_id="different")


def test_batch_bounds_duplicate_ids_and_closed_store(store_factory, make_group):
    s = store_factory(store_id="store", max_batch_size=2)
    samples = make_group()
    for invalid in ([], samples + make_group(group_id="extra"), [samples[0], samples[0]]):
        with pytest.raises(ValueError):
            s.append_trajectories(invalid, "failed")
        assert s.query() == []
    s.append_trajectories(samples, "failed")
    s.close()
    with pytest.raises(RuntimeError):
        s.query()
    small = store_factory("small", store_id="store", max_batch_bytes=1)
    with pytest.raises(ValueError, match="byte limit"):
        small.append_trajectories(samples, "too-big")
    assert small.query() == []


def test_snapshot_group_completeness_does_not_change_mid_pagination(store_factory, make_group):
    s = store_factory(store_id="store")
    samples = make_group(eligible=False)
    s.append_trajectories(make_group(group_id="first") + samples[:1], "first")
    page = s.query_page(limit=1)
    s.append_trajectories(samples[1:], "finish")
    seen = list(page.samples)
    while page.next_cursor:
        page = s.query_page(limit=1, cursor=page.next_cursor)
        seen.extend(page.samples)
    assert len(seen) == 3 and seen[-1].group_complete is False
    assert all(v.group_complete for v in s.query())


def test_sqlite_two_connections_have_shared_idempotency(tmp_path, make_group):
    path = tmp_path / "shared.db"
    first = SQLiteReplayStore(path, store_id="store")
    second = SQLiteReplayStore(path, store_id="store")
    try:
        receipt = first.append_trajectories(make_group(), "same")
        retry = second.append_trajectories(make_group(), "same")
        assert retry.replayed and retry.sample_refs == receipt.sample_refs
        assert len(second.query()) == 2
    finally:
        second.close()
        first.close()


def test_sqlite_readonly_does_not_migrate_legacy_database(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE legacy(value TEXT)")
        conn.execute("INSERT INTO legacy VALUES('preserved')")
    before = path.read_bytes()
    with pytest.raises(ReplayIntegrityError, match="migration is forbidden"):
        SQLiteReplayStore(path, read_only=True)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM legacy").fetchone()[0] == "preserved"
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'replay_%'").fetchone()[0] == 0


def test_sqlite_mid_transaction_failure_rolls_back(tmp_path, make_group):
    s = SQLiteReplayStore(tmp_path / "atomic.db", store_id="store")
    try:
        s.conn.execute("CREATE TRIGGER fail BEFORE INSERT ON replay_samples WHEN NEW.sample_index=1 BEGIN SELECT RAISE(ABORT,'injected'); END")
        with pytest.raises(sqlite3.IntegrityError):
            s.append_trajectories(make_group(), "failed")
        assert s.query() == [] and s.conn.execute("SELECT COUNT(*) FROM replay_groups").fetchone()[0] == 0
        s.conn.execute("DROP TRIGGER fail")
        receipt = s.append_trajectories(make_group(), "failed")
        assert [r[2] for r in receipt.sample_refs] == [1, 2]
        s.conn.execute("UPDATE replay_samples SET reward=123 WHERE created_seq=1")
        with pytest.raises(ReplayIntegrityError):
            s.query()
    finally:
        s.close()


def test_json_torn_tail_preserved_and_append_blocked(tmp_path, make_group):
    p = tmp_path / "tail.jsonl"
    s = JsonReplayStore(p, store_id="store")
    s.append_trajectories(make_group(), "first")
    s.close()
    with p.open("ab") as f:
        f.write(b'{"torn"')
    original = p.read_bytes()
    with pytest.warns(RuntimeWarning):
        ro = JsonReplayStore(p, read_only=True)
    assert len(ro.query()) == 2 and ro.ignored_tail_bytes == 7
    ro.close()
    assert p.read_bytes() == original
    with pytest.warns(RuntimeWarning):
        writer = JsonReplayStore(p)
    try:
        with pytest.raises(ReplayIntegrityError):
            writer.append_trajectories(make_group(group_id="b"), "new")
    finally:
        writer.close()
    assert p.read_bytes() == original


def test_json_durability_failure_never_returns_receipt_or_changes_memory(tmp_path, make_group, monkeypatch):
    p = tmp_path / "fsync.jsonl"
    s = JsonReplayStore(p, store_id="store")
    def fail(_):
        raise OSError("injected fsync failure")
    with monkeypatch.context() as m:
        m.setattr(os, "fsync", fail)
        with pytest.raises(OSError):
            s.append_trajectories(make_group(), "request")
    assert s.query() == []
    with pytest.raises(ReplayIntegrityError):
        s.append_trajectories(make_group(), "request")
    s.close()
    # A lost acknowledgement may have left a complete commit; reopening reconciles it.
    reopened = JsonReplayStore(p)
    try:
        retry = reopened.append_trajectories(make_group(), "request")
        assert retry.inserted_count == 0 and retry.replayed
    finally:
        reopened.close()


def test_json_single_writer_and_committed_corruption(tmp_path):
    p = tmp_path / "one-writer.jsonl"
    s = JsonReplayStore(p, store_id="store")
    try:
        with pytest.raises(RuntimeError):
            JsonReplayStore(p)
    finally:
        s.close()
    with p.open("ab") as f:
        f.write(b'{"record":{},"checksum":"bad"}\n')
    with pytest.raises(ReplayIntegrityError):
        JsonReplayStore(p, read_only=True)
