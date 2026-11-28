from dataclasses import replace
from decimal import Decimal
from unittest.mock import Mock

import pytest

from aura.memory.replay.config import ReplayConfig, replay_ratio
from aura.memory.replay.launcher import prepare_replay
from aura.memory.replay.mixing import GroupMixer, ReplayUnavailableError, group_key
from aura.memory.replay.schema import ReplayGroup


@pytest.fixture
def settings(tmp_path):
    def factory(**changes):
        values = dict(replay_ratio=0.3, replay_buffer_size=100, group_size=2, batch_groups=10,
                      max_lag=8, seed=42, store_id="store", source_experiment_id="experiment-a",
                      source_run_id="run-a", policy_lineage_id="lineage-a", model_signature="model-a",
                      tokenizer_signature="tokenizer-template-a", task_signature="gsm8k/train",
                      reward_signature="math-v1", path=str(tmp_path / "replay.db"))
        values.update(changes)
        return ReplayConfig(**values)
    return factory


def group_batch(make_group, prefix, count=10, version=3):
    return tuple(ReplayGroup(tuple(make_group(group_id=f"{prefix}-{i}", version=version))) for i in range(count))


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.3", float("nan"), float("inf")])
def test_invalid_ratio(value):
    with pytest.raises(ValueError):
        replay_ratio({"replay": {"replay_ratio": value}})


@pytest.mark.parametrize("changes", [{"replay_buffer_size": 1}, {"group_size": True}, {"max_lag": 0},
    {"seed": True}, {"replay_strategy": "unknown"}, {"rpc_timeout": float("nan")},
    {"warmup_online_steps": -1}, {"path": "relative.db"}])
def test_invalid_settings(settings, changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_disabled_ignores_unneeded_storage_settings_and_has_no_side_effects(monkeypatch):
    import random
    import sys
    actor = Mock()
    monkeypatch.setitem(sys.modules, "ray", actor)
    monkeypatch.setattr(random, "Random", Mock(side_effect=AssertionError("Must not create RNG")))
    online = {"replay": {"replay_ratio": 0, "path": "/invalid/unused", "replay_strategy": "priority"}}
    assert prepare_replay(online, object()) is None
    assert not actor.mock_calls
    assert replay_ratio({}) == 0
    with pytest.raises(ValueError):
        replay_ratio({"replay": []})


def test_exact_ratio_cold_fallback_then_warm_complete_groups(settings, make_group):
    mixer = GroupMixer(settings())
    source = Mock(side_effect=[group_batch(make_group, "a"), group_batch(make_group, "b"), group_batch(make_group, "c")])
    cold = mixer.next(source, 3)
    assert len(cold.online_groups) == 10 and cold.metrics["replay/fallback_groups"] == 3
    assert not mixer.window  # Persistence is not a training acknowledgement.
    with pytest.raises(RuntimeError):
        mixer.next(source, 3)
    mixer.commit_success(3)
    mixed = mixer.next(source, 4)
    assert len(mixed.replay_groups) == 3 and len(mixed.online_groups) == 7
    assert mixed.metrics["replay/actual_ratio"] == 0.3
    assert all(g.complete for g in mixed.groups)
    assert len({group_key(g) for g in mixed.groups}) == 10
    mixer.commit_success(4)
    assert len(mixer.pending) == 3 and source.call_count == 2
    next_mix = mixer.next(source, 4)
    assert len(next_mix.online_groups) == 7  # fetches only when pending is insufficient


def test_pending_fifo_and_no_fetch_for_replay_steps(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2, replay_ratio=0.5))
    a, b = group_batch(make_group, "a", 2), group_batch(make_group, "b", 2)
    source = Mock(side_effect=[a, b])
    mixer.next(source, 3); mixer.commit_success(3)
    second = mixer.next(source, 3); mixer.commit_success(3)
    assert second.online_groups == b[:1] and tuple(mixer.pending) == b[1:]
    third = mixer.next(source, 3)
    assert third.online_groups == b[1:] and source.call_count == 2
    mixer.commit_success(3)
    assert len(mixer.pending) == 0


def test_fractional_group_carry_is_exact_and_committed_only_after_success(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2))
    calls = 0
    def source():
        nonlocal calls
        calls += 1
        return group_batch(make_group, f"batch{calls}", 2)
    requested = []
    for _ in range(10):
        plan = mixer.next(source, 3)
        requested.append(plan.metrics["replay/requested_groups"])
        mixer.commit_success(3)
    assert sum(requested) == 6
    assert mixer.carry == Decimal(0)


def test_pure_replay_cold_fail_then_explicit_warmup_and_no_online_fallback(settings, make_group):
    cold = GroupMixer(settings(replay_ratio=1, batch_groups=2))
    source = Mock(side_effect=AssertionError("Pure replay must not fetch"))
    with pytest.raises(ReplayUnavailableError):
        cold.next(source, 3)
    source.assert_not_called()
    warm = GroupMixer(settings(replay_ratio=1, batch_groups=2, warmup_online_steps=1))
    warm.next(lambda: group_batch(make_group, "warm", 2), 3)
    warm.commit_success(3)
    pure = warm.next(source, 4)
    assert len(pure.replay_groups) == 2 and not pure.online_groups
    warm.commit_success(4)
    with pytest.raises(ReplayUnavailableError):
        warm.next(source, 11)  # lag==max_lag must be rejected.
    source.assert_not_called()


@pytest.mark.parametrize("include_expired", [False, True])
def test_pure_replay_one_valid_candidate_does_not_fetch_or_mutate(settings, make_group, include_expired):
    mixer = GroupMixer(settings(replay_ratio=1, batch_groups=2,
                                replay_buffer_size=4, warmup_online_steps=1))
    # Synthetic successfully trained window; explicitly not a live-service run.
    groups = group_batch(make_group, "valid", 1, version=4)
    if include_expired:
        groups += group_batch(make_group, "expired", 1, version=3)
    mixer.window.update((group_key(g), g) for g in groups)
    mixer.completed_steps = 1
    before = (list(mixer.window.items()), mixer.rng.getstate(), mixer.carry)
    source = Mock(side_effect=AssertionError("Pure replay must never fill online"))
    with pytest.raises(ReplayUnavailableError):
        mixer.next(source, 11)  # Only version4 is valid; version3 has lag==8.
    source.assert_not_called()
    assert (list(mixer.window.items()), mixer.rng.getstate(), mixer.carry) == before
    assert mixer.completed_steps == 1 and mixer.inflight is None and not mixer.pending


def test_candidate_shortage_uses_actual_ratio_and_online_fallback(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=10, replay_ratio=0.3))
    small = group_batch(make_group, "small", 2)
    mixer.window.update((group_key(g), g) for g in small)  # Known successful synthetic setup.
    plan = mixer.next(lambda: group_batch(make_group, "new"), 4)
    assert plan.metrics["replay/actual_groups"] == 2
    assert plan.metrics["replay/fallback_groups"] == 1
    assert plan.metrics["replay/actual_ratio"] == 0.2
    assert len(plan.online_groups) == 8


@pytest.mark.parametrize("change", [{"weight_version": 4}, {"policy_lineage_id": "other"},
    {"model_signature": "other"}, {"tokenizer_signature": "other"}, {"source_run_id": "other"}])
def test_incompatible_future_or_cross_run_groups_never_train(settings, make_group, change):
    mixer = GroupMixer(settings(batch_groups=1))
    group = ReplayGroup(tuple(replace(s, **change) for s in make_group()))
    with pytest.raises(ReplayUnavailableError):
        mixer.next(lambda: (group,), 3)
    assert not mixer.pending and not mixer.window and mixer.carry == 0


def test_staleness_checked_again_at_actor_update_boundary(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=1, max_lag=2))
    mixer.next(lambda: group_batch(make_group, "a", 1), 4)
    with pytest.raises(ReplayUnavailableError):
        mixer.commit_success(5)
    assert not mixer.window and mixer.completed_steps == 0


def test_window_capacity_never_splits_group_or_evicts_pending(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2, replay_ratio=0.5, replay_buffer_size=3))
    a, b = group_batch(make_group, "a", 2), group_batch(make_group, "b", 2)
    mixer.next(lambda: a, 3); mixer.commit_success(3)
    assert len(mixer.window) == 1
    mixer.next(lambda: b, 3); mixer.commit_success(3)
    assert len(mixer.window) == 1 and tuple(mixer.pending) == b[1:]


def test_stale_pending_is_quarantined_not_trained(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2, replay_ratio=0.5, max_lag=2))
    mixer.next(lambda: group_batch(make_group, "a", 2), 3); mixer.commit_success(3)
    mixer.next(lambda: group_batch(make_group, "b", 2), 4); mixer.commit_success(4)
    plan = mixer.next(lambda: group_batch(make_group, "fresh", 2, version=5), 5)
    assert mixer.quarantined_online == 1
    assert all(g.samples[0].weight_version == 5 for g in plan.online_groups)


def test_bad_or_duplicate_online_batch_is_rejected(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2))
    group = group_batch(make_group, "a", 1)[0]
    with pytest.raises(ValueError):
        mixer.next(lambda: (group, group), 3)
    assert not mixer.pending and mixer.carry == 0
    valid = group_batch(make_group, "b", 2)
    mixer.next(lambda: valid, 3); mixer.commit_success(3)
    with pytest.raises(ValueError):
        mixer.next(lambda: valid, 3)


def test_uniform_is_seeded_without_replacement_and_preserves_global_rng(settings, make_group):
    import random
    from collections import Counter
    groups = group_batch(make_group, "candidate", 5)
    first = GroupMixer(settings(replay_ratio=1, batch_groups=2))
    second = GroupMixer(settings(replay_ratio=1, batch_groups=2))
    for mixer in (first, second):
        mixer.window.update((group_key(g), g) for g in groups)
    state = random.getstate()
    counts = Counter()
    for _ in range(600):
        a = first.next(lambda: (), 3)
        b = second.next(lambda: (), 3)
        assert a.groups == b.groups and len({group_key(g) for g in a.groups}) == 2
        counts.update(group_key(g) for g in a.groups)
        first.commit_success(3); second.commit_success(3)
    assert random.getstate() == state
    assert all(180 < count < 300 for count in counts.values())
