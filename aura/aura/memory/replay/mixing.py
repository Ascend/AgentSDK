# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Complete-group uniform window and transactional in-process batch mixer."""
from collections import OrderedDict, deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import random
import math

from .sampling import draw_groups


class ReplayUnavailableError(RuntimeError):
    """No valid whole-group batch can be produced without changing the contract."""


def group_key(group):
    return group.samples[0].source_store_id, group.samples[0].group_id


def compatible(group, config, current_version):
    first = group.samples[0]
    if first.source_store_id != config.store_id and config.reuse_sources:
        from .reuse import effective_version
        from .schema import ReplayIntegrityError
        try:
            version = effective_version(first, config)
        except ReplayIntegrityError:
            return False
        return (group.complete and first.train_eligible
                and 0 <= current_version-version < config.max_lag)
    return (group.complete and first.train_eligible
            and first._compatibility_key == config._compatibility_key
            and 0 <= current_version - first.weight_version < config.max_lag)


@dataclass(frozen=True)
class MixResult:
    groups: tuple
    online_groups: tuple
    replay_groups: tuple
    next_carry: Decimal
    metrics: dict
    draw_probabilities: tuple = ()


class GroupMixer:
    """Only successfully updated groups enter the active window; disk history is untouched."""
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)  # Never created by the ratio=0 path.
        self.window = OrderedDict()
        self.pending = deque()
        self.seen_online = set()
        self.history_contains = None
        self.history_complete = True
        self.carry = Decimal(0)
        self.completed_steps = 0
        self.completed_groups = 0
        self.replayed_groups = 0
        self.inflight = None
        self.quarantined_online = 0
        self.priorities = {}
        self.priority_versions = {}

    def _validate(self, groups):
        keys = set()
        for group in groups:
            if not group.complete or not group.samples[0].train_eligible:
                raise ValueError("Only complete eligible groups may enter the mixer")
            key = group_key(group)
            if key in keys:
                raise ValueError("Duplicate generation group")
            keys.add(key)
        return keys

    def next(self, fetch_online, current_version):
        if self.inflight is not None:
            raise RuntimeError("Previous mixed batch has not been committed")
        if type(current_version) is not int or current_version < 0:
            raise ValueError("Invalid training weight version")
        config, m = self.config, self.config.batch_groups
        warmup = self.completed_steps < config.warmup_online_steps
        target = Decimal(m) * Decimal(str(config.replay_ratio)) + self.carry
        requested = 0 if warmup else int(target.to_integral_value(rounding=ROUND_FLOOR))
        next_carry = self.carry if warmup else target - requested
        # Quarantine stale online groups, retaining their durable archive (not consumed).
        self.pending = deque(g for g in self.pending if self._keep_pending(g, current_version))
        excluded = {group_key(g) for g in self.pending}
        candidates = [g for key, g in self.window.items()
                      if key not in excluded and compatible(g, config, current_version)
                      and (config.replay_strategy != "priority" or key in self.priorities)]
        if not warmup and config.replay_ratio == 1 and len(candidates) < m:
            raise ReplayUnavailableError("Pure replay requires enough successfully trained, compatible, non-stale groups")
        state = self.rng.getstate()
        try:
            chosen, probabilities = draw_groups(candidates, min(requested, len(candidates)),
                                                self.rng, config, self.priorities)
            online_needed = m - len(chosen)
            if len(self.pending) < online_needed:
                if not self.history_complete and self.history_contains is None:
                    raise RuntimeError('Restored compact mixer requires its durable online history ledger')
                groups = tuple(fetch_online())
                keys = self._validate(groups)
                repeated = bool(keys & self.seen_online)
                if self.history_contains is not None:
                    repeated |= self.history_contains(keys)
                if len(groups) != m or repeated:
                    raise ValueError("Online batch size mismatch or repeated online generation")
                if not all(compatible(g, config, current_version) for g in groups):
                    raise ReplayUnavailableError("Online generation has incompatible signatures or stale/future policy")
                self.pending.extend(groups)
                self.seen_online.update(keys)
            online = tuple(list(self.pending)[:online_needed])
            combined = online + chosen
            self._validate(combined)
            metrics = {"replay/requested_ratio": config.replay_ratio,
                       "replay/requested_groups": requested, "replay/actual_groups": len(chosen),
                       "replay/actual_ratio": len(chosen) / m, "replay/fallback_groups": requested-len(chosen),
                       "replay/online_groups": len(online)}
            self.inflight = MixResult(combined, online, chosen, next_carry, metrics, probabilities)
            return self.inflight
        except BaseException:
            self.rng.setstate(state)
            raise

    def wants_online_prefetch(self, *, last_step=False):
        """One reserved batch fits the 2*m pending budget without sampling RNG.

        Count even the current inflight online groups in the pending bound.
        Candidate shortage can increase demand; prefetch only when the ratio
        alone already proves that the next update needs another online batch.
        """
        config, result = self.config, self.inflight
        if (last_step or result is None or not config.online_prefetch
                or config.replay_ratio == 1 or config.max_lag < 2):
            return False
        m = config.batch_groups
        warmup = self.completed_steps + 1 < config.warmup_online_steps
        target = Decimal(m) * Decimal(str(config.replay_ratio)) + result.next_carry
        requested = 0 if warmup else int(target.to_integral_value(rounding=ROUND_FLOOR))
        remaining = len(self.pending) - len(result.online_groups)
        return len(self.pending) + m <= 2 * m and remaining < m - requested

    def _keep_pending(self, group, version):
        keep = compatible(group, self.config, version)
        if not keep:
            self.quarantined_online += 1
        return keep

    def validate_inflight(self, current_version):
        if self.inflight is None or not all(compatible(g, self.config, current_version) for g in self.inflight.groups):
            raise ReplayUnavailableError("Batch became invalid at the training-consumption boundary")

    def commit_success(self, current_version, priority_updates=None):
        self.validate_inflight(current_version)
        result = self.inflight
        updates = self.validate_priority_updates(priority_updates)
        for group in result.online_groups:
            if group_key(self.pending.popleft()) != group_key(group):
                raise RuntimeError("Pending FIFO changed during update")
            self.window[group_key(group)] = group
        while len(self.window) * self.config.group_size > self.config.replay_buffer_size:
            key, _ = self.window.popitem(last=False)
            self.priorities.pop(key, None)
            self.priority_versions.pop(key, None)
        for key, value in updates.items():
            if key in self.window:
                self.priorities[key] = value
                self.priority_versions[key] = current_version
        self.carry = result.next_carry
        self.completed_steps += 1
        self.completed_groups += len(result.groups)
        self.replayed_groups += len(result.replay_groups)
        self.inflight = None
        if self.history_contains is not None:
            self.seen_online = self.live_online_keys()
            self.history_complete = False
        return result.metrics

    def live_online_keys(self):
        return {group_key(g) for g in (*self.window.values(), *self.pending)
                if g.samples[0].source_store_id == self.config.store_id}

    def validate_priority_updates(self, updates):
        if self.config.replay_strategy != "priority":
            if updates:
                raise ValueError("Unexpected priority updates for another strategy")
            return {}
        expected = {group_key(g) for g in self.inflight.groups}
        if (not isinstance(updates, dict) or set(updates) != expected
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in updates.values())):
            raise ValueError("Priority requires actual finite advantages for every consumed complete group")
        return dict(updates)
