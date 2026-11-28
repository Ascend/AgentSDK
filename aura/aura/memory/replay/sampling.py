# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Whole-group sequential draws; probabilities are conditional, not inclusion IS."""
import math

from .codec import digest


def draw_groups(candidates, count, rng, config, priorities):
    if not count:
        return (), ()
    if count < 0 or count > len(candidates):
        raise ValueError("Invalid complete-group draw count")
    strategy = config.replay_strategy
    if strategy == "uniform":
        chosen = tuple(rng.sample(candidates, count))
        return chosen, tuple(1/(len(candidates)-i) for i in range(count))
    logits = []
    # Per-call only: equal observed priorities have identical logits. Preserve
    # every validation and arithmetic operation used to form a new logit, and
    # never retain values across priority/config updates or consume extra RNG.
    priority_logits = {}
    log_epsilon = math.log(config.priority_epsilon) if strategy == "priority" else None
    for group in candidates:
        if strategy == "priority":
            key = (group.samples[0].source_store_id, group.samples[0].group_id)
            value = priorities[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid observed GRPO priority")
            if value in priority_logits:
                logit = priority_logits[value]
            else:
                # log(value+epsilon) without overflow in the addition.
                a, b = math.log(value) if value else -math.inf, log_epsilon
                maximum = max(a, b)
                logit = config.priority_alpha * (maximum + math.log1p(math.exp(-abs(a-b))))
                priority_logits[value] = logit
        else:
            value = math.fsum(s.reward/len(group.samples) for s in group.samples)
            logit = value/config.reward_temperature
        if not math.isfinite(logit):
            raise ValueError("Sampling logits overflow; reduce alpha/reward scale or increase temperature")
        logits.append(logit)
    if len(set(logits)) == 1:
        chosen = tuple(rng.sample(candidates, count))
        return chosen, tuple(1/(len(candidates)-i) for i in range(count))
    # Keep parallel lists: pairing every candidate with a logit creates a
    # GC-tracked tuple per candidate on every draw. The order and float
    # operations below remain identical to the sequential weighted sampler.
    remaining = list(candidates)
    chosen, probabilities = [], []
    for _ in range(count):
        maximum = max(logits)
        weights = [math.exp(value-maximum) for value in logits]
        total = math.fsum(weights)
        cutoff, cumulative = rng.random()*total, 0.0
        # Same last-positive fallback as before, without scanning every weight
        # when the final entry is positive (the ordinary case).
        index = len(weights) - 1
        while weights[index] <= 0:
            index -= 1
        for i, weight in enumerate(weights):
            cumulative += weight
            if cutoff < cumulative:
                index = i
                break
        chosen.append(remaining.pop(index))
        logits.pop(index)
        probabilities.append(weights[index]/total)
    return tuple(chosen), tuple(probabilities)


def group_priorities(groups, uids, advantages, masks):
    """Extract, never recompute, native GRPO token advantages after estimation."""
    expected = {digest([g.samples[0].source_store_id, g.samples[0].group_id]): g for g in groups}
    if not (len(uids) == len(advantages) == len(masks) == sum(len(g.samples) for g in groups)):
        raise ValueError("Priority batch/group row count mismatch")
    values = {uid: [] for uid in expected}
    for uid, row, mask in zip(uids, advantages, masks):
        if uid not in values or len(row) != len(mask) or any(x not in (0, 1) for x in mask):
            raise ValueError("Priority uid/mask mismatch")
        active = [abs(float(value)) for value, keep in zip(row, mask) if keep]
        if not all(math.isfinite(x) for x in active):
            raise ValueError("Priority requires finite advantages on active responses")
        # overlong_filter masks a real trajectory's whole loss. Keep its group
        # slot, with zero contribution to the group's effective advantage.
        values[uid].append(math.fsum(x/len(active) for x in active) if active else 0.0)
    result = {}
    for uid, group in expected.items():
        members = values[uid]
        if len(members) != len(group.samples):
            raise ValueError("Priority batch lost a complete GRPO group")
        first = group.samples[0]
        result[(first.source_store_id, first.group_id)] = math.fsum(x/len(members) for x in members)
    return result
