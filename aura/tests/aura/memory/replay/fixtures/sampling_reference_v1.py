# Frozen compatibility oracle from 24fd0d25; never used by production.
import math

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
    for group in candidates:
        if strategy == "priority":
            key = (group.samples[0].source_store_id, group.samples[0].group_id)
            value = priorities[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid observed GRPO priority")
            # log(value+epsilon) without overflow in the addition.
            a, b = math.log(value) if value else -math.inf, math.log(config.priority_epsilon)
            maximum = max(a, b)
            logit = config.priority_alpha * (maximum + math.log1p(math.exp(-abs(a-b))))
        else:
            value = math.fsum(s.reward/len(group.samples) for s in group.samples)
            logit = value/config.reward_temperature
        if not math.isfinite(logit):
            raise ValueError("Sampling logits overflow; reduce alpha/reward scale or increase temperature")
        logits.append(logit)
    if len(set(logits)) == 1:
        chosen = tuple(rng.sample(candidates, count))
        return chosen, tuple(1/(len(candidates)-i) for i in range(count))
    remaining = list(zip(candidates, logits))
    chosen, probabilities = [], []
    for _ in range(count):
        maximum = max(value for _, value in remaining)
        weights = [math.exp(value-maximum) for _, value in remaining]
        total = math.fsum(weights)
        cutoff, cumulative = rng.random()*total, 0.0
        index = max(i for i, value in enumerate(weights) if value > 0)
        for i, weight in enumerate(weights):
            cumulative += weight
            if cutoff < cumulative:
                index = i
                break
        chosen.append(remaining.pop(index)[0])
        probabilities.append(weights[index]/total)
    return tuple(chosen), tuple(probabilities)
