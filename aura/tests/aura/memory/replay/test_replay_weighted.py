"""Whole-group sampling/actual GRPO-priority tests; run on server."""
from dataclasses import replace
import math
import random
from collections import Counter

import pytest

from aura.memory.replay.codec import digest
from aura.memory.replay.mixing import GroupMixer, group_key, ReplayUnavailableError
from aura.memory.replay.sampling import draw_groups, group_priorities
from aura.memory.replay.schema import ReplayGroup
from test_replay_mixing import settings, group_batch


@pytest.mark.parametrize("strategy", ["uniform","priority","reward_weighted"])
def test_strategy_seed_complete_groups_and_global_rng(settings,make_group,strategy):
    config=settings(replay_strategy=strategy,batch_groups=2,replay_ratio=1)
    groups=group_batch(make_group,"a",6)
    priorities={group_key(g):float(i) for i,g in enumerate(groups)}
    before=random.getstate()
    a=draw_groups(groups,4,random.Random(4),config,priorities)
    b=draw_groups(groups,4,random.Random(4),config,priorities)
    assert a==b and len({group_key(g) for g in a[0]})==4
    assert all(g.complete and len(g.samples)==2 for g in a[0])
    assert all(0<p<=1 for p in a[1]) and random.getstate()==before


@pytest.mark.parametrize("strategy",["priority","reward_weighted"])
def test_exact_sequential_conditional_probability_not_marginal(settings,make_group,strategy):
    groups=tuple(ReplayGroup(tuple(make_group(group_id=str(i),reward=math.log(w))))
                 for i,w in enumerate((1,3,6)))
    config=settings(replay_strategy=strategy,priority_epsilon=1.0)
    priorities={group_key(g):w-1 for g,w in zip(groups,(1,3,6))}
    class Zero:
        def random(self): return 0.0
    chosen,probabilities=draw_groups(groups,2,Zero(),config,priorities)
    assert chosen==groups[:2]
    assert probabilities==pytest.approx((0.1,3/9))


@pytest.mark.parametrize("strategy",["priority","reward_weighted"])
def test_first_draw_frequency_matches_weights(settings,make_group,strategy):
    groups=tuple(ReplayGroup(tuple(make_group(group_id=str(i),reward=math.log(w))))
                 for i,w in enumerate((1,3,6)))
    config=settings(replay_strategy=strategy,priority_epsilon=1.0)
    priorities={group_key(g):w-1 for g,w in zip(groups,(1,3,6))}
    rng=random.Random(18);counts=Counter()
    for _ in range(6000):
        chosen,_=draw_groups(groups,1,rng,config,priorities)
        counts[group_key(chosen[0])]+=1
    for g,w in zip(groups,(1,3,6)):
        assert abs(counts[group_key(g)]-600*w)<130


@pytest.mark.parametrize("strategy",["priority","reward_weighted"])
def test_zero_advantages_or_equal_rewards_are_uniform(settings,make_group,strategy):
    groups=group_batch(make_group,"g",4)
    priorities={group_key(g):0.0 for g in groups}
    config=settings(replay_strategy=strategy)
    weighted=draw_groups(groups,3,random.Random(7),config,priorities)
    uniform=draw_groups(groups,3,random.Random(7),replace(config,replay_strategy="uniform"),priorities)
    assert weighted==uniform


def test_reward_negative_and_extreme_scale_is_stable(settings,make_group):
    groups=tuple(ReplayGroup(tuple(make_group(group_id=str(i),reward=w)))
                 for i,w in enumerate((-1000,0,1000)))
    config=settings(replay_strategy="reward_weighted",reward_temperature=0.001)
    chosen,prob=draw_groups(groups,3,random.Random(3),config,{})
    assert chosen==tuple(reversed(groups)) and prob==(1,1,1)


@pytest.mark.parametrize("field,value",[
    ("priority_alpha",-1),("priority_alpha",True),("priority_alpha",float("inf")),
    ("priority_epsilon",0),("priority_epsilon",float("nan")),
    ("reward_temperature",0),("reward_temperature",-1),("reward_temperature",True)])
def test_bad_sampling_config_rejected(settings,field,value):
    with pytest.raises(ValueError):settings(**{field:value})


def test_priority_atomic_success_refresh_and_eviction(settings,make_group):
    config=settings(replay_strategy="priority",replay_ratio=0.5,batch_groups=2,replay_buffer_size=4)
    mixer=GroupMixer(config);a=group_batch(make_group,"a",2)
    result=mixer.next(lambda:a,3)
    with pytest.raises(ValueError):mixer.commit_success(3)
    assert not mixer.window and mixer.completed_steps==0 and mixer.inflight is result
    updates={group_key(g):float(i+1) for i,g in enumerate(result.groups)}
    mixer.commit_success(3,updates)
    assert mixer.priorities==updates and set(mixer.priority_versions.values())=={3}
    b=group_batch(make_group,"b",2,version=4)
    result=mixer.next(lambda:b,4)
    assert len(result.replay_groups)==1 and len(result.online_groups)==1
    refresh={group_key(g):8.0 for g in result.groups}
    mixer.commit_success(4,refresh)
    assert set(mixer.priorities)==set(mixer.window)==set(mixer.priority_versions)
    assert all(mixer.priorities[k]==8.0 for k in refresh if k in mixer.window)
    assert tuple(mixer.pending)==b[1:]


@pytest.mark.parametrize("problem",["missing","extra","negative","nan"])
def test_priority_bad_update_does_not_publish(settings,make_group,problem):
    mixer=GroupMixer(settings(replay_strategy="priority",batch_groups=1))
    groups=group_batch(make_group,"a",1)
    mixer.next(lambda:groups,3)
    updates={group_key(groups[0]):1.0}
    if problem=="missing":updates={}
    elif problem=="extra":updates[("store","other")]=1
    else:updates[group_key(groups[0])]=-1 if problem=="negative" else float("nan")
    with pytest.raises(ValueError):mixer.commit_success(3,updates)
    assert not mixer.window and not mixer.priorities and mixer.completed_steps==0


def test_priority_missing_observation_is_not_silently_sampled(settings,make_group):
    mixer=GroupMixer(settings(replay_strategy="priority",batch_groups=1,replay_ratio=1))
    group=group_batch(make_group,"g",1)[0]
    mixer.window[group_key(group)]=group
    with pytest.raises(ReplayUnavailableError):mixer.next(lambda:(),3)


@pytest.mark.parametrize("norm",[False,True])
def test_priority_uses_actual_native_grpo_advantage(settings,make_group,norm):
    torch=pytest.importorskip("torch")
    np=pytest.importorskip("numpy")
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    groups=group_batch(make_group,"g",2)
    uids=[digest(list(group_key(g))) for g in groups for _ in g.samples]
    mask=torch.tensor([[1,0,1]]*4,dtype=torch.float32)
    rewards=torch.tensor([[0,0,v] for v in (0.,1.,0.,10.)])
    adv,_=compute_grpo_outcome_advantage(rewards,mask,np.array(uids),norm_adv_by_std_in_grpo=norm)
    original=adv.clone()
    result=group_priorities(groups,uids,adv.tolist(),mask.tolist())
    for n,g in enumerate(groups):
        expected=adv[n*2:n*2+2].abs()[mask[n*2:n*2+2].bool()].mean().item()
        assert result[group_key(g)]==pytest.approx(expected)
    assert torch.equal(adv,original)
    if not norm:assert result[group_key(groups[1])]==10*result[group_key(groups[0])]


@pytest.mark.parametrize("problem",["uid","nonfinite","partial"])
def test_priority_extraction_fails_closed(settings,make_group,problem):
    groups=group_batch(make_group,"g",1);uid=digest(list(group_key(groups[0])))
    uids=[uid,uid];adv=[[1.,900.,1.],[-1.,-900.,-1.]];mask=[[1,0,1],[1,0,1]]
    if problem=="uid":uids[0]="bad"
    elif problem=="nonfinite":adv[0][0]=float("nan")
    else:uids.pop()
    with pytest.raises(ValueError):group_priorities(groups,uids,adv,mask)


def test_priority_keeps_real_overlong_filtered_group_member(settings, make_group):
    groups = group_batch(make_group, "masked", 1)
    uid = digest(list(group_key(groups[0])))
    result = group_priorities(groups, [uid, uid], [[100., 100.], [2., 2.]], [[0, 0], [1, 1]])
    assert result[group_key(groups[0])] == 1.0


@pytest.mark.parametrize("strategy", ["uniform", "priority", "reward_weighted"])
def test_weighted_lag_uses_behavior_version_not_last_priority_refresh(settings, make_group, strategy):
    mixer = GroupMixer(settings(replay_strategy=strategy, replay_ratio=1, batch_groups=2,
                                warmup_online_steps=1, max_lag=8))
    groups = group_batch(make_group, "old", 2, version=3)
    first = mixer.next(lambda: groups, 3)
    def updates(plan):
        return {group_key(g): 1. for g in plan.groups} if strategy == "priority" else None
    mixer.commit_success(3, updates(first))
    edge = mixer.next(lambda: pytest.fail("Pure replay must not fetch online"), 10)
    assert len(edge.replay_groups) == 2  # lag=max_lag-1 is retained.
    mixer.commit_success(10, updates(edge))
    state = mixer.rng.getstate()
    with pytest.raises(ReplayUnavailableError):
        mixer.next(lambda: pytest.fail("Expired pool must not silently fall back"), 11)
    assert mixer.rng.getstate() == state and mixer.completed_steps == 2
    assert all(g.samples[0].weight_version == 3 for g in mixer.window.values())
    if strategy == "priority":
        assert set(mixer.priority_versions.values()) == {10}  # Not a generation version.


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_observed_priorities_fail_before_any_draw(settings, make_group, value):
    groups = group_batch(make_group, "bad", 2)
    rng = random.Random(3)
    state = rng.getstate()
    with pytest.raises(ValueError):
        draw_groups(groups, 1, rng, settings(replay_strategy="priority"),
                    {group_key(g): value for g in groups})
    assert rng.getstate() == state


def test_zero_priority_alpha_is_exact_uniform(settings, make_group):
    groups = group_batch(make_group, "g", 3)
    priorities = {group_key(g): float(i*100) for i, g in enumerate(groups)}
    cfg = settings(replay_strategy="priority", priority_alpha=0)
    assert draw_groups(groups, 2, random.Random(7), cfg, priorities) == draw_groups(
        groups, 2, random.Random(7), replace(cfg, replay_strategy="uniform"), priorities)
