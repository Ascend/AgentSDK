"""Opt-in real server Ray runtime, no NPU or long training."""
from dataclasses import asdict, replace
from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from test_replay_actor_integration import real_ray
from test_replay_reuse import reuse_case, image, record
from test_replay_mixing import settings

pytestmark = pytest.mark.skipif(os.environ.get('RUN_REPLAY_RAY_TESTS') != '1', reason='Server real Ray opt-in')


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
@pytest.mark.parametrize('ratio', [.3, 1.0])
@pytest.mark.parametrize('strategy', ['uniform', 'priority', 'reward_weighted'])
def test_real_actor_foreign_provenance_native_grpo_is_and_restart(real_ray, reuse_case, make_group,
                                                                backend, ratio, strategy):
    import numpy as np
    import torch
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.runtime import ReplaySession
    from aura.memory.replay.schema import ReplayTensor
    from aura.memory.replay.config import ReplayConfig
    from aura.trainer.train_adapter.verl.full_async.replay_loss import apply_behavior_correction
    from omegaconf import OmegaConf

    ray = real_ray
    root, config, _ = reuse_case(backend, strategy, ratio)
    before = image(root/'source')
    actor = Episode.remote('reuse', replay_store_config={'backend': backend, 'path': config.path, 'store_id': 'store'})
    calls = []
    def online():
        index = len(calls); calls.append(index)
        refs = []
        for g in range(2):
            # Deliberately collide the first batch's group/trajectory strings
            # with the foreign store. Store-scoped identities must remain distinct.
            gid = f'g{g}' if index == 0 else f'online-{index}-{g}'
            samples = [replace(s, source_experiment_id='experiment-b', source_run_id='run-b',
                policy_lineage_id='lineage-b', reward=.1+.5*i,
                payload=dict(s.payload, rm_scores=ReplayTensor.from_values([0., 0., .1+.5*i], 'float32'),
                             token_level_rewards=ReplayTensor.from_values([0., 0., .1+.5*i], 'float32')),
                old_log_probs=ReplayTensor.from_values([-3., 0., -4.], 'float32'))
                for i, s in enumerate(make_group(group_id=gid, version=0))]
            receipt = ray.get(actor.append_replay_batch.remote(samples, f'online-{index}-{g}'))
            refs.extend(dict(trajectory_id=t, checksum=c, created_seq=n, source_store_id='store')
                        for t, c, n in receipt.sample_refs)
        return {'replay_refs': refs}
    try:
        # Exercise the actual OmegaConf -> dataclass conversion used by launcher.
        config = ReplayConfig.from_mapping(OmegaConf.create(asdict(config)))
        session = ReplaySession(config, 0, actor)
        foreign_rows = 0
        observed = []
        for version in range(4):
            raw = session.next_batch(online, version)
            assert sorted(Counter(raw['prompt_ids']).values()) == [2, 2]
            keys = list(zip(raw['replay_source_store_ids'], raw['replay_trajectory_ids']))
            assert len(set(keys)) == 4
            assert all(v == 0 for v in raw['replay_weight_versions'])
            for source, original in zip(raw['replay_source_store_ids'], raw['replay_source_weight_versions']):
                assert original == (11 if source == 'foreign' else 0)
                foreign_rows += int(source == 'foreign')
            behavior = raw['rollout_log_probs'].clone()
            tensors = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            proto = DataProto.from_dict(tensors=tensors, non_tensors={'uid': np.array(raw['prompt_ids'])})
            adv, _ = compute_grpo_outcome_advantage(proto.batch['token_level_rewards'], proto.batch['response_mask'],
                proto.non_tensor_batch['uid'], norm_adv_by_std_in_grpo=False)
            assert adv.abs().sum().item() > 0
            proto.batch['advantages'] = adv
            # Real old-policy/behavior correction, including both clip endpoints.
            delta = torch.tensor([[-2., 0., 2.]]*4, device='cpu')
            proto.batch['old_log_probs'] = behavior + delta
            metrics = apply_behavior_correction(proto)
            assert metrics['replay/is_weight_min'] == .5 and metrics['replay/is_weight_max'] == 2.
            assert torch.equal(proto.batch['rollout_log_probs'], behavior)
            from test_replay_loss_native import engine_data, loss_config
            from verl.workers.utils.losses import ppo_loss
            from verl.workers.utils.padding import no_padding_2_padding
            td = engine_data(proto)
            current = torch.full_like(td['input_ids'].values(), -4., dtype=torch.float32, requires_grad=True)
            loss, _ = ppo_loss(loss_config(), {'log_probs': current}, td)
            response = no_padding_2_padding(current, td)
            ratio_tensor = torch.exp(response-td['old_log_probs'])
            main = torch.maximum(-td['advantages']*ratio_tensor,
                                 -td['advantages']*torch.clamp(ratio_tensor, .8, 1.2))
            dual = torch.minimum(main, -3.*td['advantages'])
            manual = (torch.where(td['advantages'] < 0, dual, main)*td['rollout_is_weights']
                      *td['response_mask']).sum()/int(proto.batch['response_mask'].sum())
            gradient, = torch.autograd.grad(manual, current, retain_graph=True)
            loss.backward()
            torch.testing.assert_close(loss, manual, rtol=0, atol=1e-7)
            torch.testing.assert_close(current.grad, gradient, rtol=0, atol=1e-7)
            assert torch.isfinite(current.grad).all() and current.grad.abs().sum().item() > 0
            observed.append(dict(version=version, source_ids=raw['replay_source_store_ids'],
                original_versions=raw['replay_source_weight_versions'], mapped_versions=raw['replay_weight_versions'],
                replay_groups=len(session.mixer.inflight.replay_groups),
                probabilities=session.mixer.inflight.draw_probabilities,
                loss=float(loss.detach()), grad_l1=float(current.grad.abs().sum()), is_metrics=metrics))
            session.commit_success(version, session.priority_updates(proto))
        assert foreign_rows > 0 and (not calls if ratio == 1 else calls)
        import asyncio
        import contextlib
        from types import SimpleNamespace
        from unittest.mock import Mock
        from test_replay_backpressure import method
        trainer = SimpleNamespace(timing_raw={}, _get_samples_from_queue=lambda: (0, raw),
            _replay_session=session, config=SimpleNamespace(algorithm=SimpleNamespace(adv_estimator='grpo'),
                data=SimpleNamespace(train_batch_size=2),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2, temperature=1.))))
        trainer._extract_generate_tensors = method('aura/trainer/train_adapter/verl/full_async/full_async_trainer.py',
                                                  'FullyAsyncTrainer', '_extract_generate_tensors')
        generate = method('aura/trainer/train_adapter/verl/full_async/full_async_trainer.py',
            'FullyAsyncTrainer', '_fit_generate', DataProto=DataProto, torch=torch, np=np,
            logger=Mock(), marked_timer=lambda *a, **k: contextlib.nullcontext())
        converted = asyncio.run(generate(trainer))
        assert converted.non_tensor_batch['replay_source_weight_versions'].tolist() == raw['replay_source_weight_versions']
        assert converted.meta_info['trajectory_param_versions'] == raw['replay_weight_versions']
        assert ray.get(actor.verify_replay_sources.remote()) == {'sources': 1, 'unchanged': True}
        # Reopening a fresh actor remounts and revalidates the same frozen bytes.
        ray.get(actor.close_replay_store.remote()); ray.kill(actor, no_restart=True)
        actor = Episode.remote('reuse-reopened', replay_store_config={'backend': backend, 'path': config.path, 'store_id': 'store'})
        ray.get(actor.configure_replay_sources.remote(asdict(config)))
        sample = ray.get(actor.get_trajectory.remote('g0/trajectory/0', 'foreign'))
        assert sample.source_experiment_id == 'experiment-a' and sample.weight_version == 11
        assert len(ray.get(actor.get_reuse_group.remote('foreign', 'g0'))) == 2
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.get_reuse_group.remote('foreign', 'unauthorized'))
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.configure_replay_sources.remote(asdict(replace(config, max_lag=20))))
        assert image(root/'source') == before
        record(f'actor-{backend}-{strategy}-{ratio}', dict(observations=observed,
            source_before=before, source_after=image(root/'source'), restarts=1, foreign_rows=foreign_rows))
    finally:
        ray.get(actor.close_replay_store.remote()); ray.kill(actor, no_restart=True)


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
def test_real_transactional_session_foreign_genesis_ack_checkpoint_recover(real_ray, reuse_case, backend):
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.transaction import TransactionalReplaySession
    from aura.memory.replay.codec import canonical, digest
    from aura.memory.replay.state import snapshot_mixer
    ray = real_ray
    root, config, _ = reuse_case(backend)
    before = image(root/'source')
    actor = Episode.remote('reuse-transaction', replay_store_config={'backend': backend, 'path': config.path, 'store_id': 'store'})
    def save(step, version):
        path = root/'destination'/f'model-{step}'; path.write_bytes(canonical({'step': step, 'version': version}))
        return {'step': step, 'policy_version': version,
                'artifacts': {r: ['destination/'+path.name] for r in ('model', 'optimizer', 'scheduler', 'rng')}}
    source = {'format': 'aura.replay.source-state.v1', 'quiescent': True, 'prefetch': None,
              'producer': {'identity': digest('foreign-fixed-source'), 'state': {}}, 'queued_batches': []}
    try:
        session = TransactionalReplaySession.start(config, 0, actor, str(root/'destination'/'ledger'),
            branch='first', writer_request='first')
        session.checkpoint(root, source, save, commit_id='genesis')
        session.next_batch(lambda: pytest.fail('Foreign pure replay'), 0); session.ack_update(1)
        session.checkpoint(root, source, save, commit_id='cut1')
        from aura.trainer.train_adapter.verl.full_async.replay_recovery import select_pair
        selected = select_pair({'root': str(root), 'ledger_path': str(root/'destination'/'ledger'),
                                'from_branch': 'first'}, config, actor)
        assert selected['body']['commit_id'] == 'cut1'
        expected = snapshot_mixer(session.mixer, policy_version=1)
        ray.get(actor.close_replay_store.remote()); ray.kill(actor, no_restart=True)
        actor = Episode.remote('reuse-recovery', replay_store_config={'backend': backend, 'path': config.path, 'store_id': 'store'})
        recovered, chosen = TransactionalReplaySession.recover(config, 0, actor,
            str(root/'destination'/'ledger'), root, from_branch='first', new_branch='second', writer_request='second',
            load_model=lambda paths, step, version: {'step': step, 'policy_version': version},
            restore_source=lambda value: {'source_sha256': hashlib.sha256(canonical(value)).hexdigest()})
        assert chosen['body']['policy_version'] == 1
        assert snapshot_mixer(recovered.mixer, policy_version=1) == expected
        recovered.next_batch(lambda: pytest.fail('No online'), 1); recovered.ack_update(2)
        assert image(root/'source') == before
        record('transaction-'+backend, dict(selected_step=chosen['body']['step'], final_version=2,
            source_before=before, source_after=image(root/'source'), paired_snapshot_exact=True))
    finally:
        ray.get(actor.close_replay_store.remote()); ray.kill(actor, no_restart=True)


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
def test_launcher_supports_attested_pure_reuse_and_rejects_incompatible_before_writer(real_ray, reuse_case, backend):
    from omegaconf import OmegaConf
    from aura.memory.replay.launcher import prepare_replay
    root, config, _ = reuse_case(backend)
    def configs(options):
        train = OmegaConf.create({'replay': options, 'trainer': {'resume_mode': 'disable'},
            'algorithm': {'adv_estimator': 'grpo'}, 'data': {'train_batch_size': 2}, 'rollout': {'n': 2},
            'actor_rollout_ref': {'rollout': {'n': 2}, 'actor': {'use_rollout_log_probs': True}}})
        rollout = OmegaConf.create({'n_samples_per_prompt': 2, 'global_batch_size': 2,
            'hybrid_batch_num': 1, 'enable_version_control': True})
        return train, rollout
    invalid_path = root/'destination'/'must-not-exist'
    bad = replace(config, tokenizer_signature='wrong', path=str(invalid_path))
    with pytest.raises(ValueError): prepare_replay(*configs(asdict(bad)))
    assert not invalid_path.exists()
    actor = prepare_replay(*configs(asdict(config)))
    try:
        assert real_ray.get(actor.verify_replay_sources.remote()) == {'sources': 1, 'unchanged': True}
    finally:
        real_ray.get(actor.close_replay_store.remote()); real_ray.kill(actor, no_restart=True)
