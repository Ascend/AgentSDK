# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Explicit, local-filesystem FSDP / one-step-off paired recovery adapters."""
from collections.abc import Mapping
from pathlib import Path

from aura.memory.replay.codec import strict_json
from aura.memory.replay.schema import ReplayIntegrityError


def recovery_settings(config):
    if not isinstance(config, Mapping):
        return None  # Legacy configurations/mocks do not opt into this adapter.
    value = config.get('replay_recovery')
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('replay_recovery must be a mapping')
    if not value.get('enabled', False):
        return None
    allowed = {'enabled', 'root', 'ledger_path', 'branch', 'writer_request', 'from_branch', 'commit_id'}
    if set(value) - allowed or value.get('enabled') is not True:
        raise ValueError('Unsupported replay_recovery settings')
    from aura.memory.replay.config import replay_ratio
    from aura.memory.replay.schema import identifier
    if replay_ratio(config) <= 0:
        raise ValueError('Joint recovery requires enabled replay; zero retains the legacy path')
    if config.trainer.resume_mode != 'disable':
        raise ValueError('Use replay_recovery.from_branch, never the model-only trainer resume pointer')
    if (config.actor_rollout_ref.actor.strategy not in ('fsdp', 'fsdp2')
            or config.async_training.trigger_parameter_sync_step != 1
            or config.actor_rollout_ref.actor.ppo_epochs != 1
            or config.actor_rollout_ref.actor.ppo_mini_batch_size != config.data.train_batch_size
            or config.trainer.default_hdfs_dir is not None
            or config.trainer.del_local_ckpt_after_load):
        raise ValueError('Joint adapter requires local FSDP, one optimizer update/version and retained checkpoints')
    for key in ('root', 'ledger_path'):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise ValueError('Recovery paths must be absolute')
    for key in ('branch', 'writer_request'):
        identifier(value.get(key), key)
    from aura.memory.replay.joint import _commit_id
    _commit_id(value['branch'])  # Also a filesystem component, unlike a generic ledger identifier.
    if value.get('from_branch'):
        identifier(value['from_branch'], 'from_branch')
        if value['from_branch'] == value['branch']:
            raise ValueError('Recovery needs a fresh branch identity')
    if Path(value.root).resolve() != Path(config.trainer.default_local_dir).resolve():
        raise ValueError('Joint root must equal the local trainer checkpoint root')
    required = {'model', 'optimizer', 'extra', 'hf_model'}
    contents = config.actor_rollout_ref.actor.checkpoint
    if not required <= set(contents.save_contents) or not {'model', 'optimizer', 'extra'} <= set(contents.load_contents):
        raise ValueError('Paired recovery requires model/optimizer/extra and inference HF export')
    return dict(value)


def save_while_draining(session, source, model, root):
    """Join the real source barrier and native save before publishing anything.

    The caller holds the post-ACK training boundary: no optimizer update or
    mixer mutation can run here. Rollout generation lives on separate workers.
    A failed drain/save leaves only uncommitted artifacts; the trainer poisons
    the session and recovers from the previous paired point as before.
    """
    from concurrent.futures import ThreadPoolExecutor
    session._guard()
    if session.lease_receipt is not None or session.mixer.inflight is not None:
        raise RuntimeError('Parallel checkpoint requires a post-ACK quiescent mixer')
    step, version = session.mixer.completed_steps, session.policy_version
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='replay-native-save') as pool:
        pending = pool.submit(model.save, step, version)
        state = source.snapshot()
        receipt = pending.result()
    if session.mixer.completed_steps != step or session.policy_version != version:
        raise ReplayIntegrityError('Training boundary changed during checkpoint')

    def saved(requested_step, requested_version):
        if (requested_step, requested_version) != (step, version):
            raise ReplayIntegrityError('Pre-saved native checkpoint watermark mismatch')
        return receipt

    return session.checkpoint(root, state, saved)


class FSDPCallbacks:
    def __init__(self, workers, root, branch, owner, inference_check):
        from aura.memory.replay.joint import _commit_id, _root
        _commit_id(branch)
        root = _root(root)
        self.workers, self.root, self.branch = workers, Path(root), branch
        self.owner, self.inference_check = owner, inference_check
        self.workers.bind_replay_owner(owner)

    def confirm_update(self, version):
        receipts = self.workers.confirm_replay_update(version, self.owner)
        self._receipts(receipts, version, version)

    def save(self, step, version):
        if step != version:
            raise ReplayIntegrityError('Adapter requires one update per version')
        folder = self.root / 'model-states' / self.branch / f'global_step_{version}' / 'actor'
        folder.mkdir(parents=True, exist_ok=False)
        written = self.workers.write_replay_checkpoint(str(folder), step, version, self.owner)
        self._receipts(written, step, version)
        if any(p.is_symlink() for p in folder.rglob('*')):
            raise ReplayIntegrityError('Symlink native checkpoint component')
        expected = [f'{kind}_world_size_{r["world_size"]}_rank_{r["rank"]}.pt'
                    for r in written for kind in ('model', 'optim', 'extra_state')]
        expected.extend(f'replay_rank_{r["rank"]}.json' for r in written)
        if any(not (folder / name).is_file() for name in expected):
            raise ReplayIntegrityError('Native checkpoint is missing required rank files')
        roles = {r: [] for r in ('model', 'optimizer', 'scheduler', 'rng')}
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            relative = p.relative_to(self.root).as_posix()
            if p.name.startswith('optim_world_size_'):
                roles['optimizer'].append(relative)
            elif p.name.startswith('extra_state_world_size_'):
                roles['scheduler'].append(relative); roles['rng'].append(relative)
            else:
                roles['model'].append(relative)
        if not list((folder / 'huggingface').glob('*.safetensors')):
            raise ReplayIntegrityError('Inference checkpoint was not exported')
        return {'step': step, 'policy_version': version, 'artifacts': roles}

    @staticmethod
    def _receipts(receipts, step, version):
        if (not isinstance(receipts, list) or not receipts
                or {r['rank'] for r in receipts} != set(range(len(receipts)))
                or any(any(type(r[k]) is not int for k in ('step', 'policy_version', 'world_size', 'rank'))
                       or r['step'] != step or r['policy_version'] != version or r['world_size'] != len(receipts)
                       for r in receipts)):
            raise ReplayIntegrityError('Incomplete native worker checkpoint receipts')

    def load(self, paths, step, version):
        model_shards = [Path(p) for p in paths['model'] if Path(p).name.startswith('model_world_size_')]
        folders = {p.parent for p in model_shards}
        if len(folders) != 1:
            raise ReplayIntegrityError('Checkpoint shards do not share an actor directory')
        folder = next(iter(folders))
        actual = self.workers.load_replay_checkpoint(str(folder), step, version, self.owner)
        self._receipts(actual, step, version)
        for r in actual:
            saved = strict_json((folder / f'replay_rank_{r["rank"]}.json').read_bytes())
            if any(saved[key] != r[key] for key in ('step', 'policy_version', 'rank', 'world_size')):
                raise ReplayIntegrityError('Loaded native FSDP watermark differs from saved worker receipt')
        self.inference_check(str(folder / 'huggingface'), version)
        return {'step': step, 'policy_version': version}


def select_pair(settings, config, actor):
    from aura.memory.replay.transaction import ActorLedger
    from aura.memory.replay.joint import select_commit
    from dataclasses import asdict
    import ray
    ray.get(actor.configure_replay_ledger.remote(settings['ledger_path'], asdict(config)), timeout=config.rpc_timeout)
    ledger = ActorLedger(actor, config.rpc_timeout)
    loader = lambda tid, store_id=None: ray.get(actor.get_trajectory.remote(tid, store_id), timeout=config.rpc_timeout)
    branch = ledger.inspect(settings['from_branch'])['branch']
    return select_commit(settings['root'], config, loader, ledger.lookup_commit,
                         from_branch=settings['from_branch'], base_commit=branch.get('base_commit'),
                         commit_id=settings.get('commit_id'))
