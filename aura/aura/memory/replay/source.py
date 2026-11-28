# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Deterministic prompt cursor and real DataManager drain/restore adapter.

The cursor counts PRODUCED batches, not optimizer updates. Rebuilding verifies
the entire consumed prefix; changed data/order cannot silently resume. No tensor
or pickle is put in the source sidecar: ready batches contain durable refs only.
"""
import copy
import hashlib

from .codec import canonical, digest
from .schema import ReplayIntegrityError, integer


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ReplayIntegrityError('Unsupported prompt/cursor value')


class PromptCursor:
    def __init__(self, factory, identity, iterator=None):
        self.factory, self.identity = factory, identity
        self.iterator = iter(factory()) if iterator is None else iterator
        self.count, self.chain, self.exhausted = 0, digest([]), False

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            raise
        self.count += 1
        self.chain = digest([self.chain, plain(batch)])
        return batch

    def state(self):
        return {'format': 'aura.prompt-cursor.v1', 'identity': self.identity,
                'produced_batches': self.count, 'prefix_sha256': self.chain,
                'exhausted': self.exhausted}

    def restore(self, state):
        fields = {'format', 'identity', 'produced_batches', 'prefix_sha256', 'exhausted'}
        if (not isinstance(state, dict) or set(state) != fields
                or state['format'] != 'aura.prompt-cursor.v1'
                or state['identity'] != self.identity or type(state['exhausted']) is not bool):
            raise ReplayIntegrityError('Prompt source identity/cursor mismatch')
        integer(state['produced_batches'], 'produced batches')
        candidate = PromptCursor(self.factory, self.identity)
        for _ in range(state['produced_batches']):
            try:
                next(candidate)
            except StopIteration as error:
                raise ReplayIntegrityError('Prompt prefix shortened') from error
        if state['exhausted']:
            try:
                next(candidate)
            except StopIteration:
                pass
            else:
                raise ReplayIntegrityError('Prompt source exhaustion changed')
        if candidate.state() != state:
            raise ReplayIntegrityError('Prompt order/content changed during cursor reconstruction')
        self.iterator, self.count, self.chain, self.exhausted = (
            candidate.iterator, candidate.count, candidate.chain, candidate.exhausted)


class DataManagerSource:
    """Uses ONLY the controller copy held by the real VerlDataManager.

    Controller RPCs pause dispatch, finish the outstanding credit, wait for the
    rollout executor's active-generation barrier, then freeze the TrainQueue.
    Any ambiguity poisons the adapter; there is no destructive best-effort drain.
    """
    def __init__(self, controller, config, actor):
        self.controller, self.config, self.actor = controller, config, actor
        self.paused, self.failed = False, False

    def snapshot(self):
        if self.failed:
            raise RuntimeError('Uncertain source; restore from a paired commit')
        try:
            self.paused = True
            cursor, batches, metrics = self.controller.quiesce_replay_source()
            # Validate every opaque reference against the actual Episode actor.
            from .runtime import OnlineReplayReader
            loader = OnlineReplayReader(self.config, self.actor)
            queued = []
            for raw in batches:
                groups = loader.load_online(raw)
                versions = {s.weight_version for g in groups for s in g.samples}
                batch_ids = {s.source_batch_id for g in groups for s in g.samples}
                if len(versions) != 1 or len(batch_ids) != 1:
                    raise ReplayIntegrityError('Queued batch spans production versions/requests')
                queued.append({'batch_id': next(iter(batch_ids)), 'weight_version': next(iter(versions)),
                               'replay_refs': copy.deepcopy(raw['replay_refs'])})
            return {'format': 'aura.replay.source-state.v1', 'quiescent': True, 'prefetch': None,
                    'producer': {'identity': cursor['identity'], 'state': {
                        'cursor': cursor, 'queue_metrics': plain(metrics)}}, 'queued_batches': queued}
        except BaseException:
            self.failed = True
            raise

    def restore(self, source):
        try:
            state = source['producer']['state']
            if (source['quiescent'] is not True or source['prefetch'] is not None
                    or set(state) != {'cursor', 'queue_metrics'}
                    or source['producer']['identity'] != state['cursor']['identity']
                    or len(source['queued_batches']) != len(state['queue_metrics'])):
                raise ReplayIntegrityError('Invalid DataManager source state')
            from .runtime import OnlineReplayReader
            loader = OnlineReplayReader(self.config, self.actor)
            for raw in source['queued_batches']:
                groups = loader.load_online(raw)
                if any(s.weight_version != raw['weight_version'] or s.source_batch_id != raw['batch_id']
                       for g in groups for s in g.samples):
                    raise ReplayIntegrityError('Restored queue version/request mismatch')
            self.paused = True
            self.controller.restore_replay_source(state['cursor'], source['queued_batches'], state['queue_metrics'])
            return {'source_sha256': hashlib.sha256(canonical(source)).hexdigest()}
        except BaseException:
            self.failed = True
            raise

    def resume(self):
        if self.failed or not self.paused:
            raise RuntimeError('Only a successfully drained/restored source can resume')
        try:
            self.controller.resume_replay_source()
            self.paused = False
        except BaseException:
            self.failed = True
            raise
