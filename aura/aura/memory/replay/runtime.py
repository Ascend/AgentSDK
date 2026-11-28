# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Actor-only publisher and trainer-side replay session; no NPU work."""
from collections import OrderedDict
from .codec import ReplayCodec, digest
from .conversion import training_batch, trajectory_groups
from .mixing import GroupMixer
from .schema import ReplayGroup, ReplayIntegrityError
from .store_common import MAX_BATCH_BYTES


def _persistence_chunks(groups):
    """Plan every request before writing; never split a GRPO group."""
    chunks, samples, total, offset = [], [], 0, 0
    for index, group in enumerate(groups):
        size = sum(len(ReplayCodec.prepare(sample)[1]) for sample in group.samples)
        if len(group.samples) > 256 or size > MAX_BATCH_BYTES:
            raise ValueError(
                f'Complete replay group {group.samples[0].group_id} requires {size} encoded bytes '
                f'and {len(group.samples)} trajectories; limits are {MAX_BATCH_BYTES} bytes and 256 '
                'trajectories. Reduce sequence lengths or GRPO group size before generation.')
        if samples and (len(samples) + len(group.samples) > 256 or total + size > MAX_BATCH_BYTES):
            chunks.append((offset, samples))
            samples, total, offset = [], 0, index
        samples.extend(group.samples)
        total += size
    if samples:
        chunks.append((offset, samples))
    return chunks


def connect_actor(config):
    import ray
    return ray.get_actor(config.actor_name, namespace=config.actor_namespace)


def persist_trajectories(trajectories, config, actor, batch_id, weight_version):
    import ray
    groups = trajectory_groups(trajectories, config, batch_id, weight_version)
    refs = []
    for offset, samples in _persistence_chunks(groups):
        receipt = ray.get(actor.append_replay_batch.remote(samples, digest([batch_id, offset])),
                          timeout=config.rpc_timeout)
        expected = [(s.trajectory_id, s.payload_checksum) for s in samples]
        if (receipt.store_id != config.store_id
                or [(key, checksum) for key, checksum, _ in receipt.sample_refs] != expected):
            raise ReplayIntegrityError("Durable receipt does not match the generated payload")
        refs.extend([{"trajectory_id": key, "checksum": checksum, "created_seq": seq,
                      "source_store_id": config.store_id} for key, checksum, seq in receipt.sample_refs])
    return refs


class ReplaySession:
    def __init__(self, config, pad_token_id, actor=None):
        self.config = config
        self.actor = connect_actor(config) if actor is None else actor
        if config.reuse_sources:
            import ray
            from dataclasses import asdict
            from .reuse import seeded_mixer
            ray.get(self.actor.configure_replay_sources.remote(asdict(config)), timeout=config.rpc_timeout)
            self.mixer = seeded_mixer(config, self._load_reuse_sample)
        else:
            self.mixer = GroupMixer(config)
        self.pad_token_id = pad_token_id

    def _load_reuse_sample(self, tid, store_id, group_id=None):
        import ray
        if group_id is not None:
            request = self.actor.get_reuse_group.remote(store_id, group_id)
        else:
            request = self.actor.get_trajectory.remote(tid, store_id)
        return ray.get(request, timeout=self.config.rpc_timeout)

    def load_online(self, raw):
        import ray
        refs = raw.get("replay_refs")
        expected = self.config.batch_groups * self.config.group_size
        if not isinstance(refs, list) or len(refs) != expected:
            raise ReplayIntegrityError("Missing/malformed durable replay references in online batch")
        if (any(not isinstance(ref, dict) or set(ref) !=
                {"trajectory_id", "checksum", "created_seq", "source_store_id"} for ref in refs)
                or len({ref["trajectory_id"] for ref in refs}) != expected):
            raise ReplayIntegrityError("Invalid/duplicate online durable reference")
        samples = []
        for offset in range(0, len(refs), 256):
            chunk = refs[offset:offset+256]
            received = ray.get(self.actor.get_replay_batch.remote([ref["trajectory_id"] for ref in chunk]),
                               timeout=self.config.rpc_timeout)
            if not isinstance(received, (list, tuple)) or len(received) != len(chunk):
                raise ReplayIntegrityError("Actor returned an incomplete replay payload batch")
            samples.extend(received)
        groups = OrderedDict()
        for sample, ref in zip(samples, refs):
            if (sample is None or sample.source_store_id != self.config.store_id
                    or sample.source_store_id != ref["source_store_id"]
                    or sample.payload_checksum != ref["checksum"] or sample.created_seq != ref["created_seq"]
                    or not sample.group_complete):
                raise ReplayIntegrityError("Referenced trajectory was not durably committed as a complete group")
            groups.setdefault((sample.source_store_id, sample.group_id), []).append(sample)
        return tuple(ReplayGroup(tuple(members)) for members in groups.values())

    def next_batch(self, fetch_raw, current_version):
        def fetch():
            raw = fetch_raw()
            if raw is None:
                raise ReplayIntegrityError("Online source exhausted while a whole batch was required")
            return self.load_online(raw)
        result = self.mixer.next(fetch, current_version)
        batch = training_batch(result.groups, self.pad_token_id)
        if self.config.reuse_sources:
            from .reuse import effective_version
            # Keep provenance versions, and expose the destination lag coordinate
            # separately. Never replace the stored behavior log probabilities.
            batch['replay_source_weight_versions'] = list(batch['replay_weight_versions'])
            batch['replay_weight_versions'] = [effective_version(s, self.config)
                for group in result.groups for s in group.samples]
        return batch

    def priority_updates(self, batch):
        if self.config.replay_strategy != "priority":
            return None
        from .sampling import group_priorities
        result = group_priorities(self.mixer.inflight.groups, batch.non_tensor_batch["uid"].tolist(),
                                 batch.batch["advantages"].detach().cpu().tolist(),
                                 batch.batch["response_mask"].detach().cpu().tolist())
        return self.mixer.validate_priority_updates(result)

    def commit_success(self, current_version, priority_updates=None):
        return self.mixer.commit_success(current_version, priority_updates)


class OnlineReplayReader:
    """Validate durable online refs without creating a sampler or reseeding.

    Production launcher/session must already mount foreign sources. Keep the
    explicit full source-integrity verification at the snapshot/restore boundary,
    but do not rebuild the unrelated foreign candidate window for each queue.
    The online payload validator is EXACTLY the same one used by the trainer.
    """
    load_online = ReplaySession.load_online

    def __init__(self, config, actor):
        self.config, self.actor = config, actor
        if config.reuse_sources:
            import ray
            receipt = ray.get(actor.verify_replay_sources.remote(), timeout=config.rpc_timeout)
            if receipt != {'sources': len(config.reuse_sources), 'unchanged': True}:
                raise ReplayIntegrityError('Snapshot/restore requires already mounted, verified reuse sources')
