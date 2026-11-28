# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Immutable CPU replay facts. No Ray, torch or numpy import is required."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from itertools import chain
import struct

SCHEMA_VERSION = 1
MAX_TENSOR_BYTES = 16 * 1024 * 1024
DTYPES = {"bool": "?", "uint8": "B", "int32": "i", "int64": "q",
          "float16": "e", "float32": "f", "float64": "d"}
COMPATIBILITY_FIELDS = ("group_size", "source_store_id", "source_run_id", "source_experiment_id",
                        "policy_lineage_id", "model_signature", "tokenizer_signature", "task_signature", "reward_signature")


class ReplayConflictError(ValueError):
    """An immutable sample, group slot or request identity was reused differently."""


class ReplayIntegrityError(ValueError):
    """Malformed, unsupported or checksum-invalid persisted data."""


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string of at most 512 characters")
    return value


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


@dataclass(frozen=True)
class ReplayTensor:
    """Bounded contiguous little-endian bytes, not an executable object format."""
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"Unsupported replay dtype: {self.dtype!r}")
        if not isinstance(self.shape, (list, tuple)) or len(self.shape) > 8:
            raise ValueError("Tensor shape must have at most 8 dimensions")
        shape = tuple(integer(n, "shape dimension", maximum=MAX_TENSOR_BYTES) for n in self.shape)
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("Tensor data must be bytes-like")
        data = bytes(self.data)
        expected = math.prod(shape) * struct.calcsize("<" + DTYPES[self.dtype])
        if expected > MAX_TENSOR_BYTES or len(data) != expected:
            raise ValueError("Tensor shape/byte length mismatch or size limit exceeded")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", data)
        if self.dtype.startswith("float") and not all(map(math.isfinite, self.values())):
            raise ValueError("Replay tensors must contain finite values")

    def values(self):
        # Decode bounded blocks instead of allocating one tuple per scalar.
        # Keep streaming behavior even for the largest accepted tensor.
        code = DTYPES[self.dtype]
        width = struct.calcsize('<' + code)
        end = len(self.data) // (64 * width) * (64 * width)
        view = memoryview(self.data)
        blocks = struct.iter_unpack('<64' + code, view[:end])
        tail = struct.unpack('<' + str((len(self.data) - end) // width) + code, view[end:])
        return chain(chain.from_iterable(blocks), tail)

    @classmethod
    def from_values(cls, values, dtype="int64"):
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported replay dtype: {dtype!r}")
        values = tuple(values)
        return cls(dtype, (len(values),), struct.pack("<" + DTYPES[dtype] * len(values), *values))

    @classmethod
    def from_numpy(cls, value):
        import numpy as np
        array = np.asarray(value)
        shape, dtype = tuple(array.shape), array.dtype.name
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported replay dtype: {dtype!r}")
        array = np.ascontiguousarray(array.astype(np.dtype(dtype).newbyteorder("<"), copy=False))
        return cls(dtype, shape, array.tobytes())

    def to_numpy(self):
        import numpy as np
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype).newbyteorder("<")).reshape(self.shape).copy()

    @classmethod
    def from_torch(cls, value):
        if value.device.type != "cpu":
            raise ValueError("CPU tensors only; device transfer must be explicit upstream")
        if str(value.layout) != "torch.strided":
            raise ValueError("Only dense strided tensors are supported")
        if str(value.dtype).removeprefix("torch.") not in DTYPES:
            raise ValueError(f"Unsupported replay dtype: {value.dtype}")
        return cls.from_numpy(value.detach().contiguous().numpy())

    def to_torch(self):
        import torch
        return torch.from_numpy(self.to_numpy())


@dataclass(frozen=True)
class FrozenPayload(Mapping):
    entries: tuple[tuple[str, ReplayTensor], ...]

    def __getitem__(self, key):
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (name for name, _ in self.entries)

    def __len__(self):
        return len(self.entries)


@dataclass(frozen=True)
class ReplaySample:
    source_store_id: str
    trajectory_id: str
    prompt_id: str
    group_id: str
    sample_index: int
    group_size: int
    source_experiment_id: str
    source_run_id: str
    policy_lineage_id: str
    weight_version: int
    reward: float
    payload: Mapping[str, ReplayTensor]
    old_log_probs: ReplayTensor | None
    model_signature: str
    tokenizer_signature: str
    task_signature: str
    reward_signature: str
    source_batch_id: str
    schema_version: int = SCHEMA_VERSION
    train_eligible: bool = True
    invalid_reason: str | None = None
    # Store-assigned metadata, excluded from the immutable checksum.
    created_seq: int | None = None
    group_complete: bool | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported ReplaySample schema version")
        for name in ("source_store_id", "trajectory_id", "prompt_id", "group_id", "source_experiment_id",
                     "source_run_id", "policy_lineage_id", "model_signature", "tokenizer_signature",
                     "task_signature", "reward_signature", "source_batch_id"):
            identifier(getattr(self, name), name)
        integer(self.group_size, "group_size", 1, 256)
        integer(self.sample_index, "sample_index", maximum=self.group_size - 1)
        integer(self.weight_version, "weight_version", maximum=2**63 - 1)
        if type(self.reward) not in (int, float) or not math.isfinite(self.reward):
            raise ValueError("reward must be a finite real number, not bool")
        object.__setattr__(self, "reward", float(self.reward))
        if type(self.train_eligible) is not bool:
            raise ValueError("train_eligible must be bool")
        if self.train_eligible and self.invalid_reason is not None:
            raise ValueError("Eligible samples cannot have invalid_reason")
        if not self.train_eligible:
            identifier(self.invalid_reason, "invalid_reason")
        if self.created_seq is not None:
            integer(self.created_seq, "created_seq", 1)
        if self.group_complete is not None and type(self.group_complete) is not bool:
            raise ValueError("group_complete must be bool or None")
        if not isinstance(self.payload, Mapping) or not 3 <= len(self.payload) <= 64:
            raise ValueError("payload must be a bounded ReplayTensor mapping")
        for name, value in self.payload.items():
            identifier(name, "payload key")
            if not isinstance(value, ReplayTensor):
                raise TypeError("payload values must be ReplayTensor")
        object.__setattr__(self, "payload", FrozenPayload(tuple(sorted(self.payload.items()))))
        for name in ("prompts", "responses", "response_mask"):
            if name not in self.payload:
                raise ValueError(f"Missing payload field: {name}")
        for name in ("prompts", "responses"):
            t = self.payload[name]
            if len(t.shape) != 1 or t.dtype not in ("int32", "int64") or min(t.values(), default=0) < 0:
                raise ValueError(f"{name} must be a nonnegative integer token vector")
        if not self.payload["prompts"].shape[0]:
            raise ValueError("prompts cannot be empty")
        n = self.payload["responses"].shape[0]
        mask = self.payload["response_mask"]
        if mask.shape != (n,) or mask.dtype not in ("bool", "uint8", "int32", "int64"):
            raise ValueError("response_mask must align with responses")
        if not all(map((0, 1).__contains__, mask.values())):
            raise ValueError("response_mask must be binary")
        if self.train_eligible and not n:
            raise ValueError("Eligible samples need nonempty responses")
        if self.old_log_probs is not None:
            if (not isinstance(self.old_log_probs, ReplayTensor) or self.old_log_probs.dtype != "float32"
                    or self.old_log_probs.shape != (n,) or max(self.old_log_probs.values(), default=-math.inf) > 0):
                raise ValueError("old_log_probs must be aligned finite FP32 log probabilities")
        elif self.train_eligible:
            raise ValueError("Eligible samples require original old_log_probs")
        for name in ("token_level_rewards", "rm_scores"):
            if name in self.payload and self.payload[name].shape != (n,):
                raise ValueError(f"{name} must align with responses")
        # Derived only from validated immutable metadata, not a schema field.
        # It is rebuilt on JSON decode and is excluded from canonical checksums.
        object.__setattr__(self, "_compatibility_key", tuple(getattr(self, k) for k in COMPATIBILITY_FIELDS))

    @property
    def sample_key(self):
        return self.source_store_id, self.trajectory_id

    @property
    def prompt_key(self):
        from .codec import digest, tensor_document
        return digest(tensor_document(self.payload["prompts"]))

    @property
    def payload_checksum(self):
        from .codec import ReplayCodec
        return ReplayCodec.document(self)["checksum"]

    @property
    def tensor_specs(self):
        tensors = dict(self.payload)
        if self.old_log_probs is not None:
            tensors["old_log_probs"] = self.old_log_probs
        return {name: {"dtype": t.dtype, "shape": t.shape, "byteorder": "little", "codec": "base64"}
                for name, t in tensors.items()}


GROUP_FIELDS = ("source_store_id", "group_id", "prompt_id", "group_size", "source_experiment_id",
                "source_run_id", "source_batch_id", "policy_lineage_id", "weight_version",
                "model_signature", "tokenizer_signature", "task_signature", "reward_signature", "train_eligible")


@dataclass(frozen=True)
class ReplayGroup:
    samples: tuple[ReplaySample, ...]

    def __post_init__(self):
        samples = tuple(sorted(self.samples, key=lambda s: s.sample_index))
        if not samples:
            raise ValueError("A replay group cannot be empty")
        object.__setattr__(self, "samples", samples)
        first = samples[0]
        if any(any(getattr(s, k) != getattr(first, k) for k in GROUP_FIELDS)
               or s.payload["prompts"] != first.payload["prompts"] for s in samples):
            raise ReplayConflictError("Mixed prompt, source, policy, signature or eligibility in group")
        if len({s.sample_index for s in samples}) != len(samples) or len({s.trajectory_id for s in samples}) != len(samples):
            raise ReplayConflictError("Duplicate group member index or trajectory_id")
        if first.train_eligible and not self.complete:
            raise ValueError("Incomplete eligible group; archive explicitly instead of padding")

    @property
    def complete(self):
        return len(self.samples) == self.samples[0].group_size


@dataclass(frozen=True)
class ReplayQuery:
    trajectory_id: str | None = None
    prompt_id: str | None = None
    group_id: str | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    weight_version: int | None = None
    weight_version_min: int | None = None
    weight_version_max: int | None = None
    policy_lineage_id: str | None = None
    source_store_id: str | None = None
    source_experiment_id: str | None = None
    source_run_id: str | None = None
    train_eligible: bool | None = None
    limit: int = 100

    def __post_init__(self):
        integer(self.limit, "limit", maximum=1000)
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None or field.name == "limit":
                continue
            if field.name.startswith("weight_version"):
                integer(value, field.name, maximum=2**63 - 1)
            elif field.name in ("reward_min", "reward_max"):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Reward bounds must be finite")
            elif field.name == "train_eligible":
                if type(value) is not bool:
                    raise ValueError("train_eligible must be bool or None")
            else:
                identifier(value, field.name)
        for low, high in ((self.reward_min, self.reward_max), (self.weight_version_min, self.weight_version_max)):
            if low is not None and high is not None and low > high:
                raise ValueError("Reversed query bounds")


@dataclass(frozen=True)
class ReplayPage:
    samples: tuple[ReplaySample, ...]
    next_cursor: str | None
    snapshot_seq: int


@dataclass(frozen=True)
class ReplayReceipt:
    store_id: str
    request_id: str
    request_checksum: str
    sample_refs: tuple[tuple[str, str, int], ...]
    inserted_count: int
    duplicate_count: int
    replayed: bool = False
