from dataclasses import FrozenInstanceError, replace
import json
import struct
from types import SimpleNamespace

import pytest

from aura.memory.replay.codec import ReplayCodec, canonical, digest
from aura.memory.replay.schema import ReplayGroup, ReplayIntegrityError, ReplayTensor


def test_exact_roundtrip_and_immutable_snapshot(make_group):
    s = make_group()[0]
    encoded = ReplayCodec.encode(s)
    restored = ReplayCodec.decode(encoded)
    assert restored == s
    assert restored.old_log_probs.data == s.old_log_probs.data
    assert restored.payload["response_mask"].dtype == "bool"
    assert type(restored.train_eligible) is bool and type(restored.weight_version) is int
    assert restored.sample_key == ("store", s.trajectory_id)
    assert len(restored.prompt_key) == 64 and restored.prompt_key == s.prompt_key
    assert restored.payload_checksum == json.loads(encoded)["checksum"]
    assert restored.tensor_specs["old_log_probs"] == {
        "dtype": "float32", "shape": (3,), "byteorder": "little", "codec": "base64"}
    assert ReplayCodec.encode(replace(s, created_seq=100, group_complete=True)) == encoded
    with pytest.raises(FrozenInstanceError):
        restored.reward = 2
    with pytest.raises(TypeError):
        restored.payload["prompts"] = ReplayTensor.from_values([0])


def test_caller_dictionary_cannot_change_sample(make_group):
    s = make_group()[0]
    payload = dict(s.payload)
    copy = replace(s, payload=payload)
    payload.clear()
    assert len(copy.payload) == 4


@pytest.mark.parametrize("dtype,values", [("int64", [2**53 + 1]), ("int32", [-2, 3]),
    ("float16", [-0.5, 0.0]), ("float32", [-0.0, -0.25]), ("float64", [0.1]), ("bool", [True, False]), ("uint8", [0, 255])])
def test_dtype_shape_and_bytes_preserved(dtype, values):
    t = ReplayTensor.from_values(values, dtype)
    assert t.shape == (len(values),)
    assert list(t.values()) == values


@pytest.mark.parametrize("change", [dict(schema_version=2), dict(schema_version=True), dict(weight_version=True),
    dict(weight_version=-1), dict(reward=True), dict(reward=float("nan")), dict(reward=float("inf")),
    dict(group_size=0), dict(sample_index=2), dict(trajectory_id=""), dict(old_log_probs=None),
    dict(train_eligible=False), dict(invalid_reason="bad")])
def test_bad_sample_metadata_rejected(make_group, change):
    with pytest.raises((ValueError, TypeError)):
        replace(make_group()[0], **change)


@pytest.mark.parametrize("tensor", [lambda: ReplayTensor("object", (1,), b"0"),
    lambda: ReplayTensor("int64", (True,), b"0" * 8), lambda: ReplayTensor("int64", (2,), b"0" * 8),
    lambda: ReplayTensor("float32", (1,), struct.pack("<f", float("nan"))),
    lambda: ReplayTensor("float32", (1,), struct.pack("<f", float("inf")))])
def test_unsafe_tensor_specs_rejected(tensor):
    with pytest.raises(ValueError):
        tensor()


@pytest.mark.parametrize("key,tensor", [("prompts", ReplayTensor.from_values([-1])),
    ("responses", ReplayTensor.from_values([1.0], "float32")),
    ("response_mask", ReplayTensor.from_values([1, 2, 0])),
    ("response_mask", ReplayTensor.from_values([1])),
    ("token_level_rewards", ReplayTensor.from_values([1.0], "float32"))])
def test_misaligned_or_invalid_payload_rejected(make_group, key, tensor):
    sample = make_group()[0]
    with pytest.raises(ValueError):
        replace(sample, payload={**sample.payload, key: tensor})


def test_real_response_with_zero_loss_mask_roundtrips(make_group):
    sample = make_group()[0]
    mask = ReplayTensor.from_values([0, 0, 0])
    sample = replace(sample, payload={**sample.payload, "response_mask": mask})
    restored = ReplayCodec.decode(ReplayCodec.encode(sample))
    assert restored.train_eligible
    assert restored.payload["response_mask"] == mask
    assert restored.payload["responses"] == sample.payload["responses"]


@pytest.mark.parametrize("value", [ReplayTensor.from_values([-1.0], "float32"),
    ReplayTensor.from_values([-1.0, -1.0, -1.0], "float64"),
    ReplayTensor.from_values([0.1, -1.0, -1.0], "float32")])
def test_bad_log_probabilities_rejected(make_group, value):
    with pytest.raises(ValueError):
        replace(make_group()[0], old_log_probs=value)


def test_checksum_and_semantic_checks_are_both_required(make_group):
    doc = json.loads(ReplayCodec.encode(make_group()[0]))
    doc["sample"]["reward"] = 8
    with pytest.raises(ReplayIntegrityError):
        ReplayCodec.decode(canonical(doc))
    doc["sample"]["payload"]["responses"]["shape"] = [100]
    doc["checksum"] = digest(doc["sample"])
    with pytest.raises(ReplayIntegrityError):
        ReplayCodec.decode(canonical(doc))


@pytest.mark.parametrize("checksum", ["非" * 64, "z" * 64, "short", None])
def test_malformed_checksum_is_a_typed_integrity_error(make_group, checksum):
    doc = json.loads(ReplayCodec.encode(make_group()[0]))
    doc["checksum"] = checksum
    with pytest.raises(ReplayIntegrityError):
        ReplayCodec.decode(canonical(doc))


@pytest.mark.parametrize("data", [b'\x80\x04pickle', b'{"x":1,"x":2}', b'{"value":NaN}', b'[]'])
def test_no_pickle_duplicate_keys_or_nan(data):
    with pytest.raises(ReplayIntegrityError):
        ReplayCodec.decode(data)


def test_numpy_noncontiguous_endian_and_copy():
    np = pytest.importorskip("numpy")
    array = np.arange(12, dtype=">i8").reshape(3, 4).T
    encoded = ReplayTensor.from_numpy(array)
    decoded = encoded.to_numpy()
    assert decoded.shape == (4, 3) and (decoded == array).all()
    decoded[0, 0] = 999
    assert list(encoded.values())[0] == 0
    assert ReplayTensor.from_numpy(np.array(2.5, dtype="float32")).shape == ()
    with pytest.raises(ValueError):
        ReplayTensor.from_numpy(np.array([object()], dtype=object))


def test_non_cpu_tensor_is_never_silently_copied():
    with pytest.raises(ValueError, match="CPU"):
        ReplayTensor.from_torch(SimpleNamespace(device=SimpleNamespace(type="npu")))


def test_optional_real_torch_cpu_roundtrip():
    torch = pytest.importorskip("torch")
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device="cpu").T
    assert torch.equal(ReplayTensor.from_torch(t).to_torch(), t)


def test_group_cannot_mix_policy_or_prompt(make_group):
    group = make_group()
    with pytest.raises(ValueError):
        ReplayGroup((group[0],))
    with pytest.raises(ValueError):
        ReplayGroup((group[0], replace(group[1], weight_version=4)))
    assert ReplayGroup(tuple(group)).complete
