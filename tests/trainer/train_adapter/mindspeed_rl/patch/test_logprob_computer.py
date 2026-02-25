import sys
import types
import importlib
import torch
import pytest


@pytest.fixture(scope="function")
def mock_mindspeed_rl():
    mindspeed_rl = types.ModuleType("mindspeed_rl")
    utils = types.ModuleType("mindspeed_rl.utils")

    compute_mod = types.ModuleType("mindspeed_rl.utils.compute")
    context_mod = types.ModuleType("mindspeed_rl.utils.context_parallel")
    padding_mod = types.ModuleType("mindspeed_rl.utils.remove_padding")

    def fake_compute_log_probs(output, labels):
        return torch.ones_like(labels, dtype=torch.float32)

    def fake_vocab_parallel_entropy(output):
        return torch.zeros(output.shape[0], output.shape[1])

    class DummyParallelState:
        @staticmethod
        def get_context_parallel_world_size():
            return 1

    def fake_get_parallel_state():
        return DummyParallelState()

    compute_mod.compute_log_probs = fake_compute_log_probs
    compute_mod.vocab_parallel_entropy = fake_vocab_parallel_entropy
    compute_mod.get_parallel_state = fake_get_parallel_state

    def tensor_allgather_cp_without_pack(tensor, cp_size, index):
        return tensor

    context_mod.get_tensor_allgather_cp_without_pack = tensor_allgather_cp_without_pack

    def tensor_allgather_cp_with_pack(tensor, cp_size, index):
        return tensor

    context_mod.get_tensor_allgather_cp_with_pack = tensor_allgather_cp_with_pack


    def postprocess_packed_seqs(tensor, *args, **kwargs):
        return tensor

    padding_mod.postprocess_packed_seqs = postprocess_packed_seqs


    sys.modules["mindspeed_rl"] = mindspeed_rl
    sys.modules["mindspeed_rl.utils"] = utils
    sys.modules["mindspeed_rl.utils.compute"] = compute_mod
    sys.modules["mindspeed_rl.utils.context_parallel"] = context_mod
    sys.modules["mindspeed_rl.utils.remove_padding"] = padding_mod

    yield

    for name in list(sys.modules.keys()):
        if name.startswith("mindspeed_rl"):
            del sys.modules[name]


@pytest.fixture
def module_under_test(mock_mindspeed_rl):
    module_path = (
        "agentic_rl.trainer.train_adapter.mindspeed_rl.patch.logprob_computer"
    )

    if module_path in sys.modules:
        del sys.modules[module_path]

    return importlib.import_module(module_path)


class DummyComputer:
    @staticmethod
    def _get_log_probs_remove_prompt_pad(tensor, batch):
        return tensor + 1


@pytest.fixture
def dummy_batch():
    return {
        "labels": torch.tensor([[1, 2, 3]]),
        "responses": torch.tensor([[1, 2, 3]]),
        "prompt_length": 0,
    }


@pytest.fixture
def dummy_output():
    return torch.randn(1, 3, 5)


def test_compute_without_remove_padding(
    module_under_test, dummy_batch, dummy_output
):
    compute = module_under_test.compute
    computer = DummyComputer()

    log_probs, entropy = compute(
        computer,
        dummy_output,
        dummy_batch,
        skip_entropy=False,
        use_remove_padding=False,
        index=None,
    )
    assert torch.all(log_probs == 2)
    assert torch.all(entropy == 1)
    

def test_compute_with_remove_padding(
    module_under_test, dummy_batch, dummy_output
):
    compute = module_under_test.compute
    computer = DummyComputer()

    log_probs, entropy = compute(
        computer,
        dummy_output,
        dummy_batch,
        skip_entropy=False,
        use_remove_padding=True,
        index=None,
        seqlens_in_batch=None,
        cu_seqlens_padded=None,
    )

    assert torch.all(log_probs == 1)
    assert torch.all(entropy == 0)
