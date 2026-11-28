# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
"""Regression for the pinned verl dense-input compatibility patch."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('response_mask', [[1, 1, 1], [1, 0, 0]])
def test_attention_mask_covers_input_not_loss_tokens(response_mask):
    torch = pytest.importorskip('torch')
    engine = pytest.importorskip('verl.workers.engine.fsdp.transformer_impl')

    class MicroBatch(dict):
        batch_size = (2,)

    inputs = [torch.arange(9), torch.arange(6)]
    micro = MicroBatch(
        input_ids=torch.nested.as_nested_tensor(inputs, layout=torch.jagged),
        position_ids=torch.nested.as_nested_tensor(inputs, layout=torch.jagged),
        loss_mask=torch.tensor([response_mask, response_mask]),
        temperature=1.0,
        use_remove_padding=False,
    )
    model_inputs, _ = engine.FSDPEngineWithLMHead.prepare_model_inputs(
        SimpleNamespace(ulysses_sequence_parallel_size=1), micro)
    expected = torch.tensor([[1] * 9, [1] * 6 + [0] * 3], dtype=torch.int32)
    assert torch.equal(model_inputs['attention_mask'], expected)
    assert torch.equal(micro['loss_mask'], torch.tensor([response_mask, response_mask]))
