# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
"""Exercise the real early-off-policy branch without importing NPU/Ray."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize('batch_count', [1, 3])
def test_second_batch_advances_version_window_without_loading_weights(batch_count):
    source = Path(__file__).resolve().parents[6] / 'aura/trainer/rollout/rollout_worker.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RolloutWorker')
    method = next(n for n in cls.body if getattr(n, 'name', '') == 'update_model_weights')
    module = ast.Module(body=[method], type_ignores=[])
    mocked_ray = MagicMock()
    namespace = {'ray': mocked_ray, 'logger': MagicMock()}
    exec(compile(module, str(source), 'exec'), namespace)
    worker = SimpleNamespace(
        use_on_policy=False, iteration=1, rollout_weight_manager=MagicMock())
    asyncio.run(namespace['update_model_weights'](worker, actual_batch_num=batch_count))
    worker.rollout_weight_manager.update_max_version.remote.assert_called_once_with(add_version_num=batch_count)
    mocked_ray.get.assert_called_once()
