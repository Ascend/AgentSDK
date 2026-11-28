# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
import ast
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def source_tree():
    source = Path(__file__).resolve().parents[6] / 'aura/trainer/rollout/rollout_service.py'
    return source, ast.parse(source.read_text(encoding='utf-8'))


@pytest.mark.parametrize('entry', ['start_rollout', 'start_async_rollout_worker'])
def test_restore_precedes_ready_in_both_entries(entry):
    _, tree = source_tree()
    function = next(n for n in tree.body if getattr(n, 'name', '') == entry)
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
    restore = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == '_restore_inference_checkpoint')
    ready = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'send_ready_to_train')
    assert restore.lineno < ready.lineno


@pytest.mark.parametrize('path', ['', '/checkpoints/global_step_5/actor/huggingface'])
@pytest.mark.parametrize('count', [1, 2, 3])
def test_restore_loads_weights_and_resets_cache_or_is_cold_start_noop(path, count):
    source, tree = source_tree()
    function = next(n for n in tree.body if getattr(n, 'name', '') == '_restore_inference_checkpoint')
    namespace = {'os': os, 'asyncio': asyncio, 'logger': MagicMock()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    router = SimpleNamespace(update_weights=AsyncMock(return_value=[None] * count),
                             reset_prefix_cache=AsyncMock(return_value=[True] * count),
                             infer_manager=SimpleNamespace(get_instance=SimpleNamespace(
                                 remote=AsyncMock(return_value=SimpleNamespace(executor_list=[object()] * count)))))
    factory = SimpleNamespace(create=AsyncMock(return_value=router))
    with patch.dict(os.environ, {'AURA_RESUME_HF_PATH': path}), patch.dict(
        sys.modules, {'aura.runner.infer_router': SimpleNamespace(InferRouter=factory)}
    ):
        namespace['_restore_inference_checkpoint']('model')
    if path:
        router.update_weights.assert_awaited_once_with(model_name='model', kwargs_list=[{'path': [path]}] * count)
        router.reset_prefix_cache.assert_awaited_once_with(model_name='model')
    else:
        factory.create.assert_not_awaited()


@pytest.mark.parametrize('fault', ['load', 'load_count', 'cache_false', 'cache_count', 'cache_503'])
def test_restore_failure_cannot_report_ready(fault):
    source, tree = source_tree()
    function = next(n for n in tree.body if getattr(n, 'name', '') == '_restore_inference_checkpoint')
    namespace = {'os': os, 'asyncio': asyncio, 'logger': MagicMock()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    router = SimpleNamespace(update_weights=AsyncMock(return_value=[None, None]),
        reset_prefix_cache=AsyncMock(return_value=[True, True]), infer_manager=SimpleNamespace(
            get_instance=SimpleNamespace(remote=AsyncMock(return_value=SimpleNamespace(executor_list=[1, 2])))))
    if fault == 'load': router.update_weights.side_effect = RuntimeError('load failed')
    if fault == 'load_count': router.update_weights.return_value = [None]
    if fault == 'cache_false': router.reset_prefix_cache.return_value = [True, False]
    if fault == 'cache_count': router.reset_prefix_cache.return_value = [True]
    if fault == 'cache_503': router.reset_prefix_cache.side_effect = RuntimeError('HTTP 503')
    factory = SimpleNamespace(create=AsyncMock(return_value=router))
    with patch.dict(os.environ, {'AURA_RESUME_HF_PATH': '/checkpoint'}), patch.dict(
            sys.modules, {'aura.runner.infer_router': SimpleNamespace(InferRouter=factory)}):
        with pytest.raises(RuntimeError):
            namespace['_restore_inference_checkpoint']('model')
    namespace['logger'].info.assert_not_called()
    if fault.startswith('load'):
        router.reset_prefix_cache.assert_not_awaited()
