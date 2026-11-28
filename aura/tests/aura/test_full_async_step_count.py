"""Exercise the real fit_step body without importing Ray or NPU dependencies."""
import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize("start,total", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_exact_completed_steps(start, total):
    path = Path(__file__).resolve().parents[2] / "aura/trainer/train_adapter/verl/full_async/full_async_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FullyAsyncTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "fit_step")
    logger = Mock()
    stop = type("TrainingStopException", (Exception,), {})
    namespace = {"logger": logger, "TrainingStopException": stop,
                 "marked_timer": lambda *a, **k: nullcontext()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    trainer = SimpleNamespace(global_steps=start, total_train_steps=total,
                              epoch=0, local_trigger_step=1, _replay_session=None)
    for node in ast.walk(method):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_fit_"):
            setattr(trainer, node.attr, Mock(side_effect=lambda *a, **k: a[0] if a else None))
    trainer._fit_generate = AsyncMock(return_value=object())
    trainer._fit_update_weights = AsyncMock()
    completed = []

    def postprocess():
        completed.append(trainer.metrics["training/global_step"])
        trainer.global_steps += 1

    trainer._fit_postprocess_step = postprocess

    async def run():
        # Bound the loop so a broken termination condition cannot hang the test.
        for _ in range(total - start + 2):
            try:
                await namespace["fit_step"](trainer)
            except stop:
                return
        pytest.fail("Training did not stop at the requested step")

    asyncio.run(run())
    assert completed == list(range(start, total + 1))
    assert trainer._fit_update_actor.call_count == total - start + 1
    assert trainer._fit_update_weights.await_count == total - start + 1
    assert trainer.global_steps == total + 1
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert [m for m in messages if m.startswith("finished train iteration")] == [
        f"finished train iteration {step}/{total}" for step in completed
    ]
