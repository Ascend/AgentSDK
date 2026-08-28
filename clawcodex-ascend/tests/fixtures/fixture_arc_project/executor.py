"""ARC-style executor dispatch table fixture."""

from __future__ import annotations

from stages import Stage


def _execute_preprocess() -> None:
    pass


def _execute_analyze() -> None:
    pass


def _execute_generate() -> None:
    pass


_STAGE_EXECUTORS: dict[Stage, object] = {
    Stage.PREPROCESS: _execute_preprocess,
    Stage.ANALYZE: _execute_analyze,
    Stage.GENERATE: _execute_generate,
}
