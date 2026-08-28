"""ARC-style pipeline fixture (dict comp NEXT_STAGE + CONTRACTS dict)."""

from __future__ import annotations

from enum import IntEnum


class Stage(IntEnum):
    PREPROCESS = 1
    ANALYZE = 2
    GENERATE = 3


STAGE_SEQUENCE: tuple[Stage, ...] = tuple(Stage)

NEXT_STAGE: dict[Stage, Stage | None] = {
    stage: STAGE_SEQUENCE[idx + 1] if idx + 1 < len(STAGE_SEQUENCE) else None
    for idx, stage in enumerate(STAGE_SEQUENCE)
}

GATE_STAGES: frozenset[Stage] = frozenset({Stage.ANALYZE})

DECISION_ROLLBACK: dict[str, Stage] = {
    "refine": Stage.PREPROCESS,
    "proceed": Stage.GENERATE,
}
