"""ARC-style contracts module fixture."""

from __future__ import annotations

from dataclasses import dataclass

from stages import Stage


@dataclass(frozen=True)
class StageContract:
    stage: Stage
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]


CONTRACTS: dict[Stage, StageContract] = {
    Stage.PREPROCESS: StageContract(
        stage=Stage.PREPROCESS,
        input_files=(),
        output_files=("normalized.json",),
    ),
    Stage.ANALYZE: StageContract(
        stage=Stage.ANALYZE,
        input_files=("normalized.json",),
        output_files=("analysis.json",),
    ),
    Stage.GENERATE: StageContract(
        stage=Stage.GENERATE,
        input_files=("analysis.json",),
        output_files=("output.json",),
    ),
}
