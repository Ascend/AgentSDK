from dataclasses import dataclass


@dataclass
class PreprocessStageContract:
    input_files: list[str]
    output_files: list[str]


PREPROCESS_CONTRACT = PreprocessStageContract(input_files=["raw/*"], output_files=["normalized.json"])
