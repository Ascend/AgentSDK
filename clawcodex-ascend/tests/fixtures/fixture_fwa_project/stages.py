"""FWA fixture — full workflow patterns."""

from enum import IntEnum


class Stage(IntEnum):
    PREPROCESS = 1
    ANALYZE = 2
    GENERATE = 3
