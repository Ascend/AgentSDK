"""Hybrid fixture — stage enum + pipeline directory, no full DAG."""

from enum import IntEnum


class Phase(IntEnum):
    INGEST = 1
    TRANSFORM = 2
