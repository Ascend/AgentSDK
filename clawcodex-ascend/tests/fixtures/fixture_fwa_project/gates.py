from stages import Stage

GATE_STAGES = frozenset({Stage.ANALYZE})

GATE_ROLLBACK = {
    Stage.ANALYZE: Stage.PREPROCESS,
}
