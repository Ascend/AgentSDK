from stages import Stage

NEXT_STAGE = {
    Stage.PREPROCESS: Stage.ANALYZE,
    Stage.ANALYZE: Stage.GENERATE,
}
