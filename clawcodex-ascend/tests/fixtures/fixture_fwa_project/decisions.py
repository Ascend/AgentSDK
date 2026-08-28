from stages import Stage


def decide_pivot(score: float) -> str:
    """Decide whether to proceed or rollback."""
    if score < 0.5:
        return "refine"
    return "proceed"


def should_continue() -> Stage:
    return Stage.GENERATE
