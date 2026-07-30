"""Context collapse domain exceptions."""


class ContextCollapseError(RuntimeError):
    """Base error for context collapse operations."""


class TokenCountUnavailableError(ContextCollapseError):
    """Raised when no token counter implementation can be loaded.

    This is distinct from a zero-count: it means the runtime could not
    find ``tiktoken`` and no fallback was registered. The caller can
    treat this as a transient error (e.g. log a warning and use a
    heuristic counter instead of failing the request).
    """


class SummaryGeneratorError(ContextCollapseError):
    """Raised when a registered summary generator fails to produce output."""


class CollapseStateCorruptError(ContextCollapseError):
    """Raised when a collapse state file cannot be parsed on load."""


class CollapseStateNotFoundError(ContextCollapseError):
    """Raised when loading a collapse state file that does not exist."""


class ContextLengthExceededError(ContextCollapseError):
    """Raised when the input token count exceeds the configured ceiling.

    The 413 emergency-recovery path catches this and triggers a
    single-shot collapse, then re-raises if the context is still over
    budget after the recovery attempt.
    """
