"""Kairos / Brief scheduling service-layer exceptions."""


class KairosError(RuntimeError):
    """Base error for kairos operations."""


class TickConfigError(KairosError):
    """Raised when a :class:`TickConfig` fails validation."""


class SchedulerStateError(KairosError):
    """Raised when a scheduler operation is invalid for the current state.

    Examples: starting an already-running scheduler, stopping one that
    was never started, or registering a callback after shutdown.
    """


class DailyLogError(KairosError):
    """Raised when the daily log writer cannot append or read an entry."""


class BriefGenerationError(KairosError):
    """Raised when a brief cannot be produced from the supplied snapshot."""
