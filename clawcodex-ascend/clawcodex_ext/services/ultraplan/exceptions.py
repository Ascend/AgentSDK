"""Ultraplan domain exceptions."""


class UltraplanError(RuntimeError):
    """Base error for ultraplan operations."""


class PlanNotFoundError(UltraplanError):
    """Raised when loading a plan id that does not exist on disk."""


class DuplicatePlanIdError(UltraplanError):
    """Raised when a plan with the same id is registered twice."""


class DuplicateStepIdError(UltraplanError):
    """Raised when a step id is reused within the same sub-plan."""


class DuplicateSubPlanIdError(UltraplanError):
    """Raised when a sub-plan id is reused within the same plan."""


class UnknownStepReferenceError(UltraplanError):
    """Raised when a step's ``depends_on`` references an unknown step id."""


class StepNotFoundError(UltraplanError):
    """Raised when operating on a step id that does not exist in the plan."""


class SubPlanNotFoundError(UltraplanError):
    """Raised when operating on a sub-plan id that does not exist in the plan."""


class IllegalStepTransitionError(UltraplanError):
    """Raised when a state transition violates the step state machine."""


class StepHasDependentsError(UltraplanError):
    """Raised when removing a step that is depended on by other steps."""


class PlanCorruptError(UltraplanError):
    """Raised when a plan file cannot be parsed or fails validation on load."""


class VerificationCheckFailedError(UltraplanError):
    """Raised when an acceptance criterion check returns a non-truthy result."""


class UnknownCheckKindError(UltraplanError):
    """Raised when a verifier is asked to run an unregistered check kind."""


class UnsafeCheckExpressionError(UltraplanError):
    """Raised when a ``PYTHON_PREDICATE`` expression fails the safety filter."""


class PlannerFailedError(UltraplanError):
    """Raised when an LLM response cannot be turned into a valid plan."""


class ProviderUnavailableError(UltraplanError):
    """Raised when the current LLM provider cannot be used for planning."""


class TemplateNotFoundError(UltraplanError):
    """Raised when a named ultraplan template cannot be found."""


class CCRUnavailableError(UltraplanError):
    """Raised when CCR remote execution is requested but unavailable."""


class CCRTimeoutError(UltraplanError):
    """Raised when a CCR remote stream times out."""
