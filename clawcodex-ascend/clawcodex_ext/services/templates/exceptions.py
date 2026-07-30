"""Templates service-layer exceptions."""

from __future__ import annotations


class TemplatesError(RuntimeError):
    """Base error for templates operations."""


class TemplateNotFoundError(TemplatesError):
    """Raised when a named template cannot be located in the registry."""


class TemplateAlreadyExistsError(TemplatesError):
    """Raised when registering a template id that is already present."""


class TemplateValidationError(TemplatesError):
    """Raised when a template payload fails domain validation.

    Distinct from a generic ``ValueError`` raised by ``__post_init__``:
    this is raised when a template is added to a registry or stored on
    disk in a state that violates the contracts documented in the plan.
    """


class TemplateCorruptError(TemplatesError):
    """Raised when a template file cannot be parsed on load.

    Triggered by malformed YAML, missing required keys, or a from_dict
    payload that fails strict validation. The original parse error is
    chained via ``__cause__`` so callers can surface it.
    """


class TemplateResolutionError(TemplatesError):
    """Raised when a resolver cannot reconcile a base + override pair.

    Example: an override references a field that is not declared by the
    base template, or the merge produces an inconsistent type (e.g.
    overriding a string with a dict).
    """


# ---------------------------------------------------------------------------
# F-95 (P95-B / P95-C / P95-G) — product-layer exceptions
# ---------------------------------------------------------------------------


class TemplateRenderError(TemplatesError):
    """Raised when a template cannot be rendered.

    Examples: a required variable was not supplied, a value violates the
    variable's ``pattern`` / ``choices`` constraint, or the template body
    references a name that does not match any declared variable.
    """


class TemplateUnsafePathError(TemplatesError):
    """Raised when a generated output path escapes the workspace root.

    The generator (:class:`TemplateGenerator`) rejects any resolved path
    that falls outside the configured workspace so a malicious or
    misconfigured template cannot write outside the user's project.
    """


class TemplateOverwriteError(TemplatesError):
    """Raised when the generator would overwrite an existing file.

    The generator's default behaviour is no-overwrite; the caller must
    pass ``overwrite=True`` to explicitly authorise the replacement.
    """


class TemplateCompatibilityError(TemplatesError):
    """Raised when a template is not compatible with the current clawcodex
    install — its declared ``schema_version`` is unsupported, or its
    ``min_clawcodex_version`` is higher than the running build.
    """
