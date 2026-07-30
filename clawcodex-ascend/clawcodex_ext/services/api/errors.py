from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import httpx

    _HTTPX_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError,
        httpx.ProxyError,
        httpx.UnsupportedProtocol,
    )
except ImportError:  # httpx is optional at runtime for some embed paths
    httpx = None  # type: ignore[assignment]
    _HTTPX_TRANSPORT_ERRORS = ()

API_ERROR_MESSAGE_PREFIX = "API Error"
PROMPT_TOO_LONG_ERROR_MESSAGE = "Prompt is too long"
# Surfaced when the provider rejects an image because the selected model
# does not accept image input (e.g. OpenRouter routing a request to a
# text-only DeepSeek endpoint and returning
# "No endpoints found that support image input" / 404).
#
# The user-facing wording follows the shape of TypeScript's friendly
# error messages at typescript/src/services/api/errors.ts (e.g.
# getPdfInvalidErrorMessage / getImageTooLargeErrorMessage) — clear
# about what happened and what the user can do. The TypeScript code
# has no dedicated handler for this specific capability rejection (the
# error would land in the catch-all at typescript/src/query.ts:1065),
# so this Python branch is genuinely new behaviour, not a port. We
# additionally strip the offending image from history (see
# QueryEngine.submit_message); TypeScript expects the user to manually
# "Double press esc" via the Ink MessageSelector, which the Rich REPL
# does not have.
IMAGE_UNSUPPORTED_ERROR_MESSAGE = (
    "The current model does not accept image input. The image has been "
    "removed from conversation history so subsequent requests will work. "
    "Switch to a vision-capable model to process images."
)


class PromptTooLongError(Exception):
    def __init__(
        self,
        message: str = PROMPT_TOO_LONG_ERROR_MESSAGE,
        actual_tokens: int | None = None,
        limit_tokens: int | None = None,
    ):
        super().__init__(message)
        self.actual_tokens = actual_tokens
        self.limit_tokens = limit_tokens

    @property
    def token_gap(self) -> int | None:
        if self.actual_tokens is not None and self.limit_tokens is not None:
            gap = self.actual_tokens - self.limit_tokens
            return gap if gap > 0 else None
        return None


class MaxOutputTokensError(Exception):
    def __init__(self, message: str = "Max output tokens reached"):
        super().__init__(message)


class RateLimitError(Exception):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status: int = 429,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class OverloadedError(Exception):
    def __init__(self, message: str = "API overloaded", status: int = 529):
        super().__init__(message)
        self.status = status


class FallbackTriggeredError(Exception):
    """Downstream compatibility signal for the retained retry engine."""

    def __init__(self, original_model: str, fallback_model: str):
        super().__init__(f"Model fallback triggered: {original_model} -> {fallback_model}")
        self.original_model = original_model
        self.fallback_model = fallback_model


class APIConnectionError(Exception):
    def __init__(self, message: str = "API connection error"):
        super().__init__(message)


class APITimeoutError(Exception):
    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)


class InvalidAPIKeyError(Exception):
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message)


def parse_prompt_too_long_token_counts(raw_message: str) -> tuple[int | None, int | None]:
    match = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", raw_message, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def is_prompt_too_long_error(error: Exception) -> bool:
    msg = str(error).lower()
    return "prompt is too long" in msg or "prompt_too_long" in msg


def is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, RateLimitError):
        return True
    status = getattr(error, "status", getattr(error, "status_code", None))
    return status == 429


def is_overloaded_error(error: Exception) -> bool:
    if isinstance(error, OverloadedError):
        return True
    status = getattr(error, "status", getattr(error, "status_code", None))
    return status == 529


def is_quota_exhausted(error: Exception) -> bool:
    msg = str(error).lower()
    status = getattr(error, "status", getattr(error, "status_code", None))
    return status == 429 and ("limit: 0" in msg or "exceeded your current quota" in msg)


def is_invalid_api_key(error: Exception) -> bool:
    status = getattr(error, "status", getattr(error, "status_code", None))
    return status == 401


def is_api_timeout_error(error: BaseException) -> bool:
    """Recognize local and provider-SDK timeout wrappers.

    OpenAI-compatible providers surface socket timeouts as
    ``openai.APITimeoutError``. That class inherits the SDK's ``APIError``
    hierarchy rather than Python's built-in :class:`TimeoutError`, so a plain
    ``isinstance(error, TimeoutError)`` check incorrectly treats it as a
    permanent unknown error.

    Match the local compatibility exception directly and provider wrappers by
    class name. The latter keeps this shared module independent of optional
    provider SDK imports while remaining narrower than message matching.
    """

    return isinstance(error, (APITimeoutError, TimeoutError)) or any(
        cls.__name__ == "APITimeoutError" for cls in type(error).__mro__
    )


def is_media_size_error(raw: str) -> bool:
    return (
        ("image exceeds" in raw and "maximum" in raw)
        or ("image dimensions exceed" in raw and "many-image" in raw)
        or bool(re.search(r"maximum of \d+ PDF pages", raw))
    )


def is_image_unsupported_error(raw: str) -> bool:
    """Detect provider errors meaning "this model does not accept image input".

    Distinct from ``is_media_size_error`` (image too large): the model
    has zero image capability, so stripping or resizing won't help —
    history must be sanitized so the request can be re-issued text-only.

    Pattern set covers OpenRouter's real wording plus likely paraphrases
    from other OpenAI-compatible providers. Match is case-insensitive
    because providers paraphrase casing inconsistently.
    """
    low = raw.lower()
    return (
        "no endpoints found that support image" in low
        or "does not support image" in low
        or "doesn't support image" in low
        or "image input is not supported" in low
        or "image_input_not_supported" in low
        or "model does not accept image" in low
    )


def is_transient_upstream_not_found_error(error: Exception) -> bool:
    """Detect proxy/gateway 404s that represent a transient upstream miss.

    A plain 404 usually means a bad model, bad route, or unsupported
    capability and should not be retried. Some OpenAI-compatible gateways
    occasionally wrap an upstream ``NotFoundError`` as a 404
    ``upstream_error`` while the requested endpoint/model is otherwise
    valid; those are safe to send through the normal retry loop.
    """
    status = getattr(error, "status", getattr(error, "status_code", None))
    raw = str(error)
    low = raw.lower()
    if status is None and ("error code: 404" in low or "'code': '404'" in low):
        status = 404
    if status != 404:
        return False
    return (
        "upstream_error" in low
        and ("notfounderror" in low or "not found" in low)
        and ("openaiexception" in low or "upstream" in low)
    )


def is_httpx_transport_error(error: BaseException) -> bool:
    """True when ``error`` is an httpx transport-layer exception.

    ``httpx.RemoteProtocolError`` (the source of the
    ``peer closed connection without sending complete message body
    (incomplete chunked read)`` message), ``ReadError``, ``ConnectError``
    and friends do NOT inherit from ``ConnectionError`` / ``OSError`` —
    they live on the ``httpx.HTTPError`` branch. So the broad
    ``isinstance(error, ConnectionError)`` check in
    :func:`categorize_retryable_api_error` misses them and they fall
    through to the unknown-error bail-out, even though they are
    transient and retryable.
    """
    if httpx is None:
        return False
    return isinstance(error, _HTTPX_TRANSPORT_ERRORS)


def normalize_httpx_transport_error(error: BaseException) -> BaseException:
    """Wrap httpx transport errors in :class:`APIConnectionError` if applicable.

    Provider stream loops encounter raw ``httpx.RemoteProtocolError`` etc.
    when a chunked response is interrupted by a gateway timeout, RST, or
    idle-timeout close. By the time those exceptions reach the retry
    classifier, wrapping them as :class:`APIConnectionError` keeps the
    downstream error-handling story (retry decisions, user-facing
    messages) uniform with the legacy ``ConnectionError`` /
    ``httpx.ConnectError`` paths.

    If the input is not an httpx transport error, it is returned
    unchanged so callers can use this helper as a no-op passthrough.
    """
    if is_httpx_transport_error(error):
        return APIConnectionError(str(error) or "upstream stream interrupted")
    return error


#: Phrases that reliably identify an interrupted chunked HTTP response,
#: independent of which library wrapped the failure. Used as a string
#: fallback when ``isinstance`` cannot classify (e.g. the SDK wrapped
#: the original exception in a custom class).
_TRANSPORT_ERROR_PHRASES: tuple[str, ...] = (
    "peer closed connection",
    "incomplete chunked read",
    "response ended prematurely",
    "connection reset",
    "connection aborted",
    "connection broken",
    "broken pipe",
    "unexpected eof",
)


def is_transport_chunked_read_error(raw: str) -> bool:
    """Substring-based fallback for interrupted chunked HTTP responses."""
    low = raw.lower()
    return any(phrase in low for phrase in _TRANSPORT_ERROR_PHRASES)


@dataclass(frozen=True)
class ErrorClassification:
    retryable: bool
    error_type: str
    message: str


def categorize_retryable_api_error(error: Exception) -> ErrorClassification:
    if is_quota_exhausted(error):
        return ErrorClassification(
            retryable=False,
            error_type="quota_exhausted",
            message="API quota exhausted",
        )

    if is_invalid_api_key(error):
        return ErrorClassification(
            retryable=False,
            error_type="invalid_api_key",
            message="Invalid API key",
        )

    if is_prompt_too_long_error(error):
        return ErrorClassification(
            retryable=False,
            error_type="prompt_too_long",
            message=str(error),
        )

    if is_image_unsupported_error(str(error)):
        # Model-capability rejection — retrying won't help. The query
        # layer tags it ``image_unsupported`` and the engine strips
        # images from history; this classification keeps the retry
        # layer from looping on a permanent failure.
        return ErrorClassification(
            retryable=False,
            error_type="image_unsupported",
            message=str(error),
        )

    if is_transient_upstream_not_found_error(error):
        return ErrorClassification(
            retryable=True,
            error_type="upstream_not_found",
            message="Transient upstream NotFound (404)",
        )

    if is_httpx_transport_error(error):
        return ErrorClassification(
            retryable=True,
            error_type="transport_error",
            message=str(error),
        )

    if is_transport_chunked_read_error(str(error)):
        return ErrorClassification(
            retryable=True,
            error_type="transport_error",
            message=str(error),
        )

    if is_api_timeout_error(error):
        return ErrorClassification(
            retryable=True,
            error_type="timeout",
            message=str(error),
        )

    if is_overloaded_error(error):
        return ErrorClassification(
            retryable=True,
            error_type="overloaded",
            message="API overloaded (529)",
        )

    if is_rate_limit_error(error):
        return ErrorClassification(
            retryable=True,
            error_type="rate_limit",
            message="Rate limited (429)",
        )

    status = getattr(error, "status", getattr(error, "status_code", None))
    if status and status >= 500:
        return ErrorClassification(
            retryable=True,
            error_type="server_error",
            message=f"Server error ({status})",
        )

    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return ErrorClassification(
            retryable=True,
            error_type="connection_error",
            message=str(error),
        )

    return ErrorClassification(
        retryable=False,
        error_type="unknown",
        message=str(error),
    )
