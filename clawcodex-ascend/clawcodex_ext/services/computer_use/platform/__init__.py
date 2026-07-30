"""Platform-specific Computer Use backends.

The real Linux backend is the only shipped implementation. macOS and Windows
stubs are intentionally absent; ``build_provider_suite`` falls back to the
null suite on unsupported platforms.
"""

from __future__ import annotations

from ..exceptions import ComputerUseError
from .linux import (
    ALLOW_ENV_VAR,
    LinuxBackend,
    LinuxClipboardManager,
    LinuxInputSimulator,
    LinuxScreenshotProvider,
    LinuxWindowManager,
    build_linux_suite,
    default_linux_backend,
)
from .null import (
    NullClipboardManager,
    NullInputSimulator,
    NullScreenshotProvider,
    NullWindowManager,
    build_null_suite,
)

__all__ = [
    "ALLOW_ENV_VAR",
    "LinuxBackend",
    "LinuxClipboardManager",
    "LinuxInputSimulator",
    "LinuxScreenshotProvider",
    "LinuxWindowManager",
    "NullClipboardManager",
    "NullInputSimulator",
    "NullScreenshotProvider",
    "NullWindowManager",
    "build_linux_suite",
    "build_null_suite",
    "default_linux_backend",
]


_UNSUPPORTED_PLATFORMS: frozenset[str] = frozenset()


def _current_platform() -> str:
    import sys

    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in {"win32", "cygwin"}:
        return "windows"
    return sys.platform or "unknown"


def build_provider_suite(
    platform: str | None = None,
    *,
    backend: LinuxBackend | None = None,
    recorder=None,
) -> dict[str, object]:
    """Return a dict of provider instances for the requested platform.

    The default platform is the runtime platform. Unsupported platforms return
    the null suite so callers can still wire Computer Use tools without a
    hard crash.
    """
    name = (platform or _current_platform()).lower()
    if name == "linux":
        return build_linux_suite(backend=backend, recorder=recorder)
    if name in _UNSUPPORTED_PLATFORMS:
        raise ComputerUseError(f"platform {name!r} is explicitly disabled")
    return build_null_suite()
