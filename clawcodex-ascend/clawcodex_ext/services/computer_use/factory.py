"""Factory entry point that builds a Computer Use provider suite.

The factory is intentionally a thin shim. It does *not* import the
``clawcodex_ext.tool_system`` Tool type so that this package can be used in
unit tests and in environments where the Tool factory is not available. The
deferred integration is documented in the FEATURE_PLAN §7.2 and will be wired
in a later iteration.
"""

from __future__ import annotations

from typing import Any

from .base import ClipboardManager, InputSimulator, ScreenshotProvider, WindowManager
from .dry_run import DryRunRecorder
from .platform import build_provider_suite

ComputerUseSuite = dict[str, Any]


def build_computer_use_suite(
    platform: str | None = None,
    *,
    recorder: DryRunRecorder | None = None,
    backend: Any = None,
) -> ComputerUseSuite:
    shared_recorder = recorder or DryRunRecorder()
    suite = build_provider_suite(platform=platform, backend=backend, recorder=shared_recorder)
    suite["recorder"] = shared_recorder
    # Lightweight sanity check: every provider must implement the right ABC.
    screenshot = suite.get("screenshot")
    input_sim = suite.get("input")
    clipboard = suite.get("clipboard")
    window = suite.get("window")
    for provider, expected in [
        (screenshot, ScreenshotProvider),
        (input_sim, InputSimulator),
        (clipboard, ClipboardManager),
        (window, WindowManager),
    ]:
        if provider is not None and not isinstance(provider, expected):
            raise TypeError(f"provider for {expected.__name__} must be a {expected.__name__}")
    return suite


__all__ = ["ComputerUseSuite", "build_computer_use_suite"]
