"""Computer Use service primitives (F-61 first iteration).

This package ships the cross-platform ABCs, a safety-gated Linux backend, a
null backend for tests and unsupported platforms, and a thin factory used by
later Tool integration work. See ``docs/FEATURE_PLAN.md`` §7.2 for the full
F-61 scope; macOS / Windows backends, the real ``build_computer_use_tools``
Tool factory, and the consent modal are explicitly deferred to later
iterations.
"""

from __future__ import annotations

from .base import ClipboardManager, InputSimulator, ScreenshotProvider, WindowManager
from .dry_run import DryRunRecorder
from .exceptions import (
    BinaryNotFoundError,
    ComputerUseError,
    CoordinatesOutOfBoundsError,
    SafetyViolationError,
    WindowNotFoundError,
)
from .factory import ComputerUseSuite, build_computer_use_suite
from .models import InputAction, MouseButton, ScreenRegion, ScrollDirection, WindowRef
from .platform import (
    ALLOW_ENV_VAR,
    LinuxBackend,
    LinuxClipboardManager,
    LinuxInputSimulator,
    LinuxScreenshotProvider,
    LinuxWindowManager,
    NullClipboardManager,
    NullInputSimulator,
    NullScreenshotProvider,
    NullWindowManager,
    build_linux_suite,
    build_null_suite,
    build_provider_suite,
    default_linux_backend,
)

__all__ = [
    "ALLOW_ENV_VAR",
    "BinaryNotFoundError",
    "ClipboardManager",
    "ComputerUseError",
    "ComputerUseSuite",
    "CoordinatesOutOfBoundsError",
    "DryRunRecorder",
    "InputAction",
    "InputSimulator",
    "LinuxBackend",
    "LinuxClipboardManager",
    "LinuxInputSimulator",
    "LinuxScreenshotProvider",
    "LinuxWindowManager",
    "MouseButton",
    "NullClipboardManager",
    "NullInputSimulator",
    "NullScreenshotProvider",
    "NullWindowManager",
    "SafetyViolationError",
    "ScreenRegion",
    "ScreenshotProvider",
    "ScrollDirection",
    "WindowManager",
    "WindowNotFoundError",
    "WindowRef",
    "build_computer_use_suite",
    "build_linux_suite",
    "build_null_suite",
    "build_provider_suite",
    "default_linux_backend",
]
