"""Linux Computer Use backend (scrot + xdotool).

Safety properties (these are the entire reason this module exists in this shape):

* Subprocesses are invoked as argv lists only — never ``shell=True`` — so
  user input cannot inject shell metacharacters.
* Every call passes an explicit ``timeout`` and never relies on the
  subprocess hanging.
* All public methods are dry-run by default. The real ``subprocess`` runner is
  only reached when an instance is constructed with ``dry_run=False`` AND the
  environment variable ``CLAWCODEX_COMPUTER_USE_ALLOW`` is set to a truthy
  value. The default environment therefore cannot move a real mouse or type
  into a real terminal.
* Coordinates are bounded-validated before any subprocess is launched.
* Missing system binaries raise :class:`BinaryNotFoundError` instead of
  silently producing a zero-byte screenshot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import (
    ClipboardManager,
    InputSimulator,
    ScreenshotProvider,
    WindowManager,
)
from ..dry_run import DryRunRecorder
from ..exceptions import (
    BinaryNotFoundError,
    CoordinatesOutOfBoundsError,
    SafetyViolationError,
    WindowNotFoundError,
)
from ..models import MouseButton, ScreenRegion, WindowRef

RunFn = Callable[..., subprocess.CompletedProcess[bytes]]

ALLOW_ENV_VAR = "CLAWCODEX_COMPUTER_USE_ALLOW"

MAX_COORD = 32767


@dataclass
class LinuxBackend:
    """Configuration for the Linux backend.

    Tests can override ``runner`` to avoid touching the real filesystem or
    spawning real processes. The default runner is :func:`subprocess.run`.
    """

    screenshot_binary: str = "scrot"
    import_binary: str = "import"
    xdotool_binary: str = "xdotool"
    xclip_binary: str = "xclip"
    wmctrl_binary: str = "wmctrl"
    timeout_seconds: float = 5.0
    dry_run: bool = True
    allowed: bool = False
    runner: RunFn = field(default=subprocess.run)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def is_allowed(self) -> bool:
        return self.allowed and not self.dry_run

    def require_allowed(self) -> None:
        if not self.is_allowed():
            raise SafetyViolationError(
                f"Real Linux Computer Use is disabled. Set {ALLOW_ENV_VAR}=1 and pass dry_run=False to enable."
            )


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def default_linux_backend() -> LinuxBackend:
    return LinuxBackend(
        dry_run=not _truthy_env(ALLOW_ENV_VAR),
        allowed=_truthy_env(ALLOW_ENV_VAR),
    )


def _validate_xy(x: int, y: int) -> None:
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("x and y must be integers")
    if x < 0 or y < 0 or x > MAX_COORD or y > MAX_COORD:
        raise CoordinatesOutOfBoundsError(f"coordinates ({x}, {y}) out of supported bounds")


def _check_binary(runner: RunFn, binary: str, *, timeout: float) -> None:
    if shutil.which(binary) is None:
        raise BinaryNotFoundError(f"required binary not found: {binary}")
    # ``which`` already verified presence, but a short ``--version`` probe
    # surfaces obviously broken installs (e.g. permission denied) early.
    try:
        runner(
            [binary, "--version"],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BinaryNotFoundError(f"binary {binary!r} is not usable: {exc}") from exc


class LinuxScreenshotProvider(ScreenshotProvider):
    def __init__(
        self,
        backend: LinuxBackend | None = None,
        recorder: DryRunRecorder | None = None,
    ) -> None:
        self._backend = backend or default_linux_backend()
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return self._backend.dry_run

    def capture_fullscreen(self) -> bytes:
        if self.is_dry_run:
            self._recorder.record_screenshot("fullscreen", None)
            return b""
        self._backend.require_allowed()
        _check_binary(
            self._backend.runner,
            self._backend.screenshot_binary,
            timeout=self._backend.timeout_seconds,
        )
        result = self._backend.runner(
            [self._backend.screenshot_binary, "-o", "-"],
            capture_output=True,
            check=True,
            timeout=self._backend.timeout_seconds,
        )
        return result.stdout

    def capture_region(self, region: ScreenRegion) -> bytes:
        if self.is_dry_run:
            self._recorder.record_screenshot("region", None, **region.to_dict())
            return b""
        self._backend.require_allowed()
        _check_binary(
            self._backend.runner,
            self._backend.screenshot_binary,
            timeout=self._backend.timeout_seconds,
        )
        geom = f"{region.x},{region.y},{region.width},{region.height}"
        result = self._backend.runner(
            [self._backend.screenshot_binary, "-o", "-a", geom, "-"],
            capture_output=True,
            check=True,
            timeout=self._backend.timeout_seconds,
        )
        return result.stdout

    def capture_window(self, window: WindowRef) -> bytes | None:
        if self.is_dry_run:
            self._recorder.record_screenshot("window", None, **window.to_dict())
            return None
        self._backend.require_allowed()
        _check_binary(
            self._backend.runner,
            self._backend.xdotool_binary,
            timeout=self._backend.timeout_seconds,
        )
        _check_binary(self._backend.runner, self._backend.import_binary, timeout=self._backend.timeout_seconds)
        result = self._backend.runner(
            [self._backend.xdotool_binary, "search", "--name", window.title],
            capture_output=True,
            check=False,
            timeout=self._backend.timeout_seconds,
        )
        wid = result.stdout.decode("utf-8", errors="replace").strip()
        if not wid:
            raise WindowNotFoundError(f"no window matching {window.title!r}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._backend.runner(
                [self._backend.import_binary, "-window", wid, str(tmp_path)],
                capture_output=True,
                check=True,
                timeout=self._backend.timeout_seconds,
            )
            return tmp_path.read_bytes()
        finally:
            with __import__("contextlib").suppress(FileNotFoundError):
                tmp_path.unlink()


_BUTTON_TO_XDOTOOL = {
    MouseButton.LEFT: "1",
    MouseButton.MIDDLE: "2",
    MouseButton.RIGHT: "3",
}


class LinuxInputSimulator(InputSimulator):
    def __init__(
        self,
        backend: LinuxBackend | None = None,
        recorder: DryRunRecorder | None = None,
    ) -> None:
        self._backend = backend or default_linux_backend()
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return self._backend.dry_run

    def _xdotool(self, args: Iterable[str], *, check: bool = True) -> None:
        if self.is_dry_run:
            self._recorder.record_action("xdotool", args=list(args))
            return
        self._backend.require_allowed()
        _check_binary(
            self._backend.runner,
            self._backend.xdotool_binary,
            timeout=self._backend.timeout_seconds,
        )
        self._backend.runner(
            [self._backend.xdotool_binary, *args],
            capture_output=True,
            check=check,
            timeout=self._backend.timeout_seconds,
        )

    def move_mouse(self, x: int, y: int) -> None:
        _validate_xy(x, y)
        self._xdotool(["mousemove", str(x), str(y)])

    def click(
        self,
        button: MouseButton = MouseButton.LEFT,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        if x is not None and y is not None:
            _validate_xy(x, y)
            self.move_mouse(x, y)
        self._xdotool(["click", _BUTTON_TO_XDOTOOL[button]])

    def double_click(self, *, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            _validate_xy(x, y)
            self.move_mouse(x, y)
        self._xdotool(["click", "--repeat", "2", "1"])

    def type_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # ``--`` prevents the typed string from being interpreted as xdotool
        # flags, even when it starts with ``-``.
        self._xdotool(["type", "--", text])

    def press_key(self, key: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        # Disallow whitespace inside the key name to avoid argv smuggling.
        if any(ch.isspace() for ch in key):
            raise ValueError("key must not contain whitespace")
        self._xdotool(["key", key])

    def scroll(self, dx: int = 0, dy: int = 1) -> None:
        if not isinstance(dx, int) or not isinstance(dy, int):
            raise TypeError("dx and dy must be integers")
        if dx == 0 and dy == 0:
            return
        if dy != 0:
            self._xdotool(["click", "5" if dy > 0 else "4"])
        if dx != 0:
            self._xdotool(["click", "7" if dx > 0 else "6"])

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        _validate_xy(start_x, start_y)
        _validate_xy(end_x, end_y)
        self._xdotool(["mousemove", str(start_x), str(start_y)])
        self._xdotool(["mousedown", "1"])
        self._xdotool(["mousemove", str(end_x), str(end_y)])
        self._xdotool(["mouseup", "1"])


class LinuxClipboardManager(ClipboardManager):
    def __init__(
        self,
        backend: LinuxBackend | None = None,
        recorder: DryRunRecorder | None = None,
    ) -> None:
        self._backend = backend or default_linux_backend()
        self._recorder = recorder or DryRunRecorder()
        self._text = ""

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return self._backend.dry_run

    def get_text(self) -> str:
        if self.is_dry_run:
            return self._text
        self._backend.require_allowed()
        _check_binary(self._backend.runner, self._backend.xclip_binary, timeout=self._backend.timeout_seconds)
        result = self._backend.runner(
            [self._backend.xclip_binary, "-selection", "clipboard", "-out"],
            capture_output=True,
            check=True,
            timeout=self._backend.timeout_seconds,
        )
        return result.stdout.decode("utf-8", errors="replace")

    def set_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.is_dry_run:
            self._text = text
            self._recorder.record_action("clipboard_set", length=len(text))
            return
        self._backend.require_allowed()
        _check_binary(self._backend.runner, self._backend.xclip_binary, timeout=self._backend.timeout_seconds)
        self._backend.runner(
            [self._backend.xclip_binary, "-selection", "clipboard"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=self._backend.timeout_seconds,
        )


class LinuxWindowManager(WindowManager):
    def __init__(
        self,
        backend: LinuxBackend | None = None,
        recorder: DryRunRecorder | None = None,
    ) -> None:
        self._backend = backend or default_linux_backend()
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return self._backend.dry_run

    def list_windows(self) -> list[WindowRef]:
        if self.is_dry_run:
            return []
        self._backend.require_allowed()
        _check_binary(self._backend.runner, self._backend.wmctrl_binary, timeout=self._backend.timeout_seconds)
        result = self._backend.runner(
            [self._backend.wmctrl_binary, "-l"],
            capture_output=True,
            check=True,
            timeout=self._backend.timeout_seconds,
        )
        windows: list[WindowRef] = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            window_id, _desktop, _host, title = parts
            try:
                windows.append(WindowRef(title=title.strip(), window_id=window_id))
            except ValueError:
                continue
        return windows

    def focus_window(self, window: WindowRef) -> bool:
        if self.is_dry_run:
            self._recorder.record_action("focus_window", **window.to_dict())
            return False
        self._backend.require_allowed()
        _check_binary(self._backend.runner, self._backend.wmctrl_binary, timeout=self._backend.timeout_seconds)
        result = self._backend.runner(
            [self._backend.wmctrl_binary, "-a", window.title],
            capture_output=True,
            check=False,
            timeout=self._backend.timeout_seconds,
        )
        return result.returncode == 0

    def close_window(self, window: WindowRef) -> bool:
        if self.is_dry_run:
            self._recorder.record_action("close_window", **window.to_dict())
            return False
        self._backend.require_allowed()
        _check_binary(self._backend.runner, self._backend.wmctrl_binary, timeout=self._backend.timeout_seconds)
        result = self._backend.runner(
            [self._backend.wmctrl_binary, "-c", window.title],
            capture_output=True,
            check=False,
            timeout=self._backend.timeout_seconds,
        )
        return result.returncode == 0


def build_linux_suite(
    backend: LinuxBackend | None = None,
    recorder: DryRunRecorder | None = None,
) -> dict[str, Any]:
    backend = backend or default_linux_backend()
    recorder = recorder or DryRunRecorder()
    return {
        "recorder": recorder,
        "screenshot": LinuxScreenshotProvider(backend, recorder),
        "input": LinuxInputSimulator(backend, recorder),
        "clipboard": LinuxClipboardManager(backend, recorder),
        "window": LinuxWindowManager(backend, recorder),
    }
