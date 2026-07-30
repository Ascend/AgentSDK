"""Null Computer Use providers for tests and unsupported platforms.

These providers never touch the real desktop, never spawn subprocesses, and
record their actions in a ``DryRunRecorder`` so tests can assert what *would*
have happened. They are the safest default for any environment that has not
explicitly opted in to real input simulation.
"""

from __future__ import annotations

from typing import Any

from ..base import ClipboardManager, InputSimulator, ScreenshotProvider, WindowManager
from ..dry_run import DryRunRecorder
from ..models import MouseButton, ScreenRegion, WindowRef

PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class NullScreenshotProvider(ScreenshotProvider):
    def __init__(self, recorder: DryRunRecorder | None = None) -> None:
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return True

    def capture_fullscreen(self) -> bytes:
        self._recorder.record_screenshot("fullscreen", PNG_1x1)
        return PNG_1x1

    def capture_region(self, region: ScreenRegion) -> bytes:
        self._recorder.record_screenshot("region", PNG_1x1, **region.to_dict())
        return PNG_1x1

    def capture_window(self, window: WindowRef) -> bytes | None:  # pylint: disable=useless-return
        self._recorder.record_screenshot("window", None, **window.to_dict())
        return


class NullInputSimulator(InputSimulator):
    def __init__(self, recorder: DryRunRecorder | None = None) -> None:
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return True

    def move_mouse(self, x: int, y: int) -> None:
        self._validate_xy(x, y)
        self._recorder.record_action("move_mouse", x=x, y=y)

    def click(
        self,
        button: MouseButton = MouseButton.LEFT,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        if x is not None and y is not None:
            self._validate_xy(x, y)
        self._recorder.record_action("click", button=button.value, x=x, y=y)

    def double_click(self, *, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self._validate_xy(x, y)
        self._recorder.record_action("double_click", x=x, y=y)

    def type_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._recorder.record_action("type_text", text=text)

    def press_key(self, key: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        self._recorder.record_action("press_key", key=key)

    def scroll(self, dx: int = 0, dy: int = 1) -> None:
        self._recorder.record_action("scroll", dx=dx, dy=dy)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self._validate_xy(start_x, start_y)
        self._validate_xy(end_x, end_y)
        self._recorder.record_action(
            "drag",
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )

    @staticmethod
    def _validate_xy(x: int, y: int) -> None:
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("x and y must be integers")
        if x < 0 or y < 0 or x > 32767 or y > 32767:
            raise ValueError("coordinates out of supported bounds")


class NullClipboardManager(ClipboardManager):
    def __init__(self, recorder: DryRunRecorder | None = None) -> None:
        self._recorder = recorder or DryRunRecorder()
        self._text = ""

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return True

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._text = text
        self._recorder.record_action("clipboard_set", length=len(text))


class NullWindowManager(WindowManager):
    def __init__(self, recorder: DryRunRecorder | None = None) -> None:
        self._recorder = recorder or DryRunRecorder()

    @property
    def recorder(self) -> DryRunRecorder:
        return self._recorder

    @property
    def is_dry_run(self) -> bool:
        return True

    def list_windows(self) -> list[WindowRef]:
        return []

    def focus_window(self, window: WindowRef) -> bool:
        self._recorder.record_action("focus_window", **window.to_dict())
        return False

    def close_window(self, window: WindowRef) -> bool:
        self._recorder.record_action("close_window", **window.to_dict())
        return False


def build_null_suite(recorder: DryRunRecorder | None = None) -> dict[str, Any]:
    recorder = recorder or DryRunRecorder()
    return {
        "recorder": recorder,
        "screenshot": NullScreenshotProvider(recorder),
        "input": NullInputSimulator(recorder),
        "clipboard": NullClipboardManager(recorder),
        "window": NullWindowManager(recorder),
    }
