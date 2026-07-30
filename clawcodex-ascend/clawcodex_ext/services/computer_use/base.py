"""Abstract base classes for Computer Use providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import MouseButton, ScreenRegion, WindowRef


class ScreenshotProvider(ABC):
    """Cross-platform screenshot interface. All methods return PNG bytes."""

    @abstractmethod
    def capture_fullscreen(self) -> bytes: ...

    @abstractmethod
    def capture_region(self, region: ScreenRegion) -> bytes: ...

    @abstractmethod
    def capture_window(self, window: WindowRef) -> bytes | None: ...

    @property
    @abstractmethod
    def is_dry_run(self) -> bool: ...


class InputSimulator(ABC):
    """Cross-platform keyboard/mouse simulation. Coordinates are absolute."""

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None: ...

    @abstractmethod
    def click(
        self,
        button: MouseButton = MouseButton.LEFT,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> None: ...

    @abstractmethod
    def double_click(self, *, x: int | None = None, y: int | None = None) -> None: ...

    @abstractmethod
    def type_text(self, text: str) -> None: ...

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Send a single keystroke, e.g. 'enter', 'escape', 'ctrl+c'."""

    @abstractmethod
    def scroll(self, dx: int = 0, dy: int = 1) -> None: ...

    @abstractmethod
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None: ...

    @property
    @abstractmethod
    def is_dry_run(self) -> bool: ...


class ClipboardManager(ABC):
    """Cross-platform text clipboard interface."""

    @abstractmethod
    def get_text(self) -> str: ...

    @abstractmethod
    def set_text(self, text: str) -> None: ...

    @property
    @abstractmethod
    def is_dry_run(self) -> bool: ...


class WindowManager(ABC):
    """Cross-platform window manager interface."""

    @abstractmethod
    def list_windows(self) -> list[WindowRef]: ...

    @abstractmethod
    def focus_window(self, window: WindowRef) -> bool: ...

    @abstractmethod
    def close_window(self, window: WindowRef) -> bool: ...

    @property
    @abstractmethod
    def is_dry_run(self) -> bool: ...
