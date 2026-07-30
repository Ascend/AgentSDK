"""Computer Use data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MouseButton(str, Enum):
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class ScrollDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class ScreenRegion:
    x: int = 0
    y: int = 0
    width: int = 1920
    height: int = 1080

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ScreenRegion width/height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("ScreenRegion origin must be non-negative")
        if self.x + self.width > 32767 or self.y + self.height > 32767:
            raise ValueError("ScreenRegion extends past supported bounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreenRegion:
        if not isinstance(data, dict):
            raise TypeError("ScreenRegion data must be a JSON object")
        return cls(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
        )


@dataclass(frozen=True)
class WindowRef:
    title: str
    pid: int | None = None
    window_id: str | None = None

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("WindowRef title must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "pid": self.pid, "window_id": self.window_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowRef:
        if not isinstance(data, dict):
            raise TypeError("WindowRef data must be a JSON object")
        title = str(data.get("title", "")).strip()
        if not title:
            raise ValueError("WindowRef title must be non-empty")
        pid = data.get("pid")
        window_id = data.get("window_id")
        return cls(
            title=title,
            pid=int(pid) if pid is not None else None,
            window_id=str(window_id) if window_id is not None else None,
        )


@dataclass(frozen=True)
class InputAction:
    """A single input event recorded by the dry-run recorder or replayed later."""

    kind: str
    args: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.kind or not isinstance(self.kind, str):
            raise ValueError("InputAction.kind must be a non-empty string")
        if not isinstance(self.args, dict):
            raise TypeError("InputAction.args must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "args": dict(self.args)}
