from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

AutomationPhase = Literal["inactive", "active", "paused", "sleeping", "blocked"]
FocusLevel = Literal["full", "medium", "minimal"]


@dataclass(frozen=True)
class AutomationState:
    phase: AutomationPhase
    next_tick_at: float | None = None
    activation_source: str | None = None
    last_sleep_until: float | None = None
    tick_count: int = 0
    blocked_until: float | None = None
    focus: FocusLevel = "medium"
    last_tick_summary: str | None = None
    last_tick_at_ms: float | None = None

    @property
    def is_active(self) -> bool:
        return self.phase in ("active", "sleeping")

    @property
    def is_blocked(self) -> bool:
        return self.phase == "blocked"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
