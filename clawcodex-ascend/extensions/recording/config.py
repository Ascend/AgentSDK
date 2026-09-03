"""Recording subsystem configuration.

Holds the runtime-tunable knobs for the asciicast recorder. Per the
recording plan, defaults leave the recorder fully disabled so existing
runs are unaffected. Operators opt in by setting the config key (or by
invoking ``clawcodex record`` directly, which builds its own writer
without touching this module).

The dataclass lives here (rather than in ``src/config.py``) to keep
the recorder's footprint inside ``extensions/`` — CLAUDE.md's
decoupling mandate forbids adding feature-development modules to
``src/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = ["RecordingConfig"]


@dataclass
class RecordingConfig:
    """User-tunable recording settings.

    Attributes:
        enabled: Master switch. ``False`` (default) keeps every
            subsystem on its non-recording path; no per-frame cost.
        output_dir: Where ``clawcodex record`` writes ``.cast`` files
            when the user does not pass ``--out`` explicitly.
        sources: Default source list when ``--sources`` is omitted
            on the CLI. Empty list means the CLI must be given sources
            explicitly.
        flush_mode: ``"per_frame"`` flushes after every event (matches
            the recording plan decision; enables ``tail -f``); periodic
            modes are reserved for future tuning and not yet
            implemented.
        default_width / default_height: Terminal size used when the
            capture does not inherit one from a real TTY (e.g.
            headless orchestrator runs).
    """

    enabled: bool = False
    output_dir: Path = field(default_factory=lambda: Path(".reports/casts"))
    sources: list[str] = field(default_factory=list)
    flush_mode: Literal["per_frame"] = "per_frame"
    default_width: int = 120
    default_height: int = 36
