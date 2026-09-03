"""Self-contained asciicast v2 validator.

CI does not have the asciinema CLI available, and shell-out validation
would force every developer and CI runner to install Rust. This module
implements the v2 schema check in pure Python (zero dependencies) so
``validate_cast`` can run anywhere.

Usage::

    from extensions.recording.validate_cast import validate_cast

    errors = validate_cast(Path("demo.cast"))
    if errors:
        for err in errors:
            print(err)
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["validate_cast"]

# Required header fields per asciicast v2 spec
# (https://docs.asciinema.org/manual/asciicast/v2/).
_REQUIRED_HEADER = {"version", "width", "height"}
_VALID_VERSION = 2
# ``o`` output, ``i`` input, ``m`` marker, ``r`` resize are in the core
# spec; ``x`` (exit code) is an optional extension event supported by
# both asciinema and our native PTY recorder.
_VALID_EVENT_KINDS = {"o", "i", "m", "r", "x"}


def validate_cast(path: Path) -> list[str]:
    """Validate ``path`` as an asciicast v2 file.

    Returns a list of human-readable error strings. An empty list means
    the file is schema-conformant. The function never raises — invalid
    JSON / missing files surface as error messages so callers can
    decide whether to surface, warn, or fail.
    """
    path = Path(path)
    if not path.exists():
        return [f"file not found: {path}"]

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read {path}: {exc}"]

    lines = text.splitlines()
    if not lines:
        return [f"empty file: {path}"]

    errors: list[str] = []
    # --- header ---
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return [f"line 1: header is not valid JSON: {exc}"]

    if not isinstance(header, dict):
        return ["line 1: header must be a JSON object"]

    missing = _REQUIRED_HEADER - set(header.keys())
    if missing:
        errors.append(f"line 1: header missing required fields: {sorted(missing)}")

    if header.get("version") != _VALID_VERSION:
        errors.append(f"line 1: header.version must be {_VALID_VERSION}, got {header.get('version')!r}")

    width = header.get("width")
    height = header.get("height")
    if not isinstance(width, int) or width <= 0:
        errors.append(f"line 1: header.width must be a positive int, got {width!r}")
    if not isinstance(height, int) or height <= 0:
        errors.append(f"line 1: header.height must be a positive int, got {height!r}")

    # --- events ---
    for idx, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {idx}: invalid JSON: {exc}")
            continue
        if (
            not isinstance(event, list)
            or len(event) != 3
            or not isinstance(event[0], (int, float))
            or not isinstance(event[1], str)
            or not isinstance(event[2], (str, int))
        ):
            errors.append(
                f"line {idx}: event must be [time:number, code:str, data:str|int], "
                f"got {type(event).__name__}: {raw[:80]!r}"
            )
            continue
        kind = event[1]
        if kind not in _VALID_EVENT_KINDS:
            errors.append(f"line {idx}: event code must be one of {sorted(_VALID_EVENT_KINDS)}, got {kind!r}")
        if event[0] < 0:
            errors.append(f"line {idx}: event time must be >= 0, got {event[0]}")
        # ``x`` (exit code) data must be an integer.
        if kind == "x" and not isinstance(event[2], int):
            errors.append(f"line {idx}: 'x' event data must be an int exit code, got {event[2]!r}")

    return errors
