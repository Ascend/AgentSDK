"""Package version.

Synced with ``src.__version__`` at module load time. The value here is a
fallback for environments where ``src`` is not importable (e.g. isolated
packaging builds).
"""

from __future__ import annotations

from typing import Final

_FALLBACK_VERSION: Final[str] = "0.0.0+unknown"


def _resolve_version() -> str:
    try:
        from src import __version__ as src_version  # type: ignore[import-not-found]

        if isinstance(src_version, str) and src_version:
            return src_version
    except Exception:  # nosec B110
        # Isolated package builds intentionally fall back to a local version.
        pass
    return _FALLBACK_VERSION


__version__: Final[str] = _resolve_version()
