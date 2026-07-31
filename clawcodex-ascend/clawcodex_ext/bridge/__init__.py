"""Downstream ClawCodex bridge implementations.

Phase 4 P1 (Option A) migrated bridge modules here from ``src/bridge/``.
The per-module ``src/bridge/X.py`` files are thin ``sys.modules`` swap
facades that forward to the corresponding ``clawcodex_ext.bridge.X``
implementations. This ``__init__.py`` intentionally stays minimal to avoid
circular imports with those facades.
"""

from __future__ import annotations
