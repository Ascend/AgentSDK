"""``clawcodex cast-to-mp4`` CLI subcommand.

Thin wrapper that exposes :func:`extensions.recording.tools.cast_to_mp4.
run_cast_to_mp4_command` under :mod:`clawcodex_ext.cli.subcommand_registry`.

Layer rule (CLAUDE.md): this module sits in :mod:`extensions.recording`
(Layer 2) because it composes the converter living in
:mod:`extensions.recording.tools.cast_to_mp4` with the public CLI
subcommand registry.  Splitting converter + CLI into two modules lets
the converter be imported directly (e.g. from a future ``jupytext``
notebook) without paying for :mod:`subcommand_registry` cost.

Usage::

    clawcodex cast-to-mp4 --cast demo.cast --out demo.mp4 --fps 2
    clawcodex cast-to-mp4 --cast demo.cast --out demo.mp4 --keep-pngs
"""

from __future__ import annotations

from clawcodex_ext.cli.subcommand_registry import register

from extensions.recording.tools.cast_to_mp4 import (
    build_cast_to_mp4_parser,
    run_cast_to_mp4_command,
)

__all__ = ["build_cast_to_mp4_parser", "run_cast_to_mp4_command"]


@register("cast-to-mp4")
def _cast_to_mp4_subcommand(args: list[str]) -> int:
    """``clawcodex cast-to-mp4 ...`` handler."""
    return run_cast_to_mp4_command(args)
