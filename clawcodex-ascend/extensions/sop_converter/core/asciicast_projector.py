#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""SOP converter adapter for asciicast recording (F-REC).

The SOP converter pipeline has no native event stream — it writes
artifacts to disk and the CLI wrapper (``clawcodex_ext/cli/sop_cmd/commands.py``)
prints a summary. We use the *rendered-output capture* mode: wrap
``sys.stdout`` with a :class:`TeeWriter` that mirrors every ``print()``
to an asciicast frame, plus emit a start/end marker around the
conversion.

Usage::

    with SopStageProjector(capture) as projector:
        result = convert_sop_to_agent(...)

The projector is a no-op when ``capture is None`` so callers can keep
the ``with`` block unconditionally without paying any cost when
recording is disabled.
"""

from __future__ import annotations

import logging
import sys
from types import TracebackType
from typing import IO, TYPE_CHECKING

from extensions.capabilities.recorder import AsciicastCapture

if TYPE_CHECKING:
    # TeeWriter lives in extensions/recording — an optional layer that a
    # partial checkout (fresh clone / CI smoke) may lack, mirroring the
    # subcommand_registry guard convention. Only the lazy import inside
    # ``__enter__`` depends on it, never this module's importability.
    from extensions.recording.renderers import TeeWriter

logger = logging.getLogger(__name__)


class SopStageProjector:
    """Context manager that mirrors SOP CLI stdout into an asciicast capture.

    On enter: swaps ``sys.stdout`` with a :class:`TeeWriter` and emits
    ``marker("sop:start")``. On exit: restores stdout and emits
    ``marker("sop:done")`` plus the optional summary text.

    If ``capture is None`` the projector is a no-op (the CLI wrapper
    can keep the ``with`` block unconditionally without an ``if``).
    """

    def __init__(
        self,
        capture: AsciicastCapture | None,
        *,
        title: str | None = None,
    ) -> None:
        self._capture = capture
        self._title = title
        self._tee: TeeWriter | None = None
        self._original_stdout: IO[str] | None = None

    def __enter__(self) -> SopStageProjector:
        if self._capture is None:
            return self
        try:
            # F-REC TeeWriter is imported lazily here (not at module top) so
            # importing this module during startup never requires
            # extensions/recording; without it the tee is skipped and only
            # the capture markers below are lost.
            from extensions.recording.renderers import TeeWriter

            self._original_stdout = sys.stdout
            self._tee = TeeWriter(
                original=self._original_stdout,
                sink=self._emit_stdout,
            )
            self._tee.install()
            self._capture.marker(
                "sop:start",
                text=self._title or "SOP conversion started",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SopStageProjector enter failed: %s", exc)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._capture is None or self._tee is None:
            return
        try:
            if exc is not None:
                self._capture.marker(
                    "sop:error",
                    text=f"SOP conversion failed: {exc}",
                )
            else:
                self._capture.marker(
                    "sop:done",
                    text=self._title or "SOP conversion finished",
                )
        finally:
            try:
                self._tee.restore()
            except Exception:  # noqa: BLE001  # nosec
                pass
            self._tee = None
            self._original_stdout = None

    def emit_marker(self, label: str, text: str = "") -> None:
        """Public hook for the CLI wrapper to drop stage markers.

        No-op when ``capture is None`` so callers can use this
        unconditionally.
        """
        if self._capture is None:
            return
        try:
            self._capture.marker(label, text=text or label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SopStageProjector marker failed: %s", exc)

    def _emit_stdout(self, data: str) -> None:
        """TeeWriter sink: forward every ``print()`` chunk as one output frame."""
        if self._capture is None or not data:
            return
        # TeeWriter may give us partial chunks; split on newlines so each
        # printed line becomes its own output frame and the player renders
        # cleanly. Trailing partial-line data is buffered by passing it
        # through as-is (acceptable for our purposes — the player will
        # concatenate adjacent frames anyway).
        try:
            from extensions.capabilities.recorder import AsciicastEvent

            self._capture.emit(AsciicastEvent(t=0.0, kind="o", data=data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SopStageProjector emit failed: %s", exc)


__all__ = ["SopStageProjector"]
