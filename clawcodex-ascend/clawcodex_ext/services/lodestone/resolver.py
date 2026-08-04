#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""F-97 LODESTONE — resolver.

Maps a ``LodestoneAnchor`` + ``AnchorContext`` to a ``RenderedAnchor``:

* ``resolve(anchor, ctx)``  — one anchor.
* ``resolve_text(text, ctx)`` — scan + render every anchor in ``text``.

Responsibilities:

1.  Pick the best :class:`AnchorTarget` via environment probes, user
    preferences and ``AnchorTargetRegistry.pick``.
2.  Reject paths that escape ``workspace_root`` (path-traversal guard
    in :func:`_guard_path`).
3.  Build the substitution context for the target template and render.
4.  Honour the ``LODESTONE=off`` kill-switch and per-kind
    ``disabled_kinds`` opt-outs.
"""

# pylint: disable=cyclic-import
from __future__ import annotations

import os
from pathlib import Path

from .models import (
    AnchorContext,
    AnchorTarget,
    AnchorTargetRegistry,
    LodestoneAnchor,
    RenderedAnchor,
    Sink,
)
from .renderer import AnchorRenderer

# Editor detection ladder used to override ``default_editor`` when the
# user has not configured one explicitly.  We probe ``os.environ`` once
# per ``resolve_text`` call (cheap enough) and pass the result through
# the registry's ``pick``.


class AnchorResolver:
    """Stateless resolver. Construct once and call ``resolve`` many times."""

    def __init__(
        self,
        registry: AnchorTargetRegistry,
        renderer: AnchorRenderer | None = None,
    ) -> None:
        self._registry = registry
        self._renderer = renderer or AnchorRenderer()

    # -- single-anchor API ---------------------------------------------------

    def resolve(
        self,
        anchor: LodestoneAnchor,
        *,
        ctx: AnchorContext,
        target_override: str | None = None,
        sink: Sink | None = None,
    ) -> RenderedAnchor:
        """Pick a target + render ``anchor`` in the requested ``sink``."""
        # 1. Disabled feature → plain text
        if not ctx.config.enabled:
            return RenderedAnchor(
                anchor=anchor,
                target=None,
                link_text=anchor.raw,
                rendered=anchor.raw,
                is_anchor=False,
                fallback_reason="lodestone disabled",
                sink="text",
            )

        # 2. Per-kind disable
        if anchor.kind in ctx.config.disabled_kinds:
            return self._fallback(anchor, sink or "text", f"kind {anchor.kind} disabled")

        # 3. Path-traversal guard
        path_ok, path_reason = _guard_path(anchor, ctx)
        if not path_ok:
            return self._fallback(anchor, sink or "text", path_reason or "path unsafe")

        # 4. Pick a target
        target = self._select_target(anchor, ctx, override=target_override)
        sink = self._renderer.normalize_sink(sink or ctx.config.renderer, env=ctx.env)

        # 5. Render
        return self._renderer.render(anchor, target=target, sink=sink, ctx=ctx)

    def resolve_text(
        self,
        text: str,
        *,
        ctx: AnchorContext,
        sink: Sink | None = None,
    ) -> str:
        """Parse ``text``, render every anchor, return the merged string."""
        if not text:
            return text or ""
        sink = self._renderer.normalize_sink(sink or ctx.config.renderer, env=ctx.env)
        # The renderer keeps plain text unchanged between anchors via
        # ``render_text`` — works for all three sinks.
        return self._renderer.render_text(text, ctx=ctx, sink=sink)

    # -- internals -----------------------------------------------------------

    def _select_target(
        self,
        anchor: LodestoneAnchor,
        ctx: AnchorContext,
        *,
        override: str | None = None,
    ) -> AnchorTarget | None:
        if override:
            t = self._registry.get(override)
            if t is not None and t.kind == anchor.kind:
                return t
            return None
        # URL anchors are themselves targets — return None so the renderer
        # emits them as-is.
        if anchor.kind == "url":
            return None
        # Tracker anchors: prefer the configured tracker host.
        if anchor.kind == "tracker_issue" and ctx.config.default_tracker_repo:
            host = ctx.config.default_tracker_host
            target_id = f"tracker:{host}"
            target = self._registry.get(target_id)
            if target is not None:
                return target
        return self._registry.pick(anchor.kind, ctx=ctx)

    def _fallback(self, anchor: LodestoneAnchor, sink: Sink, reason: str) -> RenderedAnchor:
        rendered = self._renderer.render(anchor, target=None, sink=sink, ctx=None).rendered
        return RenderedAnchor(
            anchor=anchor,
            target=None,
            link_text=anchor.raw,
            rendered=rendered,
            is_anchor=False,
            fallback_reason=reason,
            sink=sink,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guard_path(anchor: LodestoneAnchor, ctx: AnchorContext) -> tuple[bool, str | None]:
    """Return ``(ok, reason)``.

    Rules (deliberately conservative — when in doubt, refuse to URL-ise):

    *   Non file anchors always pass.
    *   ``url``-kind anchors are not subject to the workspace guard
        (the URL is the anchor); the renderer rejects unknown schemes
        instead.
    *   When ``ctx.workspace_root`` is ``None`` we still accept the path
        but emit a warning reason — the caller may surface that in the
        audit log.  Returning ``ok=True`` here matches ``LODESTONE=off``
        behaviour: the tool still works without a workspace.
    *   When the resolved absolute path would escape ``workspace_root``
        (e.g. ``../../etc/passwd``) we refuse to URL-ise.
    """
    if anchor.kind not in ("file_path", "function_ref", "git_blob"):
        return True, None
    rel = anchor.file_path
    if not rel:
        return True, None
    root = ctx.workspace_root
    if root is None:
        return True, "no workspace_root; path not verified"
    try:
        candidate = Path(rel) if Path(rel).is_absolute() else (root / rel)
        candidate = candidate.resolve()
        root_resolved = root.resolve()
        candidate.relative_to(root_resolved)  # raises ValueError if outside
        return True, None
    except ValueError:
        return False, "path escapes workspace_root"
    except OSError:
        return False, "path unsafe (OS error)"


def probe_editor_from_env(env: dict[str, str] | None = None) -> str | None:
    """Return a target_id when ``TERM_PROGRAM`` / known markers hint at an IDE.

    Examples:

    *   ``TERM_PROGRAM=vscode`` → ``"vscode"``
    *   ``CURSOR_TRACE_ID=1``  → ``"cursor"``
    *   ``idea`` / ``idea64`` / ``pycharm`` on ``PATH`` → ``"idea"``
    *   ``subl`` on ``PATH``   → ``"subl"``
    """
    e = env if env is not None else dict(os.environ)
    term = (e.get("TERM_PROGRAM") or "").lower()
    if term in {"vscode", "code", "vscode-insiders"}:
        return "vscode" if "insiders" not in term else "vscode-insiders"
    if e.get("CURSOR_TRACE_ID"):
        return "cursor"
    if e.get("JETBRAINS_IDE") or e.get("IDEA_HOME"):
        return "idea"
    if _which("idea") or _which("idea64") or _which("pycharm"):
        return "idea"
    if _which("subl") or _which("sublime_text"):
        return "subl"
    return None


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


__all__ = [
    "AnchorResolver",
    "probe_editor_from_env",
]
