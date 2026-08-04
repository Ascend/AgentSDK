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

"""F-97 LODESTONE — renderer.

Three output sinks:

*   ``text``     — anchor text passed through verbatim.
*   ``markdown`` — ``[text](url)`` form.  Wrap in ``<…>`` when the URL
    itself contains brackets / pipes (CommonMark escape rule).
*   ``osc8``     — ANSI escape ``\\x1b]8;;URL\\x1b\\\\TEXT\\x1b]8;;\\x1b\\\\``
    that modern terminals (iTerm2, WezTerm, VS Code) render as
    clickable hyperlinks.

``sink="auto"`` is resolved at first-render-time to one of the three
concrete sinks, based on env hints:

*   ``TERM_PROGRAM`` in {``iTerm.app``, ``WezTerm``, ``vscode``} → ``osc8``
*   anything else → ``markdown``

Public surface:

*   :class:`AnchorRenderer` — stateless; ``render(anchor, target, sink, ctx)``
*   :meth:`AnchorRenderer.render_text(text, ctx, sink)` — parse + render
    mixed text without round-tripping through AnchorParser/Renderer
    manually.

Path / URL substitution is shared between sink forms; see
:func:`_build_template_vars`.
"""

# pylint: disable=cyclic-import
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import (
    AnchorContext,
    AnchorTarget,
    LodestoneAnchor,
    RenderedAnchor,
    Sink,
)

_AUTO_OSC8_TERM_PROGRAMS = frozenset({"iTerm.app", "WezTerm", "vscode"})


class AnchorRenderer:
    """Renders :class:`LodestoneAnchor` instances into text/markdown/osc8.

    The renderer is intentionally stateless; the same instance is safe
    to share across requests.
    """

    # -- public API ----------------------------------------------------------

    def render(
        self,
        anchor: LodestoneAnchor,
        *,
        target: AnchorTarget | None,
        sink: Sink,
        ctx: AnchorContext | None = None,
    ) -> RenderedAnchor:
        sink = self.normalize_sink(sink, env=ctx.env if ctx else None)

        link_text = anchor.link_text()
        if target is None or anchor.kind == "url":
            url = anchor.url if anchor.kind == "url" else None
            rendered = self._render_url_or_plain(link_text, url=url, sink=sink)
            return RenderedAnchor(
                anchor=anchor,
                target=None,
                link_text=link_text,
                rendered=rendered,
                is_anchor=bool(url),
                fallback_reason=None if url else "no target available",
                sink=sink,
            )

        # Build the substitution context.
        try:
            url = _render_template(target.template, _template_vars(anchor, ctx))
        except TemplateFormatError as exc:
            return RenderedAnchor(
                anchor=anchor,
                target=None,
                link_text=link_text,
                rendered=link_text,
                is_anchor=False,
                fallback_reason=f"template error: {exc}",
                sink=sink,
            )

        if not url:
            return RenderedAnchor(
                anchor=anchor,
                target=target,
                link_text=link_text,
                rendered=link_text,
                is_anchor=False,
                fallback_reason="empty url after substitution",
                sink=sink,
            )

        rendered = self._format_link(link_text, url, sink)
        # In sink="text", the caller wants the URL itself (so it can open
        # it or audit-log it).  We return ``url`` here to keep the
        # contract tight.  Use sink="markdown"/"osc8" to keep the
        # bracketed display form.
        if sink == "text":
            rendered = url
        return RenderedAnchor(
            anchor=anchor,
            target=target,
            link_text=link_text,
            rendered=rendered,
            is_anchor=True,
            sink=sink,
        )

    def render_text(
        self,
        text: str,
        *,
        ctx: AnchorContext,
        sink: Sink,
    ) -> str:
        """Parse ``text``, render every anchor, splice back into the string.

        Spans are processed right-to-left so earlier ``start`` indices
        remain valid as we rewrite the original string.
        """
        if not text:
            return text or ""
        # Imported here to avoid an import cycle when ``renderer`` is
        # imported standalone in tests.
        from .parser import AnchorParser
        from .resolver import AnchorResolver

        parser = AnchorParser()
        resolver = AnchorResolver(self._registry_for_resolve(ctx))
        sink = self.normalize_sink(sink, env=ctx.env)

        anchors = parser.parse(text)
        if not anchors:
            return text

        # Render in reverse span order so replacements don't shift indices.
        anchors_sorted = sorted(
            anchors,
            key=lambda a: a.span[0] if a.span else 0,
            reverse=True,
        )
        out = text
        for anchor in anchors_sorted:
            if anchor.span is None:
                continue
            rendered = resolver.resolve(anchor, ctx=ctx, sink=sink)
            out = out[: anchor.span[0]] + rendered.rendered + out[anchor.span[1] :]
        return out

    # -- sink helpers --------------------------------------------------------

    def normalize_sink(
        self,
        sink: Sink,
        *,
        env: dict[str, str] | None = None,
    ) -> Sink:
        """Normalize ``auto`` → one of ``text`` / ``markdown`` / ``osc8``."""
        if sink != "auto":
            return sink
        e = env if env is not None else dict(os.environ)
        term = (e.get("TERM_PROGRAM") or "").lower()
        if term in {p.lower() for p in _AUTO_OSC8_TERM_PROGRAMS}:
            return "osc8"
        return "markdown"

    # -- low-level formatting ------------------------------------------------

    def _render_url_or_plain(
        self,
        text: str,
        *,
        url: str | None,
        sink: Sink,
    ) -> str:
        if not url:
            return text
        return self._format_link(text, url, sink)

    def _format_link(self, text: str, url: str, sink: Sink) -> str:
        if sink == "text":
            # Keep the raw text — downstream audit may want a stable copy.
            return text
        if sink == "markdown":
            safe_text = _markdown_escape(text)
            safe_url = _markdown_escape_url(url)
            return f"[{safe_text}]({safe_url})"
        if sink == "osc8":
            return _osc8_hyperlink(text, url)
        # Unknown sinks fall back to text.
        return text

    # -- resolver adapter ----------------------------------------------------

    def _registry_for_resolve(self, ctx: AnchorContext):  # pragma: no cover
        # Lazy-import: avoid a circular import at module load.
        from .targets import build_default_registry

        return build_default_registry(ctx.config)


# ---------------------------------------------------------------------------
# Public ``open`` helper — call out to the system default URL handler.
# ---------------------------------------------------------------------------


class OpenLaunchError(RuntimeError):
    """Raised when the OS refuses to launch ``uri``."""


def open_uri(uri: str) -> None:
    """Open ``uri`` with the platform-default handler.

    *   macOS   → ``open <uri>``
    *   Linux   → ``xdg-open <uri>``
    *   Windows → ``cmd /c start "" <uri>``

    Raises :class:`OpenLaunchError` on any failure.
    """
    if not uri:
        raise OpenLaunchError("empty uri")

    try:
        if sys.platform == "darwin":
            rc = subprocess.call(["open", uri])  # nosec B607
        elif sys.platform.startswith("win"):
            rc = subprocess.call(["cmd", "/c", "start", "", uri], shell=False)  # nosec B607
        else:
            xdg = shutil.which("xdg-open")
            if xdg is None:
                raise OpenLaunchError("xdg-open not installed")
            rc = subprocess.call([xdg, uri])
    except (FileNotFoundError, OSError) as exc:
        raise OpenLaunchError(str(exc)) from exc
    if rc != 0:
        raise OpenLaunchError(f"open exited with code {rc}")


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TemplateFormatError(ValueError):
    """Raised when a template substitution fails (missing placeholder, etc.)."""


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _render_template(template: str, vars_map: dict[str, str]) -> str:
    """Replace ``{placeholder}`` occurrences with values from ``vars_map``.

    Unknown placeholders raise :class:`TemplateFormatError`.  Empty
    values become empty strings (the renderer treats them as a
    ``fallback_reason`` signal).
    """
    out: list[str] = []
    last = 0
    for m in _PLACEHOLDER_RE.finditer(template):
        out.append(template[last : m.start()])
        key = m.group(1)
        if key not in vars_map:
            raise TemplateFormatError(f"unknown placeholder: {key!r}")
        out.append(vars_map[key])
        last = m.end()
    out.append(template[last:])
    return "".join(out)


def _template_vars(anchor: LodestoneAnchor, ctx: AnchorContext | None) -> dict[str, str]:
    """Map an anchor + context to a placeholder dict.

    ``abs`` requires ``ctx.workspace_root``; if missing the field is left
    blank and :func:`AnchorRenderer.render` will flag an empty URL.
    """
    vars_map: dict[str, str] = {}
    ctx = ctx or AnchorContext(
        workspace_root=None,
        session_id=None,
        config=anchor.raw and _empty_config(),  # type: ignore[arg-type]
    )

    # Path
    rel = anchor.file_path or ""
    if rel:
        if ctx.workspace_root:
            try:
                if Path(rel).is_absolute():
                    vars_map["abs"] = str(Path(rel).resolve())
                else:
                    vars_map["abs"] = str((ctx.workspace_root / rel).resolve())
            except OSError:
                vars_map["abs"] = ""
        else:
            vars_map["abs"] = rel
    else:
        vars_map["abs"] = ""
    vars_map["rel"] = rel
    vars_map["line"] = str(anchor.line) if anchor.line is not None else "1"
    vars_map["col"] = str(anchor.column) if anchor.column is not None else "1"
    vars_map["end_line"] = str(anchor.end_line) if anchor.end_line is not None else ""
    vars_map["end_col"] = str(anchor.end_column) if anchor.end_column is not None else ""

    # Git / remote
    vars_map["remote"] = _remote(ctx)
    vars_map["ref"] = anchor.git_sha or ""
    vars_map["branch"] = ctx.branch or ""

    # Tracker
    if anchor.tracker_key is not None:
        host_id, key = anchor.tracker_key
        vars_map["key"] = key or ""
        vars_map["host"] = _tracker_full_host(host_id, ctx)
    else:
        vars_map["key"] = ""
        vars_map["host"] = ctx.config.default_tracker_host if ctx.config else "gitcode.com"

    # owner / repo for tracker templates
    owner, repo = _owner_repo(ctx)
    vars_map["owner"] = owner
    vars_map["repo"] = repo

    # Linear workspace — heuristic: ``ctx.config.extra_hosts`` slot or
    # ``"default"``.  Tool users override per call.
    vars_map["workspace"] = ctx.config.extra_hosts[0] if ctx.config and ctx.config.extra_hosts else "default"
    return vars_map


def _empty_config():
    from .models import LodestoneConfig

    return LodestoneConfig()


def _remote(ctx: AnchorContext) -> str:
    url = ctx.remote_url or ""
    # Trim ``.git`` so generated URLs omit it.
    url = url.removesuffix(".git")
    return url


def _owner_repo(ctx: AnchorContext) -> tuple[str, str]:
    cfg = ctx.config
    if cfg and cfg.default_tracker_repo:
        return cfg.default_tracker_repo
    # Try to parse from remote URL.
    if ctx.remote_url:
        from .fingerprint import parse_remote_url

        parsed = parse_remote_url(ctx.remote_url)
        if parsed:
            return parsed[1], parsed[2]
    return "", ""


def _tracker_full_host(short_host: str | None, ctx: AnchorContext) -> str:
    """Map a tracker_key host token (``gitcode`` / ``linear`` / ``github``)
    to the full domain used in template substitution.

    Defaults to ``ctx.config.default_tracker_host`` when no token is set
    or no canonical mapping is known.  Domain is preserved unchanged
    when already ``*.com`` / ``*.app``.
    """
    if not short_host:
        return ctx.config.default_tracker_host if ctx.config else "gitcode.com"
    h = short_host.lower()
    if "." in h:
        return h  # already a domain
    mapping = {
        "gitcode": "gitcode.com",
        "github": "github.com",
        "gitee": "gitee.com",
        "linear": "linear.app",
    }
    return mapping.get(h, h + ".com")


# ---------------------------------------------------------------------------
# Markdown / OSC 8 helpers
# ---------------------------------------------------------------------------


def _markdown_escape(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]").replace("\n", " ")


_MD_UNSAFE_URL_RE = re.compile(r"[\[\]\(\)\s]")


def _markdown_escape_url(url: str) -> str:
    if _MD_UNSAFE_URL_RE.search(url):
        return "<" + url.replace(">", "%3E") + ">"
    return url


def _osc8_hyperlink(text: str, url: str) -> str:
    """ANSI escape to hyperlink ``text`` to ``url``."""

    # OSC 8 ; params ; URI ST  <text>  OSC 8 ; ; ST
    # ``\x1b]8;;URL\x1b\\TEXT\x1b]8;;\x1b\\``
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


__all__ = [
    "AnchorRenderer",
    "OpenLaunchError",
    "TemplateFormatError",
    "open_uri",
]


# ``AnchorContext`` + ``LodestoneAnchor`` references — kept for typing clarity.
_ = (AnchorContext, LodestoneAnchor)


# Silence unused-import lint for ``dataclass`` references.
_ = (dataclass,)
