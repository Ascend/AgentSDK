#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""OKLCH-based REPL color scheme.

Provides a perceptually-uniform palette built on the OKLCH color space.
All color tokens are defined as OKLCH(L, C, H) tuples and converted to
sRGB hex at import time so Rich / prompt_toolkit can consume them.

Why OKLCH:
  - Perceptually uniform: equal deltas in L/C/H produce visually
    equal deltas — unlike HSL or sRGB, where a 20° hue shift may
    look dramatic in one region and invisible in another.
  - Lightness-independent chroma: C controls saturation without
    affecting perceived brightness, so semantic colors (red/green/
    amber) remain equally legible on the same background.
  - Wide gamut: OKLCH encodes Display P3 natively; sRGB clamping
    is a one-line clamp at the output stage.

Usage::

    from clawcodex_ext.repl.color_scheme import build_rich_theme, build_ptk_style, DARK

    console = Console(theme=build_rich_theme(DARK))
    prompt_style = Style.from_dict(build_ptk_style(DARK))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

# ---------------------------------------------------------------------------
# OKLCH → sRGB conversion  (no external dependencies)
# ---------------------------------------------------------------------------
# Reference: https://bottosson.github.io/posts/oklab/
#
# OKLCH  ->  OKLab  ->  Linear sRGB  ->  sRGB (gamma-corrected)

_SRGB_TRANSFER_CUTOFF = 0.0031308


def _srgb_transfer(c: float) -> float:
    """Gamma-compand from linear to sRGB."""
    c = max(0.0, min(1.0, c))
    if c <= _SRGB_TRANSFER_CUTOFF:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def oklch_to_hex(lightness: float, c: float, h: float) -> str:
    """Convert OKLCH to sRGB hex string ``#rrggbb``.

    Parameters
    ----------
    l : float
        Lightness in [0, 1].
    c : float
        Chroma (saturation) in [0, ~0.4].
    h : float
        Hue angle in degrees [0, 360).

    Returns
    -------
    str
        Six-digit hex colour, e.g. ``#1a1a2e``.
    """
    h_rad = math.radians(h)
    a = c * math.cos(h_rad)
    b_val = c * math.sin(h_rad)

    # OKLab → linear sRGB (Bottosson 2021 matrix)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b_val
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b_val
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b_val

    l3 = l_**3
    m3 = m_**3
    s3 = s_**3

    r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    r_g = _srgb_transfer(r)
    g_g = _srgb_transfer(g)
    b_g = _srgb_transfer(b)

    return f"#{int(r_g * 255 + 0.5):02x}{int(g_g * 255 + 0.5):02x}{int(b_g * 255 + 0.5):02x}"


# ---------------------------------------------------------------------------
# Palette definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class REPLPalette:
    """Perceptually-uniform REPL colour palette.

    Every colour is defined in OKLCH space and converted to sRGB hex
    at construction time.  Access the hex string directly (``palette.text``)
    or get a Rich-compatible style name pair via :meth:`rich_style`.
    """

    name: str
    # -- Base surface colours (neutral blue-gray hue 280) --
    background: str  # Page / screen background
    surface: str  # Card / panel / input-row background
    surface_alt: str  # Hover / alternate row
    border: str  # Subtle borders and separators
    # -- Text --
    text: str  # Primary body text (near-white on dark)
    text_muted: str  # Secondary / metadata text
    # -- Semantic accents --
    primary: str  # Brand / interactive accent (warm orange)
    secondary: str  # Secondary text accent (neutral)
    success: str  # Positive outcomes (green)
    warning: str  # Attention (amber)
    error: str  # Errors / failures (red)
    info: str  # Informational text (neutral)
    # -- Role-specific --
    user_bg: str  # User-message background highlight
    agent_label: str  # Agent / assistant name
    tool_name: str  # Tool-call header
    tool_call: str  # Tool-call argument text
    tool_result: str  # Tool-result preview
    tool_error: str  # Tool-error text
    diff_add: str  # Diff addition highlight
    diff_remove: str  # Diff removal highlight
    spinner: str  # LiveStatus spinner / progress
    # -- prompt_toolkit specific --
    prompt_bg: str  # Input prompt background
    prompt_fg: str  # Input prompt foreground
    toolbar: str  # Bottom toolbar text
    completion_current_bg: str  # Completion-menu highlighted background
    completion_current_fg: str  # Completion-menu highlighted foreground
    # Keep this defaulted field last so positional downstream constructors remain
    # source-compatible with palettes created before the shimmer color was added.
    spinner_highlight: str = "#eb9f7f"  # Busy-line shimmer band


# ── Built-in palettes ───────────────────────────────────────────────────
# Exact tokens from the pinned official ui-tui theme. Keeping them as literals
# preserves the official warm-orange brand accent and neutral secondary chrome.

DARK = REPLPalette(
    name="dark",
    background="#1a1a1a",
    surface="#1f1f1f",
    surface_alt="#383838",
    border="#505050",
    text="#ffffff",
    text_muted="#999999",
    primary="#d77757",
    secondary="#bbbbbb",
    success="#4eba65",
    warning="#ffc107",
    error="#ff6b80",
    info="#bbbbbb",
    # Role
    user_bg="#373737",
    agent_label="#ffffff",
    tool_name="#bbbbbb",
    tool_call="#999999",
    tool_result="#999999",
    tool_error="#ff6b80",
    diff_add="#225c2b",
    diff_remove="#7a2936",
    spinner="#d77757",
    spinner_highlight="#eb9f7f",
    # Prompt-specific
    prompt_bg="#373737",
    prompt_fg="#e6e6e6",
    toolbar="#999999",
    completion_current_bg="#383838",
    completion_current_fg="#ffffff",
)


LIGHT = REPLPalette(
    name="light",
    background="#ffffff",
    surface="#f5f5f5",
    surface_alt="#eed6ce",
    border="#afafaf",
    text="#000000",
    text_muted="#666666",
    primary="#d77757",
    secondary="#666666",
    success="#2c7a39",
    warning="#966c1e",
    error="#ab2b3f",
    info="#666666",
    # Role
    user_bg="#f0f0f0",
    agent_label="#000000",
    tool_name="#666666",
    tool_call="#666666",
    tool_result="#666666",
    tool_error="#ab2b3f",
    diff_add="#69db7c",
    diff_remove="#ffa8b4",
    spinner="#d77757",
    spinner_highlight="#f59575",
    # Prompt-specific
    prompt_bg="#f0f0f0",
    prompt_fg="#2b2b2b",
    toolbar="#666666",
    completion_current_bg="#eed6ce",
    completion_current_fg="#000000",
)


# ── Registry ────────────────────────────────────────────────────────────

_PALETTES: dict[str, REPLPalette] = {
    "dark": DARK,
    "light": LIGHT,
}


def get_repl_palette(name: str | None = None) -> REPLPalette:
    """Return a palette by name, falling back to dark."""
    if not name:
        return DARK
    return _PALETTES.get(name.strip().lower(), DARK)


# ---------------------------------------------------------------------------
# Rich Theme builder  —  map ANSI colour names to OKLCH palette values
# ---------------------------------------------------------------------------
# Rich's ``Console(theme=Theme(...))`` overrides named styles so that
# inline markup like ``[error]error[/error]`` renders with our OKLCH red
# instead of the terminal's default ANSI colour.
#
# We also register custom semantic names (``muted``, ``call``, ``agent``)
# for explicit use in new code; the legacy ANSI names ensure every
# existing ``[error]`` / ``[success]`` / ``[warning]`` etc. in the codebase
# instantly picks up the new palette.


def build_rich_theme(palette: REPLPalette | None = None) -> dict:
    """Build a Rich ``Theme``-compatible style dict.

    Returns a dict suitable for ``Console(theme=Theme(...))``.

    Unifies both OKLCH semantic names AND ANSI colour-name aliases so that
    legacy markup like ``[yellow]`` automatically picks up the OKLCH amber
    without per-file changes.  The ANSI aliases are safe for *simple* colour
    tags (``[yellow]``, ``[red]``, ``[bold yellow]``, etc.); composite styles
    like ``[bold red on black]`` are rare in this codebase and handled
    correctly by Rich's style resolution.
    """

    p = palette or DARK
    theme_dict: dict[str, str] = {
        # ── Custom semantic names ──────────────────────────────────────
        "error": p.error,
        "success": p.success,
        "warning": p.warning,
        "info": p.info,
        "muted": p.text_muted,
        "agent": p.agent_label,
        "tool": p.tool_name,
        "call": p.tool_call,
        "result": p.tool_result,
        "spinner": p.spinner,
        "spinner_highlight": p.spinner_highlight,
        "primary": p.primary,
        "secondary": p.secondary,
        "user_bg": p.user_bg,
        "diff_add": p.diff_add,
        "diff_remove": p.diff_remove,
        "key_label": p.warning,  # amber — key names (model=, configured=)
        "value_text": p.secondary,  # purple — string values (MiniMax-M3, yes)
        "version_num": p.info,  # sky-blue — version numbers (2.0)
        # ── ANSI aliases (unified → OKLCH) ────────────────────────────
        # Every ``[red]`` / ``[green]`` / ``[yellow]`` … in the codebase
        # now renders through the OKLCH palette without touching source.
        "red": p.error,
        "green": p.success,
        "yellow": p.warning,
        "blue": p.primary,
        "cyan": p.info,
        "magenta": p.secondary,
    }
    return theme_dict


# ---------------------------------------------------------------------------
# Unified Console factory  —  one import, always OKLCH
# ---------------------------------------------------------------------------

_OKLCH_CONSOLE_CACHE: dict[str, Any] = {}


def build_oklch_console(
    palette: REPLPalette | None = None,
) -> Any:
    """Return a Rich ``Console`` with the OKLCH theme applied.

    Use this instead of bare ``Console()`` everywhere so the perceptually-
    uniform palette is consistent across REPL, CLI, TUI, and session
    browser output.
    """
    from rich.console import Console
    from rich.theme import Theme as _RichTheme

    p = palette or DARK
    return Console(theme=_RichTheme(build_rich_theme(p)), highlight=False)


# ---------------------------------------------------------------------------
# ANSI → OKLCH rewriter  —  for non-Rich contexts (prompt_toolkit, etc.)
# ---------------------------------------------------------------------------

_ANSI_TO_OKLCH: dict[str, str] = {
    "red": "error",
    "green": "success",
    "yellow": "warning",
    "blue": "primary",
    "cyan": "info",
    "magenta": "secondary",
}


def ansi_to_oklch(text: str) -> str:
    """Rewrite ANSI colour tags (``[red]``) to OKLCH semantic names
    (``[error]``).

    Intended for contexts where the Rich theme does not apply — e.g.
    prompt_toolkit ``FormattedText`` rendering, plain ``print()``, or
    any pipeline that strips Rich markup before rendering.
    """
    import re

    def _replace(m: re.Match) -> str:
        slash = m.group(1)  # '' or '/'
        name = m.group(2)
        mapped = _ANSI_TO_OKLCH.get(name, name)
        return f"[{slash}{mapped}]"

    return re.sub(r"\[(/?)(red|green|yellow|blue|cyan|magenta)\]", _replace, text)


# ---------------------------------------------------------------------------
# prompt_toolkit Style builder
# ---------------------------------------------------------------------------


def build_ptk_style(palette: REPLPalette | None = None) -> Dict[str, str]:
    """Build a prompt_toolkit ``Style.from_dict(...)`` dict.

    Controls the input prompt row, bottom toolbar, and completion-menu
    appearance.  Every colour is sourced from the OKLCH palette.
    """
    p = palette or DARK
    return {
        # Input prompt — subtle background highlight so the user input
        # row reads as a discrete block above the transcript.
        "prompt": f"bold fg:{p.prompt_fg} bg:{p.prompt_bg}",
        # Bottom toolbar — compact status row (model · provider · cwd …)
        "bottom-toolbar": f"fg:{p.toolbar} bg:default",
        # Completion menu (slash-command autocomplete)
        "completion-menu": "bg:default",
        "completion-menu.completion": f"fg:{p.text_muted} bg:default",
        "completion-menu.completion.current": (f"fg:{p.completion_current_fg} bg:{p.completion_current_bg} bold"),
        "completion-menu.meta.completion": f"fg:{p.toolbar} bg:default",
        "completion-menu.meta.completion.current": (f"fg:{p.completion_current_fg} bg:{p.completion_current_bg}"),
        "completion.command": f"bold fg:{p.success}",
        "completion.tag": f"italic fg:{p.info}",
        "completion.description": f"fg:{p.text_muted}",
    }
