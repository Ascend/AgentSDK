"""Convert an asciicast v2 ``.cast`` into an MP4.

The conversion is a 2-step pipeline:

1.  :func:`render_cast_to_pngs` walks the ``.cast`` NDJSON stream and
    emits one PNG per ``dashboard:snapshot`` marker.  Each PNG is held
    on screen for the gap between consecutive markers (with a 1-second
    floor so a fast demo doesn't flash by); the last frame gets an
    extra 2-second tail.
2.  :func:`pngs_to_mp4` duplicates the PNGs to fill the hold time, hands
    the sequence to ``ffmpeg`` via its concat demuxer, and lets it
    produce the final MP4.

Why a wrapper instead of pushing the user at ``asciinema agg`` or
``ffmpeg`` directly?  ``agg`` only writes GIF (no MP4); piping ``ffmpeg``
by hand requires juggling the concat manifest, durations, the
``yuv420p`` colorspace quirk and ``+faststart`` for browser playback —
all of which are easy to get wrong.  This module codifies the same
choices the reference impl in ``/tmp/cast_to_mp4.py`` made, so the
output is reproducible and CI-friendly.

Output format notes:

* ``-vf format=yuv420p`` is mandatory: Chrome and Safari refuse to play
  h264 in any other colorspace. Source: ffmpeg h264 wiki (the wiki
  page on ``yuv420p``-only MP4 is the canonical guidance).
* ``-movflags +faststart`` moves the ``moov`` atom to the front of the
  file so the browser can stream-instead-of-buffering. Source: ffmpeg
  official docs on ``faststart``.
* Emoji rendering — Pillow's default font fallback does not ship with
  emoji glyphs on most Linux systems, so emoji codepoints in the
  ``.cast`` would render as empty tofu rectangles. This module always
  substitutes them with ASCII labels in the PNG output. The original
  ``.cast`` keeps the emoji intact for ``asciinema-player`` playback.

Dependencies:

* ``Pillow`` (imported lazily — if missing, we raise a clear
  ``ModuleNotFoundError`` with the install command).
* ``ffmpeg`` on ``PATH`` (the conversion raises ``RuntimeError`` with a
  clear instruction when it's missing).

The module is **Layer 2** (extensions) per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = [
    "render_cast_to_pngs",
    "pngs_to_mp4",
    "convert_cast_to_mp4",
    "build_cast_to_mp4_parser",
    "run_cast_to_mp4_command",
]


# ---------------------------------------------------------------------------
# Pillow import guard
# ---------------------------------------------------------------------------


def _import_pillow():
    """Import :mod:`PIL` lazily and surface a friendly error if missing.

    Recording is the primary purpose of the recording module; pulling
    Pillow at module import would impose a heavy dependency on
    everyone. Callers of this converter get a precise install hint
    instead.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

        return Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError("cast_to_mp4 requires Pillow. Install with: pip install Pillow") from exc


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------


def _font(size: int, ImageFont, *, bold: bool = False):
    """Return the best monospace ``ImageFont`` Pillow can find on this box.

    Falls back to ``ImageFont.load_default()`` if no TTF is reachable —
    the output is still readable, just bitmap-aliased.
    """
    candidates: tuple[str, ...]
    if bold:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
        )
    else:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# ANSI color handling for terminal-accurate rendering
# ---------------------------------------------------------------------------


#: Default 16-color ANSI palette (xterm standard).
_ANSI_PALETTE = {
    "default": (201, 209, 217),
    "black": (0, 0, 0),
    "red": (205, 49, 49),
    "green": (13, 188, 121),
    "yellow": (229, 229, 16),
    "blue": (36, 114, 200),
    "magenta": (188, 63, 188),
    "cyan": (17, 168, 205),
    "white": (229, 229, 229),
    "brightblack": (102, 102, 102),
    "brightred": (241, 76, 76),
    "brightgreen": (35, 209, 139),
    "brightyellow": (245, 245, 67),
    "brightblue": (59, 142, 234),
    "brightmagenta": (214, 112, 214),
    "brightcyan": (41, 184, 219),
    "brightwhite": (255, 255, 255),
}


def _resolve_color(value: str | int, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Resolve a pyte color value to an RGB tuple."""
    if value == "default":
        return default
    if isinstance(value, int):
        # 256-color index — map the 6x6x6 color cube and grayscale ramp.
        if 0 <= value <= 15:
            name = list(_ANSI_PALETTE)[value + 1]
            return _ANSI_PALETTE[name]
        if 16 <= value <= 231:
            value -= 16
            r = (value // 36) * 51
            g = ((value // 6) % 6) * 51
            b = (value % 6) * 51
            return (r, g, b)
        if 232 <= value <= 255:
            gray = 8 + (value - 232) * 10
            return (gray, gray, gray)
        return default
    if isinstance(value, str):
        value = value.lower()
        if value in _ANSI_PALETTE:
            return _ANSI_PALETTE[value]
        if len(value) == 6:
            try:
                return (
                    int(value[0:2], 16),
                    int(value[2:4], 16),
                    int(value[4:6], 16),
                )
            except ValueError:
                pass
    return default


# ---------------------------------------------------------------------------
# Rendering: .cast → PNG sequence
# ---------------------------------------------------------------------------

#: Status colors used in the rendered kanban / dashboard panels. Pulled
#: out as module-level constants so callers (e.g. tests) can swap them
#: for predictable snapshot comparisons.
BADGE_PALETTE = {
    "pending": (110, 118, 129),
    "running": (88, 166, 255),
    "done": (63, 185, 80),
    "failed": (248, 81, 73),
    "blocked": (210, 153, 34),
}


def _load_jsonl_entry(line: str, cast_path: Path, role: str) -> object:
    """Parse one NDJSON line, normalizing JSON errors to ``RuntimeError``.

    ``json.loads`` raises :class:`json.JSONDecodeError` (which inherits
    ``ValueError``) on malformed input. CLI users should not see a raw
    traceback for that — they should see a one-line error naming the
    file and the offending entry. ``role`` is ``"header"`` or
    ``"event frame"`` and is included in the message so the user knows
    whether the bad line was the file header or an event frame.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{cast_path}: malformed {role} at line {exc.lineno} column {exc.colno}: {exc.msg}") from exc


def _badge_substitutions(palette: dict[str, tuple[int, int, int]]):
    """Build the (emoji, label, color) tuples used by the renderer.

    Rendering order matters: longer emoji come first so multi-codepoint
    sequences (none today, but cheap insurance) match before a prefix
    emoji does.
    """
    return (
        ("⏳", "[pending]", palette["pending"]),
        ("🔵", "[running]", palette["running"]),
        ("✅", "[done]", palette["done"]),
        ("❌", "[failed]", palette["failed"]),
        ("🚧", "[blocked]", palette["blocked"]),
    )


def render_cast_to_pngs(
    cast_path: Path,
    out_dir: Path,
    *,
    width_px: int = 960,
    height_px: int = 480,
    bg: tuple[int, int, int] = (14, 17, 22),
    fg: tuple[int, int, int] = (201, 209, 217),
    palette: dict[str, tuple[int, int, int]] | None = None,
    snapshot_markers: tuple[str, ...] = ("dashboard:snapshot", "repl:prompt:start"),
    fallback_interval_s: float = 2.0,
) -> list[tuple[Path, float]]:
    """Render one PNG per meaningful snapshot from ``cast_path``.

    By default it looks for ``dashboard:snapshot`` or
    ``repl:prompt:start`` markers.  If neither is present, it falls back
    to sampling all ``"o"`` frames at ``fallback_interval_s`` intervals
    so arbitrary asciicast recordings (e.g. a real REPL session) can
    still be converted to MP4.

    Returns ``[(png_path, hold_seconds)]``.  ``hold_seconds`` is how
    long the player should display that frame before moving on, derived
    from the gap between consecutive markers (or intervals in fallback
    mode).  The final frame is held for 2 extra seconds so the demo
    doesn't disappear too abruptly.
    """
    Image, ImageDraw, ImageFont = _import_pillow()
    raw = cast_path.read_text(encoding="utf-8").splitlines()
    if not raw:
        raise RuntimeError(f"{cast_path} is empty")
    header = _load_jsonl_entry(raw[0], cast_path, "header")
    title = header.get("title") or cast_path.stem

    events = [_load_jsonl_entry(line, cast_path, "event frame") for line in raw[1:] if line.strip()]

    snapshots = _group_output_by_markers(events, markers=snapshot_markers)
    if not snapshots:
        snapshots = _sample_output_by_interval(events, interval_s=fallback_interval_s)
    if not snapshots:
        raise RuntimeError("no renderable output frames found in .cast")

    holds = _compute_hold_seconds(snapshots)

    font = _font(14, ImageFont)
    badge_font = _font(13, ImageFont)
    ansi_font = _font(14, ImageFont)
    ansi_font_bold = _font(14, ImageFont, bold=True)
    effective_palette = palette or BADGE_PALETTE

    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect ANSI output: if any snapshot contains an escape byte we
    # render the whole cast as a terminal screen so colors / cursor moves
    # survive into the MP4.
    has_ansi = any("\x1b" in chunk for _, outputs in snapshots for chunk in outputs)

    pairs: list[tuple[Path, float]] = []
    for i, ((_, outputs), hold) in enumerate(zip(snapshots, holds)):
        png = out_dir / f"frame_{i:03d}.png"
        if has_ansi:
            _draw_frame_ansi(
                png,
                outputs=outputs,
                title=title,
                width_px=width_px,
                height_px=height_px,
                header_width=header.get("width", 80),
                header_height=header.get("height", 24),
                bg=bg,
                fg=fg,
                font=ansi_font,
                bold_font=ansi_font_bold,
                Image=Image,
                ImageDraw=ImageDraw,
            )
        else:
            _draw_frame(
                png,
                outputs=outputs,
                title=title,
                width_px=width_px,
                height_px=height_px,
                bg=bg,
                fg=fg,
                palette=effective_palette,
                font=font,
                badge_font=badge_font,
                Image=Image,
                ImageDraw=ImageDraw,
            )
        pairs.append((png, hold))

    return pairs


def _group_output_by_markers(
    events: list[list[object]],
    *,
    markers: tuple[str, ...],
) -> list[tuple[float, list[str]]]:
    """Group ``"o"`` frames between recognized markers."""
    snapshots: list[tuple[float, list[str]]] = []
    current_marker_t: float | None = None
    current_outputs: list[str] = []
    for event in events:
        if event[1] == "m" and event[2] in markers:
            if current_marker_t is not None:
                snapshots.append((current_marker_t, current_outputs))
            current_marker_t = event[0]
            current_outputs = []
        elif event[1] == "o" and current_marker_t is not None:
            current_outputs.append(event[2])
    if current_marker_t is not None:
        snapshots.append((current_marker_t, current_outputs))
    return snapshots


def _sample_output_by_interval(
    events: list[list[object]],
    *,
    interval_s: float,
) -> list[tuple[float, list[str]]]:
    """Fallback sampler: one snapshot per ``interval_s`` of ``"o"`` output."""
    snapshots: list[tuple[float, list[str]]] = []
    current_outputs: list[str] = []
    last_t: float | None = None
    for event in events:
        if event[1] != "o":
            continue
        t = float(event[0])
        if last_t is None or t - last_t >= interval_s:
            if last_t is not None:
                snapshots.append((last_t, current_outputs))
            last_t = t
            current_outputs = [event[2]]
        else:
            current_outputs.append(event[2])
    if last_t is not None:
        snapshots.append((last_t, current_outputs))
    return snapshots


def _compute_hold_seconds(
    snapshots: list[tuple[float, list[str]]],
    *,
    min_hold: float = 1.0,
    tail_hold: float = 2.0,
) -> list[float]:
    """Compute per-frame hold time from the timestamp gaps.

    A ``min_hold`` floor avoids a fast demo flashing by in a fraction of
    a second; the last frame gets the ``tail_hold`` extra seconds.
    """
    holds: list[float] = []
    for i, (t, _) in enumerate(snapshots):
        if i + 1 < len(snapshots):
            hold = max(min_hold, snapshots[i + 1][0] - t)
        else:
            hold = tail_hold
        holds.append(hold)
    return holds


def _draw_frame(
    png: Path,
    *,
    outputs: list[str],
    title: str,
    width_px: int,
    height_px: int,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    palette: dict[str, tuple[int, int, int]],
    font,
    badge_font,
    Image,
    ImageDraw,
) -> None:
    """Render one PNG frame from a list of captured output chunks.

    Every emoji badge in any line is replaced with its ASCII label and
    the line picks up the badge's color; lines without any badge stay
    in the default ``fg``.
    """
    img = Image.new("RGB", (width_px, height_px), color=bg)
    draw = ImageDraw.Draw(img)
    # Title strip.
    draw.rectangle((0, 0, width_px, 32), fill=(22, 27, 34))
    draw.text((12, 8), title, fill=fg, font=font)

    y = 48
    bottom_limit = height_px - 24
    substitutions = _badge_substitutions(palette)
    for chunk in outputs:
        for line in chunk.splitlines():
            if not line.strip():
                y += 8
                continue
            color = fg
            display_line = line
            # Replace ALL emoji badges in the line (not just the first).
            # Track whether any badge was present so the line gets a
            # non-default colour only when a badge actually appeared.
            any_replaced = False
            for emoji, ascii_label, badge_color in substitutions:
                if emoji in display_line:
                    any_replaced = True
                    color = badge_color
                    display_line = display_line.replace(emoji, ascii_label)
            if any_replaced:
                # Stay on ``color`` from the *last* match (the user
                # sees a stats line like ``[..] [..] [done] [..]``
                # and the dominant label is the right-most one).
                pass
            draw.text((12, y), display_line, fill=color, font=font)
            y += 18
            if y > bottom_limit:
                break
        if y > bottom_limit:
            break

    img.save(png, "PNG")


def _draw_frame_ansi(
    png: Path,
    *,
    outputs: list[str],
    title: str,
    width_px: int,
    height_px: int,
    header_width: int,
    header_height: int,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    font,
    bold_font,
    Image,
    ImageDraw,
) -> None:
    """Render one PNG frame by emulating a terminal screen.

    Uses ``pyte`` to parse ANSI escape sequences (colors, cursor moves,
    bold, etc.) and then draws each cell of the resulting screen buffer
    with Pillow. This is what makes recordings of the real interactive
    REPL look like the reference screenshot instead of raw escape-code
    soup.
    """
    try:
        import pyte  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("ANSI-aware MP4 rendering requires `pyte`. Install with: pip install pyte") from exc

    screen = pyte.Screen(header_width, header_height)
    stream = pyte.Stream(screen)
    for chunk in outputs:
        stream.feed(chunk)

    img = Image.new("RGB", (width_px, height_px), color=bg)
    draw = ImageDraw.Draw(img)
    # Title strip.
    draw.rectangle((0, 0, width_px, 32), fill=(22, 27, 34))
    draw.text((12, 8), title, fill=fg, font=font)

    cell_w = width_px // header_width
    cell_h = (height_px - 32) // header_height
    if cell_w <= 0 or cell_h <= 0:
        raise RuntimeError(
            f"terminal dimensions too large for {width_px}x{height_px} image: {header_width}x{header_height}"
        )

    # Use a slightly smaller font size than the cell so glyphs fit.
    font_size = max(8, cell_h - 2)
    font = font.font_variant(size=font_size)
    bold_font = bold_font.font_variant(size=font_size)

    base_y = 32
    for y in range(header_height):
        row = screen.buffer[y]
        for x in range(header_width):
            char_obj = row.get(x)
            if char_obj is None:
                continue
            char = char_obj.data
            if char == " " and char_obj.bg == "default":
                continue
            px = x * cell_w
            py = base_y + y * cell_h
            cell_bg = _resolve_color(char_obj.bg, default=bg)
            cell_fg = _resolve_color(char_obj.fg, default=fg)
            draw.rectangle(
                (px, py, px + cell_w - 1, py + cell_h - 1),
                fill=cell_bg,
            )
            face = bold_font if char_obj.bold else font
            draw.text((px, py), char, fill=cell_fg, font=face)

    img.save(png, "PNG")


# ---------------------------------------------------------------------------
# Encoding: PNG sequence → MP4 via ffmpeg
# ---------------------------------------------------------------------------


def pngs_to_mp4(
    pairs: list[tuple[Path, float]],
    mp4_path: Path,
    *,
    fps: int = 4,
) -> None:
    """Stitch the PNG sequence into an MP4 at ``fps``.

    Each PNG is held on screen for ``hold`` seconds, then a hard cut
    to the next.  We duplicate each PNG to fill the hold time at the
    given ``fps`` and emit a concat demuxer list for ffmpeg.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    if fps <= 0:
        raise ValueError("fps must be a positive integer")

    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        seq_files: list[str] = []
        for idx, (png, hold) in enumerate(pairs):
            n_dupes = max(1, round(hold * fps))
            for d in range(n_dupes):
                link = td_path / f"{idx:04d}_{d:03d}.png"
                link.symlink_to(png.resolve())
                seq_files.append(link.name)

        concat_list = td_path / "concat.txt"
        with concat_list.open("w") as fp:
            for name in seq_files:
                fp.write(f"file '{name}'\n")
                fp.write(f"duration {1.0 / fps:.4f}\n")
            # Repeat the last frame so the final duration is honored.
            fp.write(f"file '{seq_files[-1]}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Combined convenience
# ---------------------------------------------------------------------------


def convert_cast_to_mp4(
    cast_path: Path,
    mp4_path: Path,
    *,
    fps: int = 4,
    width_px: int = 960,
    height_px: int = 480,
    keep_pngs: bool = False,
) -> list[tuple[Path, float]]:
    """One-shot ``.cast → MP4`` helper.

    Returns the list of ``(png_path, hold_seconds)`` for the
    intermediate frames; if ``keep_pngs`` is set, the PNGs are copied
    next to ``mp4_path`` as ``<mp4_path>.pngs/``.
    """
    mp4_path = Path(mp4_path)
    with tempfile.TemporaryDirectory() as td:
        png_dir = Path(td) / "pngs"
        pairs = render_cast_to_pngs(
            cast_path,
            png_dir,
            width_px=width_px,
            height_px=height_px,
        )
        if keep_pngs:
            keep_dir = mp4_path.with_suffix(mp4_path.suffix + ".pngs")
            keep_dir.mkdir(parents=True, exist_ok=True)
            for png, _ in pairs:
                shutil.copy(png, keep_dir / png.name)
        pngs_to_mp4(pairs, mp4_path, fps=fps)
    return pairs


# ---------------------------------------------------------------------------
# Argparse / CLI
# ---------------------------------------------------------------------------


def build_cast_to_mp4_parser() -> argparse.ArgumentParser:
    """Build the argparse for ``clawcodex cast-to-mp4 ...``."""
    p = argparse.ArgumentParser(
        prog="clawcodex cast-to-mp4",
        description=(
            "Convert an asciicast v2 .cast file produced by "
            "`clawcodex record` into a video (default: MP4 / h264). "
            "Renders one PNG per dashboard:snapshot marker, then "
            "encodes the PNG sequence via ffmpeg."
        ),
    )
    p.add_argument(
        "--cast",
        type=Path,
        required=True,
        help="Input .cast NDJSON file.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output video file path (e.g. demo.mp4).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frame rate for the rendered PNG sequence (default: 4).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=960,
        help="PNG width in pixels (default: 960).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=480,
        help="PNG height in pixels (default: 480).",
    )
    p.add_argument(
        "--keep-pngs",
        action="store_true",
        help="Copy the intermediate PNGs next to --out as <out>.pngs/.",
    )
    return p


def run_cast_to_mp4_command(args: list[str] | None = None) -> int:
    """Subcommand entry-point invoked by ``subcommand_registry``."""
    parser = build_cast_to_mp4_parser()
    parsed = parser.parse_args(args)

    cast_path = Path(parsed.cast).expanduser().resolve()
    out_path = Path(parsed.out).expanduser().resolve()

    if not cast_path.exists():
        print(f"error: input .cast not found: {cast_path}", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print(
            "error: ffmpeg not found in PATH; install ffmpeg (https://ffmpeg.org/download.html) and retry",
            file=sys.stderr,
        )
        return 2

    try:
        pairs = convert_cast_to_mp4(
            cast_path,
            out_path,
            fps=parsed.fps,
            width_px=parsed.width,
            height_px=parsed.height,
            keep_pngs=parsed.keep_pngs,
        )
    except RuntimeError as exc:
        print(f"error: cast→mp4 failed: {exc}", file=sys.stderr)
        return 1

    print(f"[cast→mp4] {out_path} — {len(pairs)} snapshot(s) at {parsed.fps}fps → {out_path.stat().st_size:,} bytes")
    if parsed.keep_pngs:
        keep_dir = out_path.with_suffix(out_path.suffix + ".pngs")
        print(f"[cast→mp4] PNGs preserved at {keep_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual smoke
    raise SystemExit(run_cast_to_mp4_command())
