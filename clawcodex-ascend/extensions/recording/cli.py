"""``clawcodex record`` CLI subcommand.

Standalone entry point that opens an asciicast v2 ``.cast`` file and
plumbs it into one or more subsystem adapters via the
:class:`RecordableSourceRegistry`. The CLI itself is intentionally
minimal — it does *not* try to drive the subsystems. Each registered
source is given the capture handle and is expected to wire itself into
whatever event stream that subsystem exposes (ProgressSink, cron
observers, HeadlessSessionOptions.on_event, DashboardSource, SOP
TeeWriter). Sources tick on their own clock; the CLI just keeps the
writer alive until either a duration elapses or the user hits Ctrl-C.

Layer rule (CLAUDE.md): this module lives under ``extensions/`` (Layer 2)
because it composes per-subsystem adapters that already live there. The
:class:`subcommand_registry` decorator makes the ``record`` subcommand
discoverable without hard-coding an import in ``clawcodex_ext/cli/``;
the registry only needs to import this module for the decorator to
fire.

Usage::

    clawcodex record --sources visualizer,cron --out demo.cast --duration 30s
    clawcodex record --sources orchestrator --out run.cast
    clawcodex record --list-sources
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from clawcodex_ext.cli.subcommand_registry import register

from extensions.capabilities.recorder import (
    AsciicastHeader,
    RecordableSource,
)
from extensions.recording.asciicast_writer import AsciicastWriter
from extensions.recording.config import RecordingConfig
from extensions.recording.registry import (
    get_default_registry,
    reset_default_registry,
)
from extensions.recording.validate_cast import validate_cast

__all__ = ["build_record_parser", "run_record_command"]

# `_factories` is imported lazily so that simply running ``clawcodex
# record`` populates the default registry with the well-known source
# factories (orchestrator, sop, visualizer, cron, query).
_FACTORIES_LOADED = False


def _ensure_factories_loaded() -> None:
    """Populate the default registry with built-in source factories.

    Idempotent. Importing :mod:`extensions.recording._factories` is
    what registers the built-in ``orchestrator``, ``sop``,
    ``visualizer``, ``cron``, and ``query`` source factories — but
    only if the parent :mod:`extensions.recording` package was loaded
    in the first place. Tests that build a private registry should
    call :func:`reset_default_registry` first.
    """
    global _FACTORIES_LOADED
    if _FACTORIES_LOADED:
        return
    _FACTORIES_LOADED = True
    # Importing this side-effect module triggers the
    # ``register_source(...)`` calls at module load.
    from extensions.recording import _factories  # noqa: F401


# ---------------------------------------------------------------------------
# Duration parser
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)
_DURATION_UNITS = {
    "": 1,
    "s": 1,
    "m": 60,
    "h": 3600,
}


def parse_duration(value: str) -> float:
    """Parse a human duration (``"30s"`` / ``"5m"`` / ``"1h"`` / ``"0"``).

    Returns the number of seconds. ``"0"`` or empty string means *no
    duration limit* — the CLI then runs until SIGINT.
    """
    if value is None:
        return 0.0
    raw = str(value).strip()
    if not raw or raw == "0":
        return 0.0
    match = _DURATION_RE.match(raw)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration {value!r}; expected e.g. 30s, 5m, 1h, 0")
    n = float(match.group(1))
    unit = match.group(2).lower()
    return n * _DURATION_UNITS[unit]


def _positive_finite_float(value: str) -> float:
    """argparse type: float, must be positive and finite (no NaN / inf)."""
    import math

    try:
        n = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}")
    if not math.isfinite(n):
        raise argparse.ArgumentTypeError(f"value must be a finite number, got {value!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive (> 0), got {value!r}")
    return n


def _non_negative_finite_float(value: str) -> float:
    """argparse type: float, must be non-negative and finite (no NaN / inf)."""
    import math

    try:
        n = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}")
    if not math.isfinite(n):
        raise argparse.ArgumentTypeError(f"value must be a finite number, got {value!r}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"value must be non-negative (>= 0), got {value!r}")
    return n


def _positive_int(value: str) -> int:
    """argparse type: int, must be >= 1."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"value must be >= 1, got {value!r}")
    return n


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_record_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``clawcodex record``."""
    p = argparse.ArgumentParser(
        prog="clawcodex record",
        description=(
            "Record one or more ClawCodex subsystems into an asciicast "
            "v2 .cast file. Runs until --duration elapses or Ctrl-C."
        ),
    )
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of source IDs to record "
            "(e.g. orchestrator,sop,visualizer,cron,query). "
            "Defaults to all registered sources when omitted."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .cast file path. Required unless --list-sources is set.",
    )
    p.add_argument(
        "--duration",
        type=parse_duration,
        default=0.0,
        help=("Auto-stop after N seconds. Accepts suffixes: 30s, 5m, 1h. 0 or empty runs until Ctrl-C. Default: 0."),
    )
    p.add_argument(
        "--width",
        type=int,
        default=120,
        help="Terminal width recorded in the .cast header (default: 120).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=36,
        help="Terminal height recorded in the .cast header (default: 36).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title written to the .cast header.",
    )
    p.add_argument(
        "--reset-registry",
        action="store_true",
        help=(
            "Drop the default source registry before recording. Test-only "
            "escape hatch; production runs should leave the registry alone."
        ),
    )
    p.add_argument(
        "--list-sources",
        action="store_true",
        help="Print registered source IDs and exit.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help=("After recording, run the self-contained validator on the output file and print any errors."),
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Run F-REC-AUTO: drive a real orchestrator batch + dashboard "
            "ticks into one .cast file. Mutually exclusive with --sources."
        ),
    )
    p.add_argument(
        "--auto-frame-delay-s",
        type=_positive_finite_float,
        default=1.0,
        help=(
            "Wall-clock seconds between dashboard ticks under --auto "
            "(default: %(default)s). Must be a positive finite number."
        ),
    )
    p.add_argument(
        "--auto-duration-s",
        type=_non_negative_finite_float,
        default=30.0,
        help=(
            "Total wall-clock budget for --auto in seconds (default: "
            "%(default)s). Must be a non-negative finite number."
        ),
    )
    p.add_argument(
        "--auto-issue-count",
        type=_positive_int,
        default=3,
        help=("Number of issues to dispatch under --auto, 1..3 (default: %(default)s)."),
    )
    # full-PTY capture. The structured mode (default) routes
    # per-subsystem events through the in-process source registry; pty
    # mode forks a real pseudo-terminal and captures the *entire screen*
    # — including prompt_toolkit's `❯` glyph, line editing, cursor
    # moves, and any Rich output — which the Rich tee cannot
    # capture because prompt_toolkit renders directly to the TTY, not
    # through Rich. The native backend uses only the standard-library
    # `pty` module; an optional `asciinema` backend is available for
    # users who already have the Rust CLI installed.
    p.add_argument(
        "--mode",
        type=str,
        choices=("structured", "pty"),
        default="structured",
        help=(
            "Recording mode. 'structured' (default) uses the source "
            "registry + per-subsystem adapters. 'pty' forks a real "
            "pseudo-terminal and records the full screen including the "
            "prompt_toolkit prompt bar and cursor."
        ),
    )
    p.add_argument(
        "--pty-cmd",
        type=str,
        default="clawcodex-dev",
        help=(
            "Under --mode pty, the command to run inside the recorded "
            "PTY (default: %(default)s). For interactive sessions use "
            "--no-pty-auto-exit; otherwise the command is executed and "
            "the PTY is closed automatically."
        ),
    )
    p.add_argument(
        "--pty-backend",
        type=str,
        choices=("native", "asciinema"),
        default="native",
        help=(
            "PTY recording backend. 'native' (default) uses the Python "
            "standard-library pty module and works everywhere. "
            "'asciinema' delegates to the external `asciinema` CLI; "
            "only the Rust build reliably captures command output in "
            "headless environments."
        ),
    )
    p.add_argument(
        "--pty-auto-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true (default), send an 'exit' keystroke after the "
            "--pty-cmd so the recording ends automatically. Use "
            "--no-pty-auto-exit for interactive sessions where you want "
            "to close the shell yourself."
        ),
    )
    p.add_argument(
        "--pty-capture-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Mirror typed input as asciicast 'i' frames (default: %(default)s). Native backend only."),
    )
    p.add_argument(
        "--pty-input-script",
        type=str,
        default="",
        help=(
            "Multi-line input to feed to the PTY after --pty-input-delay-s. "
            "Each line is sent as if the user had typed it, followed by "
            "Enter. If the value starts with '@', the rest is treated as a "
            "file path to read. Native backend only."
        ),
    )
    p.add_argument(
        "--pty-input-delay-s",
        type=_non_negative_finite_float,
        default=0.0,
        help=(
            "Seconds to wait after the PTY starts before sending the "
            "input script (default: %(default)s). Useful for REPLs "
            "that need time to render their splash screen before "
            "accepting commands. Native backend only."
        ),
    )
    p.add_argument(
        "--pty-quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --quiet to asciinema to suppress its own status "
            "messages (default: %(default)s). asciinema backend only."
        ),
    )
    p.add_argument(
        "--pty-overwrite",
        action="store_true",
        help=("Pass -y to asciinema so an existing output file is overwritten. asciinema backend only."),
    )
    return p


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def _resolve_sources(
    spec: str | None,
    registry_names: Sequence[str],
) -> list[str]:
    """Resolve the ``--sources`` string into a concrete list of IDs.

    ``None`` means *use all registered sources*. Empty string is an
    explicit user choice of *no sources* (we let it through so
    ``--list-sources`` style smoke checks stay deterministic).

    Unknown IDs are preserved in the output (in user-supplied order)
    so the caller can emit a precise ``unknown source(s): ...`` error
    rather than silently dropping the user's typo.
    """
    if spec is None:
        return list(registry_names)
    parts = [s.strip().lower() for s in spec.split(",")]
    parts = [s for s in parts if s]
    if not parts:
        return []
    # Preserve user order, deduplicate. ``registered`` is kept in
    # registry order for cases where the user passed ``None`` so the
    # .cast header ``command`` field stays predictable across runs.
    seen: set[str] = set()
    deduped: list[str] = []
    for name in parts:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _install_sigint_handler(stop_event: threading.Event) -> None:
    """Wire SIGINT into ``stop_event`` so the main loop can break out."""
    previous = signal.getsignal(signal.SIGINT)

    def _handler(signum: int, frame: object) -> None:  # noqa: ARG001
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        # SIGINT is unavailable on Windows / non-main threads; the
        # duration timer still gets us out cleanly.
        pass
    stop_event._previous_sigint = previous  # type: ignore[attr-defined]


def _restore_sigint_handler(stop_event: threading.Event) -> None:
    previous = getattr(stop_event, "_previous_sigint", None)
    if previous is None:
        return
    try:
        signal.signal(signal.SIGINT, previous)
    except (ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# full-PTY capture via external asciinema
# ---------------------------------------------------------------------------


_ASCIINEMA_INSTALL_HINT = (
    "`asciinema` not found on PATH. Install one of:\n"
    "  - Ubuntu/Debian:  sudo apt install asciinema\n"
    "  - pip (Python):   pip install asciinema\n"
    "  - cargo (Rust):   cargo install asciinema\n"
    "  - static binary:  https://github.com/asciinema/asciinema/releases"
)


def _run_pty_recording_asciinema(
    *,
    out_path: Path,
    pty_cmd: Sequence[str],
    title: str | None,
    quiet: bool,
    overwrite: bool,
    auto_exit: bool,
) -> int:
    """Drive ``asciinema rec`` to capture the full PTY screen.

    Optional full-PTY backend. asciinema 2.x writes asciicast v2 NDJSON
    directly, so the file can be re-validated by :func:`validate_cast`
    and converted to MP4 by the existing ``cast-to-mp4``
    post-processor. The .cast produced by this mode includes the
    prompt_toolkit prompt bar (``❯`` glyph, line edit, cursor moves) —
    which the Rich tee cannot capture because prompt_toolkit
    renders directly to the TTY, not via Rich.

    asciinema is treated as a soft dependency: a clear error message is
    printed when it is missing; users can fall back to the native
    ``pty`` backend (the default) which uses only the Python standard
    library.

    The command is sent to asciinema's interactive shell via stdin
    rather than using ``asciinema rec --command ...``. Some Python
    builds of asciinema 2.4.0 drop all output from ``--command`` mode
    (only the header is written), so feeding the command through the
    shell prompt works around that bug while preserving the full PTY
    screen capture.

    When ``auto_exit`` is true, an ``exit`` command is appended so the
    recording terminates automatically. For interactive sessions the
    user can pass ``--no-pty-auto-exit`` and end the recording by
    sending EOF (Ctrl-D) or typing ``exit`` in the inner shell.
    """
    if shutil.which("asciinema") is None:
        print(
            f"error: --pty-backend asciinema requires the `asciinema` CLI.\n"
            f"{_ASCIINEMA_INSTALL_HINT}\n"
            f"hint: use --pty-backend native (the default) for a "
            f"dependency-free PTY recorder.",
            file=sys.stderr,
        )
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv: list[str] = ["asciinema", "rec"]
    if quiet:
        argv.append("--quiet")
    if overwrite:
        argv.append("--yes")
    if title:
        argv.extend(["--title", title])
    argv.append(str(out_path))

    script_lines: list[str] = [shlex.join(list(pty_cmd))]
    if auto_exit:
        script_lines.append("exit")
    script = ("\n".join(script_lines) + "\n").encode("utf-8")

    print(
        f"[record/pty/asciinema] launching {' '.join(pty_cmd)}; output → {out_path}",
        file=sys.stderr,
    )
    if auto_exit:
        print(
            "[record/pty/asciinema] the inner command will run and the shell will exit automatically.",
            file=sys.stderr,
        )
    else:
        print(
            "[record/pty/asciinema] send EOF (Ctrl-D) in the inner shell, "
            "or Ctrl-C in this terminal, to end the recording.",
            file=sys.stderr,
        )

    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
    ) as proc:
        try:
            proc.communicate(input=script)
        except KeyboardInterrupt:
            # Ctrl-C at this terminal — ask asciinema to clean up.
            try:
                proc.terminate()
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()
        return proc.returncode


def _parse_pty_input_script(value: str) -> bytes:
    """Resolve ``--pty-input-script`` to raw input bytes.

    A leading ``@`` means "read this file"; otherwise the literal
    string is used, with ``\\n`` sequences normalised to real newlines.
    """
    if value.startswith("@"):
        path = Path(value[1:]).expanduser().resolve()
        return path.read_bytes()
    return value.replace("\\n", "\n").encode("utf-8")


def _run_pty_recording_native(
    *,
    out_path: Path,
    pty_cmd: Sequence[str],
    width: int,
    height: int,
    title: str | None,
    auto_exit: bool,
    capture_input: bool,
    input_delay_s: float,
    input_script_raw: str,
) -> int:
    """Native PTY backend for full-PTY capture (Python stdlib only).

    Forks the command into a pseudo-terminal and writes all terminal
    output as asciicast v2 ``"o"`` events. Input script bytes are sent
    to the PTY and optionally mirrored as ``"i"`` events. This backend
    works on any POSIX system with no external dependencies.
    """
    from extensions.recording.pty_recorder import run_pty_recording

    input_script = _parse_pty_input_script(input_script_raw)
    if auto_exit:
        input_script = input_script + b"\nexit\n"

    print(
        f"[record/pty/native] launching {' '.join(pty_cmd)} in PTY ({width}x{height}); output -> {out_path}",
        file=sys.stderr,
    )
    return run_pty_recording(
        cmd=pty_cmd,
        out_path=out_path,
        width=width,
        height=height,
        title=title,
        input_script=input_script or None,
        capture_input=capture_input,
        input_delay_s=input_delay_s,
    )


def run_record_command(args: list[str] | None = None) -> int:
    """Entry point invoked by ``subcommand_registry``."""
    parser = build_record_parser()
    parsed = parser.parse_args(args)

    if parsed.reset_registry:
        reset_default_registry()
    _ensure_factories_loaded()

    registry = get_default_registry()
    available = registry.names()

    if parsed.list_sources:
        print("registered sources:")
        if not available:
            print("  (none)")
        else:
            for name in available:
                print(f"  - {name}")
        return 0

    if parsed.out is None:
        parser.error("--out is required (or pass --list-sources)")

    out_path = Path(parsed.out).expanduser().resolve()

    # mutually exclusive with --sources. Drives a real
    # orchestrator batch + dashboard ticks into one .cast file.
    if parsed.auto:
        if parsed.sources:
            parser.error("--auto is mutually exclusive with --sources")
        if parsed.mode == "pty":
            parser.error("--auto is mutually exclusive with --mode pty")
        from extensions.recording.auto_demo import run as auto_run

        return asyncio.run(
            auto_run(
                out_path,
                duration_s=parsed.auto_duration_s,
                issue_count=parsed.auto_issue_count,
                frame_delay_s=parsed.auto_frame_delay_s,
            )
        )

    # full-PTY capture. Native backend (default) forks a real
    # pseudo-terminal via the Python standard library; optional
    # asciinema backend delegates to the external CLI. Mutually
    # exclusive with --sources and --auto. --list-sources is
    # intentionally allowed under --mode pty so the same registry view
    # is available for documentation / debugging.
    if parsed.mode == "pty":
        if parsed.sources:
            parser.error("--mode pty is mutually exclusive with --sources")
        if parsed.pty_backend == "native":
            rc = _run_pty_recording_native(
                out_path=out_path,
                pty_cmd=shlex.split(parsed.pty_cmd),
                width=parsed.width,
                height=parsed.height,
                title=parsed.title,
                auto_exit=parsed.pty_auto_exit,
                capture_input=parsed.pty_capture_input,
                input_delay_s=parsed.pty_input_delay_s,
                input_script_raw=parsed.pty_input_script,
            )
        else:
            rc = _run_pty_recording_asciinema(
                out_path=out_path,
                pty_cmd=shlex.split(parsed.pty_cmd),
                title=parsed.title,
                quiet=parsed.pty_quiet,
                overwrite=parsed.pty_overwrite,
                auto_exit=parsed.pty_auto_exit,
            )
        if rc != 0:
            return rc
        if parsed.validate:
            errors = validate_cast(out_path)
            if errors:
                print("[record/pty] validation errors:", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                return 1
            print("[record/pty] validation: OK", file=sys.stderr)
        return rc

    sources = _resolve_sources(parsed.sources, available)

    if not sources:
        print(
            "error: no sources selected. Use --sources or --list-sources.",
            file=sys.stderr,
        )
        return 2

    unknown = [s for s in sources if not registry.has(s)]
    if unknown:
        print(
            f"error: unknown source(s): {', '.join(unknown)}. Available: {', '.join(available) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    if parsed.width <= 0 or parsed.height <= 0:
        parser.error("--width and --height must be positive integers")

    header = AsciicastHeader(
        width=parsed.width,
        height=parsed.height,
        timestamp=int(time.time()),
        command=f"clawcodex record --sources {','.join(sources)}",
        title=parsed.title,
    )

    writer = AsciicastWriter(out_path, header)
    capture = writer.open()

    opened: list[tuple[str, RecordableSource]] = []
    failures: list[tuple[str, BaseException]] = []
    for source_id in sources:
        factory = registry.get(source_id)
        assert factory is not None
        try:
            source = factory(capture)
        except Exception as exc:  # pragma: no cover - adapter failure
            failures.append((source_id, exc))
            continue
        try:
            source.open(capture)
        except Exception as exc:  # pragma: no cover - adapter failure
            failures.append((source_id, exc))
            try:
                source.close()
            except Exception:  # nosec B110
                # A failed adapter open must not block cleanup of the writer.
                pass
            continue
        opened.append((source_id, source))

    if not opened:
        writer.close()
        print(
            "error: no source could be opened. " + "; ".join(f"{name}: {exc}" for name, exc in failures),
            file=sys.stderr,
        )
        return 1

    stop_event = threading.Event()
    deadline = time.monotonic() + parsed.duration if parsed.duration > 0 else None
    _install_sigint_handler(stop_event)

    print(
        f"[record] writing {out_path} ({len(opened)} source(s): {', '.join(name for name, _ in opened)})",
        file=sys.stderr,
    )
    if deadline is not None:
        print(
            f"[record] auto-stop in {parsed.duration:.1f}s (Ctrl-C to end early)",
            file=sys.stderr,
        )

    try:
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            # Sources tick on their own clock — the CLI is just the
            # lifetime owner of the writer. Sleep is short so Ctrl-C
            # latency stays low.
            stop_event.wait(timeout=0.25)
    finally:
        for _source_id, source in reversed(opened):
            try:
                source.close()
            except Exception:  # nosec B110
                # Cleanup must not raise; one misbehaving adapter
                # should not stop the others from unwinding.
                pass
        writer.close()
        _restore_sigint_handler(stop_event)

    if failures:
        for name, exc in failures:
            print(f"[record] source {name!r} failed: {exc}", file=sys.stderr)

    print(
        f"[record] done — {writer.frame_count} frame(s) written to {out_path}",
        file=sys.stderr,
    )

    if parsed.validate:
        errors = validate_cast(out_path)
        if errors:
            print("[record] validation errors:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print("[record] validation: OK", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# subcommand_registry hook
# ---------------------------------------------------------------------------


@register("record")
def _record_subcommand(args: list[str]) -> int:
    """``clawcodex record ...`` handler registered with ``subcommand_registry``."""
    return run_record_command(args)


# Re-export the default config so callers building custom CLIs (tests,
# docs examples) can read the knob set without instantiating RecordingConfig.
DEFAULT_CONFIG = RecordingConfig()
