"""Native PTY recorder.

Implements ``clawcodex record --mode pty`` without relying on the
external ``asciinema`` CLI. This is important because the Python
build of asciinema 2.4.0 drops command output in some headless /
WSL environments (only the .cast header is written), and the Rust
build is not guaranteed to be installed.

The recorder uses the standard-library :mod:`pty` module to fork a
child process attached to a pseudo-terminal, reads all terminal
output from the PTY master, and writes it as asciicast v2 ``"o"``
events. Optional input script bytes can be sent to the child's
stdin; each chunk is optionally mirrored as ``"i"`` events. When the
child exits, an ``"x"`` event is appended and the writer is closed.

This captures the full screen as a real human sees it: prompt_toolkit
prompt bar, line editing, cursor moves, Rich colors, and everything
else the application writes to the TTY.
"""

from __future__ import annotations

import json
import os
import select
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    import pty
    import termios
except ImportError:  # Windows has no POSIX terminal control modules.
    pty = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]


class _PtyRecorder:
    """Low-level PTY-to-cast recorder.

    Designed for one-shot use: construct, call :meth:`run`, then discard.
    """

    def __init__(
        self,
        *,
        cmd: Sequence[str],
        out_path: Path,
        width: int,
        height: int,
        title: str | None,
        input_script: bytes | None,
        capture_input: bool,
        env: dict[str, str] | None,
        input_delay_s: float = 0.0,
    ) -> None:
        self._cmd = list(cmd)
        self._out_path = out_path
        self._width = width
        self._height = height
        self._title = title
        self._input_script = input_script or b""
        self._capture_input = capture_input
        self._env = env if env is not None else os.environ.copy()
        self._input_delay_s = max(0.0, input_delay_s)

    def _build_header(self) -> bytes:
        header: dict[str, object] = {
            "version": 2,
            "width": self._width,
            "height": self._height,
            "timestamp": int(time.time()),
            "env": {"SHELL": self._env.get("SHELL", "/bin/bash"), "TERM": self._env.get("TERM", "xterm-256color")},
        }
        if self._title:
            header["title"] = self._title
        return json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"

    def run(self) -> int:
        """Fork the command, record its PTY output, and return its exit code."""
        if pty is None or termios is None or not hasattr(pty, "fork"):
            raise OSError("native PTY recording is unavailable on this platform")
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

        pid, master_fd = pty.fork()
        if pid == 0:
            # Child: set a sane TERM and exec the target command.
            self._env.setdefault("TERM", "xterm-256color")
            os.execvpe(self._cmd[0], self._cmd, self._env)
            # execvpe never returns on success.
            sys.exit(127)

        # Parent: configure the PTY master and record everything.
        try:
            os.set_blocking(master_fd, False)
            # Set the kernel-side terminal size so applications that
            # query TIOCGWINSZ (Rich, prompt_toolkit, curses, etc.)
            # get the requested dimensions.
            try:
                size = struct.pack("HHHH", self._height, self._width, 0, 0)
                fcntl = __import__("fcntl")
                fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size)
            except (OSError, ImportError):
                pass

            return self._record_loop(pid, master_fd)
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def _record_loop(self, pid: int, master_fd: int) -> int:
        start = time.monotonic()
        input_chunks: list[bytes] = []
        if self._input_script:
            # Split on newlines and send one chunk at a time. This gives
            # the shell a chance to render a prompt between commands.
            for line in self._input_script.split(b"\n"):
                line = line + b"\n"
                if line != b"\n":
                    input_chunks.append(line)
            # If the script ended with a trailing newline we may have an
            # empty chunk at the end; drop it.
            input_chunks = [c for c in input_chunks if c.strip() or c == b"\n"]

        output_events: list[tuple[float, str, bytes]] = []
        pending_input_index = 0
        last_input_time: float | None = None

        def _now() -> float:
            return time.monotonic() - start

        def elapsed() -> float:
            return _now()

        def _drain_output() -> None:
            while True:
                try:
                    chunk = os.read(master_fd, 8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not chunk:
                    break
                output_events.append((_now(), "o", chunk))

        def _send_next_input() -> None:
            nonlocal pending_input_index, last_input_time
            if pending_input_index >= len(input_chunks):
                return
            # Honor the initial delay so REPL splash screens have time
            # to render before we start typing.
            if _now() < self._input_delay_s:
                return
            # Rate-limit input slightly so the shell prompt has time to
            # appear between commands. The first chunk is sent immediately.
            if last_input_time is not None and _now() - last_input_time < 0.05:
                return
            chunk = input_chunks[pending_input_index]
            try:
                written = os.write(master_fd, chunk)
            except OSError:
                return
            last_input_time = _now()
            if self._capture_input:
                output_events.append((last_input_time, "i", chunk[:written]))
            pending_input_index += 1

        # SIGCHLD handler so we notice child exit promptly even when
        # select() is sleeping. We still poll select to drain output.
        child_exited = False

        def _sigchld_handler(signum: int, frame: object) -> None:  # noqa: ARG001
            nonlocal child_exited
            child_exited = True

        prev_sigchld = signal.signal(signal.SIGCHLD, _sigchld_handler)
        try:
            # Initial short wait for the shell / application to start.
            time.sleep(0.05)
            while not child_exited:
                _send_next_input()
                _drain_output()
                ready, _, _ = select.select([master_fd], [], [], 0.05)
                if master_fd in ready:
                    _drain_output()
                # Check child status non-blocking.
                waited_pid, status = os.waitpid(pid, os.WNOHANG)
                if waited_pid == pid:
                    child_exited = True
                    break

            # Drain any remaining output after the child exits.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                _drain_output()
                try:
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if master_fd not in ready:
                        break
                except (OSError, ValueError):
                    break

            exit_code = 0
            if child_exited:
                # status from the final waitpid may be unset if we broke
                # via the signal handler; collect it now.
                try:
                    _, status = os.waitpid(pid, 0)
                except ChildProcessError:
                    status = 0
                if os.WIFEXITED(status):
                    exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    exit_code = 128 + os.WTERMSIG(status)

            _drain_output()
        finally:
            signal.signal(signal.SIGCHLD, prev_sigchld)

        # Write the .cast file atomically.
        tmp_path = self._out_path.with_suffix(self._out_path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as f:
                f.write(self._build_header())
                for t, kind, payload in output_events:
                    line = json.dumps(
                        [round(t, 6), kind, payload.decode("utf-8", errors="replace")],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    f.write(line.encode("utf-8") + b"\n")
                if child_exited:
                    x_line = json.dumps([round(_now(), 6), "x", exit_code], separators=(",", ":"))
                    f.write(x_line.encode("utf-8") + b"\n")
            tmp_path.replace(self._out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return exit_code


def run_pty_recording(
    *,
    cmd: Sequence[str],
    out_path: Path,
    width: int = 120,
    height: int = 36,
    title: str | None = None,
    input_script: bytes | None = None,
    capture_input: bool = True,
    env: dict[str, str] | None = None,
    input_delay_s: float = 0.0,
) -> int:
    """Record ``cmd`` in a PTY and write an asciicast v2 ``.cast`` file.

    Args:
        cmd: Command argv to execute inside the PTY.
        out_path: Destination .cast file.
        width: Terminal width (also reported to the child via TIOCSWINSZ).
        height: Terminal height.
        title: Optional title for the .cast header.
        input_script: Bytes to feed to the PTY stdin after the child starts.
        capture_input: If true, mirror ``input_script`` chunks as ``"i"`` events.
        env: Environment for the child. ``TERM`` is set to ``xterm-256color``
            if absent.
        input_delay_s: Seconds to wait before sending the first input chunk.
            Useful for REPLs that render a splash screen before accepting
            commands.

    Returns:
        The child's exit code.
    """
    recorder = _PtyRecorder(
        cmd=cmd,
        out_path=out_path,
        width=width,
        height=height,
        title=title,
        input_script=input_script,
        capture_input=capture_input,
        env=env,
        input_delay_s=input_delay_s,
    )
    return recorder.run()


__all__ = ["run_pty_recording"]
