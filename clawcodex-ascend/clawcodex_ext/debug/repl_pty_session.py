# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Interactive PTY controller for debugging the real ClawCodex REPL."""

from __future__ import annotations
# pylint: disable=C0302,E1101

import argparse
import json
import os
import re
import select
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TextIO


ANSI_RE = re.compile(r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-_])")
READY_MARKER_RE = r"CLAWCODEX_AGENT_DEBUG::repl\.ready::[^\r\n]*(?:\r?\n)?"
PROVIDER_ERROR_RE = re.compile(
    r"("
    r"ProviderError|API\s*Error|AuthenticationError|RateLimitError|"
    r"invalid[_ -]?api[_ -]?key|API key .*not configured|litellm\.exceptions"
    r")",
    re.IGNORECASE,
)
NETWORK_ERROR_RE = re.compile(
    r"("
    r"NetworkError|Connection\s*Error|ConnectTimeout|ReadTimeout|"
    r"NameResolutionError|DNS lookup|Temporary failure in name resolution|"
    r"nodename nor servname|SSL:"
    r")",
    re.IGNORECASE,
)
STREAMING_RE = re.compile(
    r"(Thinking|Streaming|Waiting for model|[\u2800-\u28ff])",
    re.IGNORECASE,
)
CPR_WARNING_RE = re.compile(
    r"WARNING:\s+your terminal doesn't support cursor position requests \(CPR\)\.?",
    re.IGNORECASE,
)
THINKING_STATUS_RE = re.compile(
    r"[\u2800-\u28ff]?\s*Thinking[^\r\n]*",
    re.IGNORECASE,
)
STATUS_TOOLBAR_RE = re.compile(
    r".*\s·\s.*\s·\s.*\bmode:\s*.*\btokens:\s*\d+(?:\.\d+)?[kKmM]?\s+in\s*/\s*\d+(?:\.\d+)?[kKmM]?\s+out.*"
)
PERMISSION_PROMPT_RE = re.compile(
    r"Permission Required.*?(Allow\?|allow this action|Enter select|quick select)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Observation:
    ok: bool
    event: str
    delta: str
    screen: str
    artifact_dir: str
    step: int
    error: str | None = None
    kind: str = "unknown"
    state: str = "unknown"
    error_kind: str | None = None
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationClassification:
    kind: str
    state: str
    signals: tuple[str, ...]
    error_kind: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Step:
    label: str
    send: str
    expect: str
    timeout: float = 30.0


@dataclass(frozen=True)
class ArtifactResult:
    ok: bool
    artifact_dir: Path
    raw_log: Path
    clean_transcript: Path
    result_json: Path


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_repl_text(text: str) -> str:
    """Return transcript text intended for assertions and human review.

    ``raw.log`` remains the byte-faithful terminal record. This cleaned surface
    removes redraw/status noise that otherwise overwhelms result JSON and
    `clean.txt` while preserving assistant, command, and tool output.
    """

    text = strip_ansi(text)
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = CPR_WARNING_RE.sub("", raw_line)
        line = THINKING_STATUS_RE.sub("", line)
        line = re.sub(r"[\u2800-\u28ff]", "", line)
        if STATUS_TOOLBAR_RE.match(line.strip()):
            continue
        if not line.strip() and raw_line.strip():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _compact_for_expect(text: str) -> str:
    text = clean_repl_text(text)
    text = re.sub(r"[\u2800-\u28ff]", "", text)
    text = re.sub(r"Thinking[^\r\n]*", "", text)
    text = text.replace("❯", "")
    return re.sub(r"\s+", "", text)


def _matches_expected_text(expected: str, rendered_text: str) -> bool:
    if expected in rendered_text:
        return True
    compact_expected = _compact_for_expect(expected)
    if not compact_expected:
        return True
    return compact_expected in _compact_for_expect(rendered_text)


def _has_non_echo_output(delta: str, sent_text: str | None) -> bool:
    if sent_text is None:
        return bool(strip_ansi(delta).strip())
    for line in strip_ansi(delta).replace("\r", "\n").split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        if candidate == sent_text:
            continue
        if candidate.endswith(sent_text) and candidate[: -len(sent_text)].strip() in {">", "❯"}:
            continue
        return True
    return False


def _has_initial_terminal_echo(delta: str, sent_text: str | None) -> bool:
    if sent_text is None:
        return False
    for line in delta.replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        return _line_is_terminal_echo(line, sent_text)
    return False


def _line_is_terminal_echo(line: str, sent_text: str) -> bool:
    candidate = strip_ansi(line).strip()
    if not candidate:
        return False
    if candidate == sent_text:
        return True
    return candidate.endswith(sent_text) and candidate[: -len(sent_text)].strip() in {
        ">",
        "❯",
    }


def _without_initial_terminal_echo(delta: str, sent_text: str | None) -> str:
    if sent_text is None:
        return delta
    lines = delta.replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if _line_is_terminal_echo(line, sent_text):
            return "\n".join(lines[index + 1 :])
        compact_sent = re.sub(r"\s+", "", sent_text)
        compact_echo = ""
        for echo_end in range(index, len(lines)):
            candidate = strip_ansi(lines[echo_end]).strip()
            if echo_end == index and candidate.startswith("❯"):
                candidate = candidate[1:].strip()
            if echo_end == index and candidate.startswith(">"):
                candidate = candidate[1:].strip()
            if not candidate:
                continue
            compact_echo += re.sub(r"\s+", "", candidate)
            if not compact_sent.startswith(compact_echo):
                return delta
            if compact_echo == compact_sent:
                return "\n".join(lines[echo_end + 1 :])
        if compact_echo and compact_sent.startswith(compact_echo):
            return ""
        return delta
    return ""


def _first_non_empty_line(text: str) -> str | None:
    for line in strip_ansi(text).replace("\r", "\n").split("\n"):
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def _detect_rendered_error(text: str) -> tuple[str, str] | None:
    clean = strip_ansi(text)
    for line in clean.replace("\r", "\n").split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        error_like = candidate.lower().startswith(
            (
                "query error:",
                "providererror:",
                "networkerror:",
                "authenticationerror:",
                "ratelimiterror:",
                "api error:",
                "openai.",
                "httpcore.",
                "litellm.",
            )
        )
        if not error_like and candidate.lower() != "connection error.":
            continue
        if NETWORK_ERROR_RE.search(candidate):
            return "network_error", candidate
        if PROVIDER_ERROR_RE.search(candidate):
            return "provider_error", candidate
    return None


def _has_permission_prompt(text: str) -> bool:
    clean = strip_ansi(text)
    matches = list(PERMISSION_PROMPT_RE.finditer(clean))
    if not matches:
        return False

    # Terminal redraws often keep the old permission menu visible in the same
    # delta that also contains the approved tool result and a fresh prompt.
    # Classify only an active permission prompt as awaiting input.
    tail = clean[matches[-1].end() :]
    if "Tool result:" in tail or "Tool error:" in tail or "⎿" in tail or "Goodbye!" in tail:
        return False
    return True


def _classify_observation(
    *,
    delta: str,
    sent_text: str | None,
    ok: bool,
    event: str,
    error: str | None,
) -> ObservationClassification:
    clean_delta = strip_ansi(delta)
    non_echo_delta = strip_ansi(_without_initial_terminal_echo(delta, sent_text))
    signals: list[str] = []

    if _has_initial_terminal_echo(delta, sent_text):
        signals.append("input_echo")
    if STREAMING_RE.search(clean_delta):
        signals.append("streaming")
    if "CLAWCODEX_AGENT_DEBUG::repl.ready::" in clean_delta:
        signals.append("ready_marker")
    if "❯" in delta or re.search(r"(^|\n)\s*[>❯]\s*$", clean_delta):
        signals.append("prompt")

    if event == "ready":
        if "ready_marker" not in signals and "prompt" in signals:
            signals.append("prompt_ready_fallback")
        return ObservationClassification("ready", "ready", tuple(signals or ["ready_marker"]))
    if event == "stopped":
        return ObservationClassification("stopped", "stopped", tuple(signals))

    if _has_permission_prompt(non_echo_delta or clean_delta):
        signals.append("permission_prompt")
        return ObservationClassification(
            "permission_prompt",
            "awaiting_permission",
            tuple(dict.fromkeys(signals)),
        )

    rendered_error = _detect_rendered_error(non_echo_delta or clean_delta)
    if rendered_error is not None:
        error_kind, summary = rendered_error
        signals.append(error_kind)
        return ObservationClassification(
            kind=error_kind,
            state="error",
            signals=tuple(dict.fromkeys(signals)),
            error_kind=error_kind,
            error=f"{error_kind}: {summary}",
        )

    if error is not None or not ok:
        error_kind = "timeout" if error and "Timeout exceeded" in error else "controller_error"
        signals.append(error_kind)
        return ObservationClassification(
            kind=error_kind,
            state="error",
            signals=tuple(dict.fromkeys(signals)),
            error_kind=error_kind,
        )

    has_non_echo = _has_non_echo_output(delta, sent_text)
    if sent_text and sent_text.lstrip().startswith("/") and has_non_echo:
        signals.append("slash_command")
        return ObservationClassification("slash_command", "idle", tuple(dict.fromkeys(signals)))
    if has_non_echo:
        signals.append("assistant_output")
        state = "streaming" if "streaming" in signals and "prompt" not in signals else "idle"
        return ObservationClassification("assistant_output", state, tuple(dict.fromkeys(signals)))
    if "input_echo" in signals:
        return ObservationClassification("input_echo", "prompt", tuple(dict.fromkeys(signals)))

    line = _first_non_empty_line(clean_delta)
    if line:
        signals.append("terminal_output")
        return ObservationClassification("terminal_output", "idle", tuple(dict.fromkeys(signals)))

    return ObservationClassification("prompt", "prompt", tuple(dict.fromkeys(signals)))


def default_repl_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "clawcodex_ext.cli.main",
        "--legacy-repl",
        "--stream",
        "--agent-debug",
        "--permission-mode",
        "bypassPermissions",
    ]


def default_artifact_dir(root: Path | None = None) -> Path:
    base = root or Path(tempfile.gettempdir()) / "clawcodex-repl-pty"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / stamp


def build_goal_script(*, goal: str, prompt: str, expect_response: str) -> list[Step]:
    return [
        Step(label="start goal", send=f"/goal {goal}", expect="Goal set:"),
        Step(label="send prompt", send=prompt, expect=expect_response),
        Step(label="inspect goal", send="/goal", expect="Tokens:"),
        Step(label="clear goal", send="/goal clear", expect="Goal cleared"),
        Step(label="exit", send="/exit", expect="Goodbye!"),
    ]


def build_file_creation_prompt(*, target_path: str, task: str) -> str:
    return (
        f"{task}\n\n"
        f"create or overwrite the file at `{target_path}`. "
        "Use the available file-writing tool or shell command to actually create the file; "
        "do not only print code in the chat response. "
        "After writing, verify the file exists and briefly report the path."
    )


def write_artifacts(
    *,
    artifact_dir: Path,
    raw_text: str,
    command: Sequence[str],
    events: list[dict[str, object]],
    ok: bool,
    error: str | None,
) -> ArtifactResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_log = artifact_dir / "raw.log"
    clean_transcript = artifact_dir / "clean.txt"
    result_json = artifact_dir / "result.json"

    raw_log.write_text(raw_text, encoding="utf-8")
    clean_transcript.write_text(clean_repl_text(raw_text), encoding="utf-8")
    result_json.write_text(
        json.dumps(
            {
                "artifacts": {
                    "clean_transcript": str(clean_transcript),
                    "raw_log": str(raw_log),
                    "result_json": str(result_json),
                },
                "command": list(command),
                "error": error,
                "events": events,
                "ok": ok,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ArtifactResult(ok, artifact_dir, raw_log, clean_transcript, result_json)


class ReplPtySession:
    def __init__(
        self,
        *,
        command: Sequence[str],
        artifact_dir: Path,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.command = list(command)
        self.artifact_dir = artifact_dir
        self.timeout = timeout
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        self._child = None
        self._raw_chunks: list[str] = []
        self._events: list[dict[str, object]] = []
        self._step = 0

    def start(self) -> Observation:
        try:
            import pexpect
        except ModuleNotFoundError as exc:
            raise SystemExit("pexpect is required. Run with `uv run --extra dev --frozen ...`.") from exc

        from clawcodex_ext.debug.agent_debug import apply_agent_debug_environment

        child_env = os.environ.copy()
        child_env.update(self.env)
        debug_dir_raw = str(child_env.get("CLAWCODEX_AGENT_DEBUG_DIR", "")).strip()
        debug_dir = Path(debug_dir_raw).expanduser() if debug_dir_raw else self.artifact_dir / "state"
        apply_agent_debug_environment(child_env, debug_dir=debug_dir)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._child = pexpect.spawn(
            self.command[0],
            self.command[1:],
            dimensions=(40, 120),
            encoding="utf-8",
            env=child_env,
            timeout=self.timeout,
        )
        return self._expect_regex(READY_MARKER_RE, event="ready")

    def send(
        self,
        text: str,
        *,
        expect: str | None = None,
        timeout: float | None = None,
    ) -> Observation:
        before_len = len(self.raw_text)
        self._write_child_input(text, newline=True, timeout=timeout)
        if expect:
            return self._expect_exact(
                expect,
                timeout=timeout,
                event="observed",
                ignored_initial_echo=text,
                before_len=before_len,
            )
        return self._observe(timeout=timeout, ignored_initial_echo=text, before_len=before_len)

    def key(self, text: str, *, timeout: float | None = None) -> Observation:
        before_len = len(self.raw_text)
        self._write_child_input(text, newline=False, timeout=timeout)
        return self._observe(timeout=timeout, ignored_initial_echo=text, before_len=before_len)

    def _write_child_input(
        self,
        text: str,
        *,
        newline: bool,
        timeout: float | None,
    ) -> None:
        child = self._require_child()
        line_suffix = getattr(child, "linesep", "\n") if newline else ""
        payload_text = text + (line_suffix if isinstance(line_suffix, str) else "\n")
        encoding = getattr(child, "encoding", None) or "utf-8"
        payload = payload_text.encode(encoding, errors="surrogateescape")
        total_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + total_timeout
        offset = 0
        chunk_size = 512
        fd = child.child_fd
        restore_blocking: bool | None = None
        try:
            try:
                restore_blocking = os.get_blocking(fd)
                os.set_blocking(fd, False)
            except (AttributeError, OSError):
                restore_blocking = None
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out writing {len(payload)} bytes to child PTY")
                read_fds = [fd] if hasattr(child, "read_nonblocking") else []
                readable, ready, _ = select.select(read_fds, [fd], [], min(0.2, remaining))
                if readable:
                    self._drain_available(max_wait=0.01)
                if not ready:
                    continue
                try:
                    written = os.write(fd, payload[offset : offset + chunk_size])
                except BlockingIOError:
                    continue
                if written == 0:
                    raise OSError("child PTY write returned 0 bytes")
                offset += written
            if hasattr(child, "read_nonblocking"):
                self._drain_available(max_wait=0.05)
        finally:
            if restore_blocking is not None:
                try:
                    os.set_blocking(fd, restore_blocking)
                except OSError:
                    pass

    def expect(
        self,
        pattern: str,
        *,
        timeout: float | None = None,
        event: str = "observed",
        ignored_initial_echo: str | None = None,
    ) -> Observation:
        return self._expect_exact(
            pattern,
            timeout=timeout,
            event=event,
            ignored_initial_echo=ignored_initial_echo,
            before_len=None,
        )

    def observe(self, *, timeout: float | None = None) -> Observation:
        return self._observe(timeout=timeout, ignored_initial_echo=None, before_len=None)

    def _observe(
        self,
        *,
        timeout: float | None,
        ignored_initial_echo: str | None,
        before_len: int | None,
    ) -> Observation:
        child = self._require_child()
        if before_len is None:
            before_len = len(self.raw_text)
        total_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + total_timeout
        idle_deadline: float | None = None
        saw_substantive_output = False

        while time.monotonic() < deadline:
            read_timeout = 0.05
            if idle_deadline is not None:
                read_timeout = max(0.0, min(read_timeout, idle_deadline - time.monotonic()))
            try:
                chunk = child.read_nonblocking(size=4096, timeout=read_timeout)
            except Exception:
                if hasattr(child, "isalive") and not child.isalive():
                    break
                if saw_substantive_output and idle_deadline is not None:
                    if time.monotonic() >= idle_deadline:
                        break
                    continue
                if not saw_substantive_output:
                    time.sleep(0.01)
                    continue
                break
            if not chunk:
                if saw_substantive_output:
                    break
                continue
            self._collect(chunk)
            delta = self.raw_text[before_len:]
            if _has_non_echo_output(delta, ignored_initial_echo):
                saw_substantive_output = True
                idle_deadline = time.monotonic() + min(0.2, total_timeout)

        return self._observation(
            True,
            "observed",
            before_len,
            sent_text=ignored_initial_echo,
        )

    def stop(self) -> Observation:
        before_len = len(self.raw_text)
        child = self._child
        if child is not None and child.isalive():
            try:
                self._write_child_input("/exit", newline=True, timeout=min(2.0, self.timeout))
                child.expect_exact("Goodbye!", timeout=2)
                self._collect(str(child.before))
                self._collect(str(child.after))
            except Exception:
                child.close(force=True)
        self._child = None
        return self._observation(True, "stopped", before_len)

    @property
    def raw_text(self) -> str:
        return "".join(self._raw_chunks)

    @property
    def screen(self) -> str:
        return strip_ansi(self.raw_text)

    def write_artifacts(self, *, ok: bool, error: str | None = None) -> ArtifactResult:
        return write_artifacts(
            artifact_dir=self.artifact_dir,
            command=self.command,
            error=error,
            events=self._events,
            ok=ok,
            raw_text=self.raw_text,
        )

    def _require_child(self):
        if self._child is None:
            raise RuntimeError("REPL PTY session has not been started")
        return self._child

    def _expect_exact(
        self,
        pattern: str,
        *,
        timeout: float | None,
        event: str,
        ignored_initial_echo: str | None,
        before_len: int | None = None,
    ) -> Observation:
        child = self._require_child()
        if before_len is None:
            before_len = len(self.raw_text)
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            try:
                chunk = child.read_nonblocking(size=4096, timeout=0.05)
            except Exception:
                if hasattr(child, "isalive") and not child.isalive():
                    break
                time.sleep(0.01)
                continue
            self._collect(str(chunk))
            clean_delta = strip_ansi(
                _without_initial_terminal_echo(
                    self.raw_text[before_len:],
                    ignored_initial_echo,
                )
            )
            rendered_error = _detect_rendered_error(clean_delta)
            if rendered_error is not None:
                error_kind, summary = rendered_error
                return self._observation(
                    False,
                    "error",
                    before_len,
                    f"{error_kind}: {summary}",
                    sent_text=ignored_initial_echo,
                )
            if _matches_expected_text(pattern, clean_delta):
                self._drain_available()
                return self._observation(
                    True,
                    event,
                    before_len,
                    sent_text=ignored_initial_echo,
                )
        clean_delta = strip_ansi(
            _without_initial_terminal_echo(
                self.raw_text[before_len:],
                ignored_initial_echo,
            )
        )
        if _matches_expected_text(pattern, clean_delta):
            self._drain_available()
            return self._observation(
                True,
                event,
                before_len,
                sent_text=ignored_initial_echo,
            )
        error = f"Timeout exceeded while waiting for {pattern!r}"
        return self._observation(
            False,
            "error",
            before_len,
            error,
            sent_text=ignored_initial_echo,
        )

    def _expect_regex(self, pattern: str, *, event: str) -> Observation:
        child = self._require_child()
        before_len = len(self.raw_text)
        try:
            child.expect(pattern, timeout=self.timeout)
            self._collect(str(child.before))
            self._collect(str(child.after))
            self._drain_available()
            return self._observation(True, event, before_len)
        except Exception as exc:
            self._collect(str(getattr(child, "before", "")))
            if event == "ready" and self._has_prompt_ready(self.raw_text[before_len:]):
                self._drain_available()
                return self._observation(True, event, before_len)
            return self._observation(False, "error", before_len, f"{type(exc).__name__}: {exc}")

    def _collect(self, text: str) -> None:
        if text:
            self._raw_chunks.append(text)

    @staticmethod
    def _has_prompt_ready(text: str) -> bool:
        clean = strip_ansi(text)
        return "❯" in clean or bool(re.search(r"(^|\n)\s*[>❯]\s*$", clean))

    def _drain_available(self, *, max_wait: float = 0.2) -> None:
        child = self._require_child()
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            try:
                chunk = child.read_nonblocking(size=4096, timeout=0.02)
            except Exception:
                if hasattr(child, "isalive") and not child.isalive():
                    break
                time.sleep(0.01)
                continue
            if not chunk:
                break
            self._collect(str(chunk))

    def _observation(
        self,
        ok: bool,
        event: str,
        before_len: int,
        error: str | None = None,
        sent_text: str | None = None,
    ) -> Observation:
        self._step += 1
        delta = self.raw_text[before_len:]
        classification = _classify_observation(
            delta=delta,
            sent_text=sent_text,
            ok=ok,
            event=event,
            error=error,
        )
        final_error = error or classification.error
        final_ok = ok and classification.state != "error"
        final_event = "error" if classification.state == "error" else event
        obs = Observation(
            ok=final_ok,
            event=final_event,
            delta=clean_repl_text(delta),
            screen=clean_repl_text(self.raw_text),
            artifact_dir=str(self.artifact_dir),
            step=self._step,
            error=final_error,
            kind=classification.kind,
            state=classification.state,
            error_kind=classification.error_kind,
            signals=classification.signals,
        )
        self._events.append(asdict(obs))
        return obs


def _write_json(stdout: TextIO, payload: Mapping[str, object]) -> None:
    stdout.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
    stdout.flush()


def _request_echo_metadata(
    request: Mapping[str, object],
    *,
    op_override: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    op = op_override or request.get("op")
    if isinstance(op, str) and op:
        metadata["op"] = op
    label = request.get("label")
    if isinstance(label, str) and label.strip():
        metadata["label"] = label.strip()
    if op == "send":
        metadata["input_source"] = "text_file" if isinstance(request.get("text_file"), str) else "text"
    elif op in {"key", "raw"}:
        metadata["input_source"] = "raw"
    if request.get("allow_error") is True:
        metadata["allow_error"] = True
    return metadata


def _annotate_last_event(
    session: ReplPtySession,
    request: Mapping[str, object],
    *,
    op_override: str | None = None,
) -> None:
    if not session._events:
        return
    session._events[-1].update(_request_echo_metadata(request, op_override=op_override))


def _max_output_chars_from_request(request: Mapping[str, object]) -> int | None:
    value = request.get("max_output_chars")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _request_allows_error(request: Mapping[str, object]) -> bool:
    return request.get("allow_error") is True


def _record_observation_result(
    *,
    request: Mapping[str, object],
    obs: Observation,
    session_ok: bool,
    session_error: str | None,
    controller_ok: bool,
) -> tuple[bool, str | None, bool]:
    if obs.ok or _request_allows_error(request):
        return session_ok, session_error, controller_ok
    return False, obs.error or session_error, False


def _truncate_output_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        head = text[:max_chars]
        tail = ""
    else:
        head_chars = (max_chars + 1) // 2
        tail_chars = max_chars - head_chars
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars else ""
    omitted = len(text) - max_chars
    return head + f"\n...[truncated {omitted} chars; full transcript in artifacts]" + (f"\n{tail}" if tail else "")


def _observation_payload(
    obs: Observation,
    request: Mapping[str, object],
    *,
    op_override: str | None = None,
) -> Mapping[str, object]:
    payload = asdict(obs)
    payload.update(_request_echo_metadata(request, op_override=op_override))
    max_chars = _max_output_chars_from_request(request)
    if max_chars is None:
        return payload
    truncated_fields: dict[str, int] = {}
    for field in ("delta", "screen"):
        value = payload.get(field)
        if isinstance(value, str) and len(value) > max_chars:
            truncated_fields[field] = len(value)
            payload[field] = _truncate_output_text(value, max_chars)
    if truncated_fields:
        payload["truncated_fields"] = truncated_fields
    return payload


def _fold_repl_input_text(text: str) -> str:
    return re.sub(r"\s*\r?\n\s*", " ", text).strip()


def _send_text_from_request(request: Mapping[str, object]) -> str | None:
    text = request.get("text")
    if isinstance(text, str):
        return _fold_repl_input_text(text)
    text_file = request.get("text_file")
    if isinstance(text_file, str):
        file_text = Path(text_file).read_text(encoding="utf-8")
        return _fold_repl_input_text(file_text)
    return None


def run_script(
    *,
    command: Sequence[str],
    steps: Iterable[Step],
    artifact_dir: Path,
    timeout: float,
) -> ArtifactResult:
    session = ReplPtySession(command=command, artifact_dir=artifact_dir, timeout=timeout)
    ok = True
    error = None
    ready = session.start()
    if not ready.ok:
        ok = False
        error = ready.error
    for step in steps:
        if not ok:
            break
        observed = session.send(step.send, expect=step.expect, timeout=step.timeout)
        if not observed.ok:
            ok = False
            error = observed.error
            break
    if ok:
        session.stop()
    return session.write_artifacts(ok=ok, error=error)


def run_interactive_jsonl(
    *,
    stdin: TextIO,
    stdout: TextIO,
    artifact_dir: Path,
    timeout: float,
) -> int:
    session: ReplPtySession | None = None
    controller_ok = True
    session_ok = True
    session_error = None
    start_count = 0
    try:
        for raw_line in stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                op = request.get("op")
                if op == "start":
                    command = request.get("cmd") or default_repl_command()
                    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
                        _write_json(
                            stdout,
                            {
                                "error": "start.cmd must be a list of strings",
                                "event": "error",
                                "ok": False,
                            },
                        )
                        controller_ok = False
                        continue
                    if session is not None:
                        obs = session.stop()
                        _annotate_last_event(session, request, op_override="auto_stop")
                        _write_json(
                            stdout,
                            _observation_payload(obs, request, op_override="auto_stop"),
                        )
                        session.write_artifacts(ok=session_ok, error=session_error)
                        session = None
                    env = request.get("env")
                    current_artifact_dir = artifact_dir
                    if start_count > 0:
                        current_artifact_dir = artifact_dir / f"session-{start_count + 1}"
                    start_count += 1
                    session_ok = True
                    session_error = None
                    session = ReplPtySession(
                        command=command,
                        artifact_dir=current_artifact_dir,
                        env=env if isinstance(env, dict) else None,
                        timeout=float(request.get("timeout", timeout)),
                    )
                    obs = session.start()
                    _annotate_last_event(session, request)
                    _write_json(stdout, _observation_payload(obs, request))
                    session_ok, session_error, controller_ok = _record_observation_result(
                        request=request,
                        obs=obs,
                        session_ok=session_ok,
                        session_error=session_error,
                        controller_ok=controller_ok,
                    )
                    continue
                if op in {"exit", "quit"}:
                    if session is not None:
                        obs = session.stop()
                        _annotate_last_event(session, request, op_override="stop")
                        _write_json(stdout, _observation_payload(obs, request))
                        session.write_artifacts(ok=session_ok, error=session_error)
                        session = None
                    _write_json(
                        stdout,
                        {
                            "event": "controller_exit",
                            "ok": True,
                        },
                    )
                    return 0 if controller_ok else 1
                if session is None:
                    _write_json(
                        stdout,
                        {
                            "error": "session has not been started",
                            "event": "error",
                            "ok": False,
                        },
                    )
                    controller_ok = False
                    continue
                if op == "send":
                    text = _send_text_from_request(request)
                    if text is None:
                        _write_json(
                            stdout,
                            {
                                "error": ("send.text must be a string, or send.text_file must be a path string"),
                                "event": "error",
                                "ok": False,
                            },
                        )
                        session_ok = False
                        controller_ok = False
                        continue
                    expect = request.get("expect")
                    obs = session.send(
                        text,
                        expect=expect if isinstance(expect, str) else None,
                        timeout=float(request.get("timeout", timeout)),
                    )
                    _annotate_last_event(session, request)
                    _write_json(stdout, _observation_payload(obs, request))
                    session_ok, session_error, controller_ok = _record_observation_result(
                        request=request,
                        obs=obs,
                        session_ok=session_ok,
                        session_error=session_error,
                        controller_ok=controller_ok,
                    )
                    continue
                if op in {"key", "raw"}:
                    text = request.get("text")
                    if not isinstance(text, str):
                        _write_json(
                            stdout,
                            {
                                "error": f"{op}.text must be a string",
                                "event": "error",
                                "ok": False,
                            },
                        )
                        session_ok = False
                        controller_ok = False
                        continue
                    obs = session.key(
                        text,
                        timeout=float(request.get("timeout", timeout)),
                    )
                    _annotate_last_event(session, request)
                    _write_json(stdout, _observation_payload(obs, request))
                    session_ok, session_error, controller_ok = _record_observation_result(
                        request=request,
                        obs=obs,
                        session_ok=session_ok,
                        session_error=session_error,
                        controller_ok=controller_ok,
                    )
                    continue
                if op == "observe":
                    obs = session.observe(timeout=float(request.get("timeout", timeout)))
                    _annotate_last_event(session, request)
                    _write_json(stdout, _observation_payload(obs, request))
                    session_ok, session_error, controller_ok = _record_observation_result(
                        request=request,
                        obs=obs,
                        session_ok=session_ok,
                        session_error=session_error,
                        controller_ok=controller_ok,
                    )
                    continue
                if op == "stop":
                    obs = session.stop()
                    _annotate_last_event(session, request)
                    _write_json(stdout, _observation_payload(obs, request))
                    session_ok, session_error, controller_ok = _record_observation_result(
                        request=request,
                        obs=obs,
                        session_ok=session_ok,
                        session_error=session_error,
                        controller_ok=controller_ok,
                    )
                    session.write_artifacts(ok=session_ok, error=session_error)
                    session = None
                    continue
                _write_json(
                    stdout,
                    {
                        "error": f"unknown op: {op!r}",
                        "event": "error",
                        "ok": False,
                    },
                )
                controller_ok = False
            except Exception as exc:
                _write_json(
                    stdout,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "event": "error",
                        "ok": False,
                    },
                )
                controller_ok = False
    except KeyboardInterrupt:
        controller_ok = False
        error = "KeyboardInterrupt"
        _write_json(stdout, {"error": error, "event": "error", "ok": False})
        if session is not None:
            try:
                cleanup_request = {
                    "op": "interrupt_stop",
                    "label": "cleanup after KeyboardInterrupt",
                }
                obs = session.stop()
                _annotate_last_event(
                    session,
                    cleanup_request,
                    op_override="interrupt_stop",
                )
                _write_json(
                    stdout,
                    _observation_payload(
                        obs,
                        cleanup_request,
                        op_override="interrupt_stop",
                    ),
                )
            except Exception as exc:
                _write_json(
                    stdout,
                    {
                        "error": (f"cleanup failed after KeyboardInterrupt: {type(exc).__name__}: {exc}"),
                        "event": "error",
                        "ok": False,
                    },
                )
            session.write_artifacts(ok=False, error=error)
        return 130
    if session is not None:
        session.stop()
        session.write_artifacts(ok=session_ok, error=session_error)
    return 0 if controller_ok else 1


def run_adaptive_jsonl(
    *,
    first_request: Mapping[str, object],
    decide_next: Callable[[dict[str, object]], Mapping[str, object] | None],
    stdout: TextIO,
    artifact_dir: Path,
    timeout: float,
    max_turns: int = 100,
) -> int:
    """Run the JSONL controller with each next op chosen from the last response."""

    pending: list[dict[str, object]] = [dict(first_request)]
    queued_count = 1

    class AdaptiveStdin:
        def __iter__(self) -> "AdaptiveStdin":
            return self

        def __next__(self) -> str:
            if not pending:
                raise StopIteration
            return json.dumps(pending.pop(0), ensure_ascii=False) + "\n"

    class AdaptiveStdout:
        def __init__(self) -> None:
            self._buffer = ""

        def write(self, text: str) -> int:
            nonlocal queued_count
            stdout.write(text)
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                request = decide_next(payload)
                if request is None:
                    continue
                queued_count += 1
                if queued_count > max_turns:
                    raise RuntimeError("adaptive JSONL controller exceeded max_turns")
                pending.append(dict(request))
            return len(text)

        def flush(self) -> None:
            stdout.flush()

    return run_interactive_jsonl(
        stdin=AdaptiveStdin(),
        stdout=AdaptiveStdout(),
        artifact_dir=artifact_dir,
        timeout=timeout,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control the real ClawCodex REPL through a PTY.")
    sub = parser.add_subparsers(dest="mode", required=True)

    interactive = sub.add_parser("interactive", help="Run JSONL controller over stdin/stdout.")
    interactive.add_argument("--timeout", type=float, default=30.0)
    interactive.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "clawcodex-repl-pty",
    )

    script = sub.add_parser("run-script", help="Run the built-in goal smoke script.")
    script.add_argument("--goal", default="Verify REPL PTY debug harness")
    script.add_argument(
        "--prompt",
        default="Return the exact token named goal pty ok using hyphens and uppercase.",
    )
    script.add_argument("--expect-response", default="GOAL-PTY-OK")
    script.add_argument("--timeout", type=float, default=60.0)
    script.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "clawcodex-repl-pty",
    )
    script.add_argument("--cmd", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir = default_artifact_dir(args.artifact_root)
    if args.mode == "interactive":
        return run_interactive_jsonl(
            stdin=sys.stdin,
            stdout=sys.stdout,
            artifact_dir=artifact_dir,
            timeout=args.timeout,
        )
    if args.mode == "run-script":
        command = args.cmd if args.cmd else default_repl_command()
        result = run_script(
            command=command,
            steps=build_goal_script(
                goal=args.goal,
                prompt=args.prompt,
                expect_response=args.expect_response,
            ),
            artifact_dir=artifact_dir,
            timeout=args.timeout,
        )
        print(f"ok={result.ok}")
        print(f"artifacts={result.artifact_dir}")
        return 0 if result.ok else 1
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
