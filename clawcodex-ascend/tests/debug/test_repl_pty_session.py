#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

from __future__ import annotations

import io
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from clawcodex_ext.debug.repl_pty_session import (
    ReplPtySession,
    Step,
    build_goal_script,
    default_repl_command,
    run_interactive_jsonl,
    run_script,
    strip_ansi,
    write_artifacts,
)
import clawcodex_ext.debug.repl_pty_session as repl_pty_session


def fake_command() -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("fake_repl_child.py"))]


class _FakeChild:
    def __init__(self, fd: int) -> None:
        self.child_fd = fd
        self.linesep = "\n"
        self.encoding = "utf-8"


class _FakeClosableChild(_FakeChild):
    def __init__(self, fd: int) -> None:
        super().__init__(fd)
        self.closed_force: bool | None = None

    def isalive(self) -> bool:
        return self.closed_force is None

    def close(self, *, force: bool = False) -> None:
        self.closed_force = force


def test_strip_ansi_removes_escape_sequences() -> None:
    assert strip_ansi("\x1b[31mGoal active\x1b[0m") == "Goal active"


def test_default_repl_command_uses_current_python_module() -> None:
    command = default_repl_command()

    assert command[:3] == [sys.executable, "-m", "clawcodex_ext.cli.main"]
    assert "--legacy-repl" in command
    assert "--stream" in command
    assert "--agent-debug" in command
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"


def test_compatibility_script_exposes_controller_cli() -> None:
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/debug/repl_pty_session.py"), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Control the real ClawCodex REPL through a PTY." in result.stdout
    assert "interactive" in result.stdout
    assert "run-script" in result.stdout


def test_session_start_send_observe_stop(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)

    ready = session.start()
    assert ready.ok is True
    assert ready.event == "ready"
    assert ready.kind == "ready"
    assert ready.state == "ready"
    assert "CLAWCODEX_AGENT_DEBUG::repl.ready::" in ready.delta

    observed = session.send("/goal verify", expect="Goal set:")
    assert observed.ok is True
    assert observed.event == "observed"
    assert observed.kind == "slash_command"
    assert observed.state == "idle"
    assert "input_echo" in observed.signals
    assert "Goal set: verify" in observed.delta
    assert "Goal set: verify" in observed.screen

    observed = session.send("/goal", expect="Tokens:")
    assert observed.ok is True
    assert "Tokens: 82" in observed.delta

    stopped = session.stop()
    assert stopped.ok is True
    assert stopped.event == "stopped"


def test_session_send_write_timeout_does_not_block_on_full_child_fd(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        while True:
            try:
                os.write(write_fd, b"x" * 65536)
            except BlockingIOError:
                break
        os.set_blocking(write_fd, True)
        session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=0.02)
        session._child = _FakeChild(write_fd)  # type: ignore[assignment]

        try:
            session.send("long prompt that cannot fit in the pty buffer", timeout=0.02)
        except TimeoutError as exc:
            assert "timed out writing" in str(exc)
            assert "child PTY" in str(exc)
        else:
            raise AssertionError("send should not block indefinitely or succeed")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_session_stop_write_timeout_force_closes_full_child_fd(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        while True:
            try:
                os.write(write_fd, b"x" * 65536)
            except BlockingIOError:
                break
        os.set_blocking(write_fd, True)
        session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=0.02)
        child = _FakeClosableChild(write_fd)
        session._child = child  # type: ignore[assignment]

        stopped = session.stop()

        assert stopped.ok is True
        assert stopped.event == "stopped"
        assert child.closed_force is True
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_session_start_accepts_prompt_when_ready_marker_is_missing(tmp_path: Path) -> None:
    session = ReplPtySession(
        command=fake_command(),
        artifact_dir=tmp_path,
        timeout=0.5,
        env={"FAKE_REPL_READY_MARKER": "0"},
    )

    ready = session.start()

    assert ready.ok is True
    assert ready.event == "ready"
    assert ready.kind == "ready"
    assert ready.state == "ready"
    assert "prompt_ready_fallback" in ready.signals
    assert "CLAWCODEX_AGENT_DEBUG::repl.ready::" not in ready.delta
    session.stop()


def test_outer_transcript_audit_ignores_prompts_and_flags_tool_calls(tmp_path: Path) -> None:
    transcript = tmp_path / "clean.txt"
    transcript.write_text(
        "\n".join(
            [
                "Hard rule: do not read tests/skills or clawcodex_ext/debug/repl_pty_session.py.",
                "● Bash (cd /repo && .venv/bin/python3 /repo/.agents/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py --repo-root /repo)",
                "● Bash (bash /repo/.agents/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.sh --repo-root /repo)",
                "● Bash (cat /tmp/driver_stderr.txt)",
                '  ⎿ Traceback: File "/repo/clawcodex_ext/debug/repl_pty_session.py", line 1035',
                "● Bash (find tests/skills/clawcodex-repl-pty-debug -maxdepth 2 -type f)",
                "● Read (./clawcodex_ext/debug/repl_pty_session.py · lines 1-80)",
            ]
        ),
        encoding="utf-8",
    )
    script = Path("tests/skills/clawcodex-repl-pty-debug/scripts/audit_outer_transcript.py")

    result = subprocess.run(
        [sys.executable, str(script), "--json", str(transcript)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["tool"] == "Bash"
    assert payload["findings"][0]["target"] == "tests/skills"
    assert payload["findings"][1]["tool"] == "Read"
    assert payload["findings"][1]["target"] == "clawcodex_ext/debug/repl_pty_session.py"


def test_outer_transcript_audit_can_require_adaptive_order(tmp_path: Path) -> None:
    passing = tmp_path / "passing-clean.txt"
    passing.write_text(
        "\n".join(
            [
                'controller response: {"event":"observed","kind":"assistant_output","state":"idle","signals":["prompt"]}',
                "decision: B wrote the first file, so ask B to add the footer next.",
                'next controller op: {"op":"send","label":"turn2 add footer"}',
            ]
        ),
        encoding="utf-8",
    )
    failing = tmp_path / "failing-clean.txt"
    failing.write_text(
        "\n".join(
            [
                'controller response: {"event":"observed","kind":"assistant_output","state":"idle"}',
                'next controller op: {"op":"send","label":"turn2 add footer"}',
            ]
        ),
        encoding="utf-8",
    )
    script = Path("tests/skills/clawcodex-repl-pty-debug/scripts/audit_outer_transcript.py")
    root = Path(__file__).resolve().parents[2]

    passed = subprocess.run(
        [sys.executable, str(script), "--json", "--require-adaptive-order", str(passing)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    failed = subprocess.run(
        [sys.executable, str(script), "--json", "--require-adaptive-order", str(failing)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert passed.returncode == 0, passed.stderr
    passing_payload = json.loads(passed.stdout)
    assert passing_payload["ok"] is True
    assert passing_payload["adaptive_order"]["ok"] is True
    assert passing_payload["findings"] == []

    assert failed.returncode == 1
    failing_payload = json.loads(failed.stdout)
    assert failing_payload["ok"] is False
    assert failing_payload["adaptive_order"]["ok"] is False
    assert failing_payload["findings"][0]["type"] == "adaptive_order"


def test_outer_transcript_audit_reads_adaptive_driver_jsonl_order(tmp_path: Path) -> None:
    passing = tmp_path / "adaptive-driver.jsonl"
    passing.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "observed",
                        "label": "turn1",
                        "kind": "assistant_output",
                        "state": "idle",
                        "delta": "wrote first file",
                    }
                ),
                json.dumps(
                    {
                        "event": "decider_request",
                        "source": "decide_next(response)",
                        "basis": {"event": "observed", "label": "turn1"},
                        "request": {"op": "send", "label": "turn2"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failing = tmp_path / "driver-without-response.jsonl"
    failing.write_text(
        json.dumps(
            {
                "event": "decider_request",
                "source": "decide_next(response)",
                "basis": {"event": "observed"},
                "request": {"op": "send", "label": "turn2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path("tests/skills/clawcodex-repl-pty-debug/scripts/audit_outer_transcript.py")
    root = Path(__file__).resolve().parents[2]

    passed = subprocess.run(
        [sys.executable, str(script), "--json", "--require-adaptive-order", str(passing)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    failed = subprocess.run(
        [sys.executable, str(script), "--json", "--require-adaptive-order", str(failing)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert passed.returncode == 0, passed.stderr
    passing_payload = json.loads(passed.stdout)
    assert passing_payload["adaptive_order"]["ok"] is True
    assert passing_payload["adaptive_order"]["records"][0]["structured"] is True

    assert failed.returncode == 1
    failing_payload = json.loads(failed.stdout)
    assert failing_payload["adaptive_order"]["ok"] is False
    assert failing_payload["adaptive_order"]["records"][0]["response_line"] is None


def test_session_expect_matches_token_split_by_terminal_noise(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("interleaved-token", expect="GOAL-PTY-OK")

    assert observed.ok is True
    assert "GOAL-PTY" in observed.delta
    assert "-OK" in observed.delta
    session.stop()


def test_session_delta_filters_cpr_spinner_and_status_noise(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("noisy-assistant", expect="NOISY-ASSISTANT-OK")

    assert observed.ok is True
    assert observed.kind == "assistant_output"
    assert "NOISY-ASSISTANT-OK" in observed.delta
    assert "doesn't support cursor position requests" not in observed.delta
    assert "Thinking" not in observed.delta
    assert "deepseek-v4-flash" not in observed.delta
    assert "⠋" not in observed.delta
    session.stop()


def test_completed_assistant_output_with_spinner_redraw_returns_idle(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("interleaved-token-with-prompt", expect="PTY-SMOKE-OK")

    assert observed.ok is True
    assert observed.kind == "assistant_output"
    assert observed.state == "idle"
    assert "streaming" in observed.signals
    assert "prompt" in observed.signals
    session.stop()


def test_session_classifies_assistant_and_error_output(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    assistant = session.send("hello", expect="echo:hello")
    assert assistant.ok is True
    assert assistant.kind == "assistant_output"
    assert assistant.state == "idle"
    assert "assistant_output" in assistant.signals

    provider_error = session.send("provider-error", timeout=0.5)
    assert provider_error.ok is False
    assert provider_error.event == "error"
    assert provider_error.kind == "provider_error"
    assert provider_error.state == "error"
    assert provider_error.error_kind == "provider_error"
    assert "ProviderError: invalid_api_key" in (provider_error.error or "")

    network_error = session.send("network-error", timeout=0.5)
    assert network_error.ok is False
    assert network_error.event == "error"
    assert network_error.kind == "network_error"
    assert network_error.state == "error"
    assert network_error.error_kind == "network_error"
    assert "DNS lookup failed" in (network_error.error or "")

    rendered_network_error = session.send("rendered-connection-error", timeout=0.5)
    assert rendered_network_error.ok is False
    assert rendered_network_error.event == "error"
    assert rendered_network_error.kind == "network_error"
    assert rendered_network_error.state == "error"
    assert rendered_network_error.error_kind == "network_error"
    assert "Connection error." in (rendered_network_error.error or "")

    session.stop()


def test_successful_tool_output_can_contain_error_examples(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("/tool Skill doc-error-text", expect='"success": true')

    assert observed.ok is True
    assert observed.event == "observed"
    assert observed.kind == "slash_command"
    assert observed.error_kind is None
    assert "network_error" not in observed.signals
    assert "`Connection error`" in observed.delta
    session.stop()


def test_session_classifies_permission_prompt(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    permission = session.send("permission-prompt", timeout=0.5)

    assert permission.ok is True
    assert permission.event == "observed"
    assert permission.kind == "permission_prompt"
    assert permission.state == "awaiting_permission"
    assert permission.error_kind is None
    assert "permission_prompt" in permission.signals
    session.stop()


def test_session_does_not_keep_resolved_permission_prompt_active(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    resolved = session.send("permission-resolved", timeout=0.5)

    assert resolved.ok is True
    assert resolved.event == "observed"
    assert resolved.kind == "assistant_output"
    assert resolved.state == "idle"
    assert resolved.error_kind is None
    assert "permission_prompt" not in resolved.signals
    session.stop()


def test_session_does_not_keep_rendered_tool_result_permission_prompt_active(
    tmp_path: Path,
) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    resolved = session.send("permission-resolved-rendered", timeout=0.5)

    assert resolved.ok is True
    assert resolved.event == "observed"
    assert resolved.kind == "assistant_output"
    assert resolved.state == "idle"
    assert resolved.error_kind is None
    assert "permission_prompt" not in resolved.signals
    session.stop()


def test_session_key_resolves_interactive_permission_prompt(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    permission = session.send("permission-interactive", timeout=0.5)
    assert permission.ok is True
    assert permission.kind == "permission_prompt"
    assert permission.state == "awaiting_permission"
    assert "permission_prompt" in permission.signals

    resolved = session.key("y", timeout=1.0)

    assert resolved.ok is True
    assert resolved.event == "observed"
    assert resolved.kind == "assistant_output"
    assert resolved.state == "idle"
    assert resolved.error_kind is None
    assert "permission_prompt" not in resolved.signals
    assert "SC5-PERMISSION-KEY-OK" in resolved.delta
    session.stop()


def test_session_key_denies_interactive_permission_prompt_and_continues(
    tmp_path: Path,
) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    permission = session.send("permission-interactive", timeout=0.5)
    assert permission.ok is True
    assert permission.kind == "permission_prompt"
    assert permission.state == "awaiting_permission"

    denied = session.key("n", timeout=1.0)

    assert denied.ok is True
    assert denied.event == "observed"
    assert denied.kind == "assistant_output"
    assert denied.state == "idle"
    assert denied.error_kind is None
    assert "permission_prompt" not in denied.signals
    assert "permission denied by fake child" in denied.delta

    continued = session.send("after-deny-check", expect="echo:after-deny-check")
    assert continued.ok is True
    assert continued.kind == "assistant_output"
    session.stop()


def test_session_expect_ignores_initial_terminal_echo(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send(
        "silent ECHO-ONLY-MATCH",
        expect="ECHO-ONLY-MATCH",
        timeout=0.2,
    )

    assert observed.ok is False
    assert observed.event == "error"
    assert "silent ECHO-ONLY-MATCH" in observed.delta
    session.stop()


def test_session_expect_ignores_wrapped_initial_terminal_echo(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()
    sent = "silent " + ("wrapped-input " * 30) + "WRAPPED-ECHO-ONLY-MATCH"

    observed = session.send(sent, expect="WRAPPED-ECHO-ONLY-MATCH", timeout=0.2)

    assert observed.ok is False
    assert observed.event == "error"
    assert "WRAPPED-ECHO-ONLY-MATCH" in observed.delta
    session.stop()


def test_wrapped_initial_terminal_echo_is_removed_before_expect_matching() -> None:
    sent = "long prompt " + ("wrapped input " * 20) + "ECHO-ONLY-TOKEN"
    wrapped_echo = "❯ " + sent[:90] + "\n" + sent[90:180] + "\n" + sent[180:] + "\n"
    delta = wrapped_echo + "\nAssistant\n❯ "

    stripped = repl_pty_session._without_initial_terminal_echo(delta, sent)

    assert "ECHO-ONLY-TOKEN" not in stripped
    assert "Assistant" in stripped


def test_partial_initial_terminal_echo_is_not_matchable() -> None:
    sent = "long prompt " + ("wrapped input " * 20) + "ECHO-ONLY-TOKEN and more trailing text"
    partial_echo = "❯ " + sent[: sent.index("and more")]

    stripped = repl_pty_session._without_initial_terminal_echo(partial_echo, sent)

    assert stripped == ""


def test_session_expect_does_not_match_loose_subsequence(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("token-status", expect="Tokens used:", timeout=0.3)

    assert observed.ok is False
    assert "Tokens: 0" in observed.delta
    session.stop()


def test_send_without_expect_waits_for_output_after_terminal_echo(tmp_path: Path) -> None:
    session = ReplPtySession(command=fake_command(), artifact_dir=tmp_path, timeout=5.0)
    session.start()

    observed = session.send("delayed-output", timeout=1.5)

    assert observed.ok is True
    assert "delayed-output" in observed.delta
    assert "late-output" in observed.delta
    session.stop()


def test_write_artifacts_writes_raw_clean_and_result(tmp_path: Path) -> None:
    result = write_artifacts(
        artifact_dir=tmp_path,
        raw_text="\x1b[32mGoal active\x1b[0m",
        command=["fake"],
        events=[{"event": "observed", "kind": "slash_command", "ok": True}],
        ok=True,
        error=None,
    )

    assert result.raw_log.read_text() == "\x1b[32mGoal active\x1b[0m"
    assert result.clean_transcript.read_text() == "Goal active"
    payload = json.loads(result.result_json.read_text())
    assert payload["ok"] is True
    assert payload["command"] == ["fake"]
    assert payload["events"][0]["kind"] == "slash_command"


def test_write_artifacts_clean_transcript_filters_cpr_spinner_and_status_noise(
    tmp_path: Path,
) -> None:
    result = write_artifacts(
        artifact_dir=tmp_path,
        raw_text=(
            "WARNING: your terminal doesn't support cursor position requests (CPR).\n"
            "⠋ Thinking…  (esc to interrupt · ctrl+b background · enter to queue · 0s)\n"
            "deepseek · deepseek-v4-flash · /tmp/workspace · mode: Default · turns: 0 · tokens: 0 in / 0 out\n"
            "NOISY-ASSISTANT-OK\n"
        ),
        command=["fake"],
        events=[],
        ok=True,
        error=None,
    )

    clean = result.clean_transcript.read_text()
    assert "NOISY-ASSISTANT-OK" in clean
    assert "doesn't support cursor position requests" not in clean
    assert "Thinking" not in clean
    assert "deepseek-v4-flash" not in clean


def test_build_file_creation_prompt_requires_actual_file_and_path_verification() -> None:
    build_file_creation_prompt = getattr(
        repl_pty_session,
        "build_file_creation_prompt",
        None,
    )
    assert build_file_creation_prompt is not None

    prompt = build_file_creation_prompt(
        target_path="/tmp/hello/hello_server.py",
        task="create a Python http.server Hello World page",
    )

    assert "/tmp/hello/hello_server.py" in prompt
    assert "create or overwrite the file" in prompt
    assert "do not only print code" in prompt
    assert "verify the file exists" in prompt


def test_build_goal_script_is_a_regression_scenario() -> None:
    assert build_goal_script(
        goal="verify",
        prompt="hello",
        expect_response="echo:hello",
    ) == [
        Step(label="start goal", send="/goal verify", expect="Goal set:"),
        Step(label="send prompt", send="hello", expect="echo:hello"),
        Step(label="inspect goal", send="/goal", expect="Tokens:"),
        Step(label="clear goal", send="/goal clear", expect="Goal cleared"),
        Step(label="exit", send="/exit", expect="Goodbye!"),
    ]


def test_run_script_executes_goal_steps(tmp_path: Path) -> None:
    result = run_script(
        command=fake_command(),
        steps=build_goal_script(
            goal="verify",
            prompt="hello",
            expect_response="echo:hello",
        ),
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert result.ok is True
    assert "Tokens: 82" in result.clean_transcript.read_text()
    payload = json.loads(result.result_json.read_text())
    kinds = [event["kind"] for event in payload["events"]]
    assert "slash_command" in kinds
    assert "assistant_output" in kinds


def test_run_interactive_jsonl_supports_dynamic_turns(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps({"op": "send", "text": "/goal verify", "expect": "Goal set:"}),
                json.dumps({"op": "send", "text": "hello", "expect": "echo:hello"}),
                json.dumps({"op": "observe", "timeout": 0.1}),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["event"] == "ready"
    assert lines[1]["ok"] is True
    assert "Goal set: verify" in lines[1]["delta"]
    assert "echo:hello" in lines[2]["delta"]
    assert lines[-1]["event"] == "stopped"


def test_run_adaptive_jsonl_chooses_next_turn_from_previous_response(
    tmp_path: Path,
) -> None:
    run_adaptive_jsonl = getattr(repl_pty_session, "run_adaptive_jsonl", None)
    assert run_adaptive_jsonl is not None

    decisions: list[tuple[str, str, str | None]] = []

    def decide_next(payload: dict[str, object]) -> dict[str, object] | None:
        kind = str(payload.get("kind"))
        state = str(payload.get("state"))
        label = payload.get("label")
        decisions.append((kind, state, label if isinstance(label, str) else None))

        if payload.get("event") == "ready":
            return {
                "op": "send",
                "text": "/goal adaptive-local",
                "expect": "Goal set:",
                "label": "turn1 set goal after ready",
            }
        if label == "turn1 set goal after ready":
            assert "Goal set: adaptive-local" in str(payload.get("delta"))
            return {
                "op": "send",
                "text": "/goal",
                "expect": "Tokens:",
                "label": "turn2 inspect after goal set",
            }
        if label == "turn2 inspect after goal set":
            assert "Status: active" in str(payload.get("delta"))
            return {
                "op": "send",
                "text": "hello-adaptive",
                "expect": "echo:hello-adaptive",
                "label": "turn3 assistant after goal proof",
            }
        if label == "turn3 assistant after goal proof":
            assert payload.get("kind") == "assistant_output"
            return {"op": "stop", "label": "stop after adaptive proof"}
        if label == "stop after adaptive proof":
            return {"op": "exit", "label": "exit adaptive controller"}
        return None

    stdout = io.StringIO()

    rc = run_adaptive_jsonl(
        first_request={"op": "start", "cmd": fake_command(), "label": "start adaptive child"},
        decide_next=decide_next,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["label"] for line in lines[:-1]] == [
        "start adaptive child",
        "turn1 set goal after ready",
        "turn2 inspect after goal set",
        "turn3 assistant after goal proof",
        "stop after adaptive proof",
    ]
    assert decisions == [
        ("ready", "ready", "start adaptive child"),
        ("slash_command", "idle", "turn1 set goal after ready"),
        ("slash_command", "idle", "turn2 inspect after goal set"),
        ("assistant_output", "idle", "turn3 assistant after goal proof"),
        ("stopped", "stopped", "stop after adaptive proof"),
        ("None", "None", None),
    ]


def test_run_adaptive_jsonl_can_decide_from_filesystem_evidence(
    tmp_path: Path,
) -> None:
    run_adaptive_jsonl = getattr(repl_pty_session, "run_adaptive_jsonl", None)
    assert run_adaptive_jsonl is not None

    page_path = tmp_path / "adaptive-page.html"
    decision_path = tmp_path / "adaptive-decision.json"
    decisions: list[tuple[str, str]] = []

    def decide_next(payload: dict[str, object]) -> dict[str, object] | None:
        label = payload.get("label")

        if payload.get("event") == "ready":
            decisions.append(("ready", "send writer command"))
            return {
                "op": "send",
                "text": f"write-adaptive-file {page_path}",
                "expect": "ADAPTIVE-FILE-WROTE",
                "label": "turn1 write page",
            }
        if label == "turn1 write page":
            page_text = page_path.read_text(encoding="utf-8")
            assert "<h1>Adaptive PTY</h1>" in page_text
            decisions.append(("page exists", "send footer command"))
            return {
                "op": "send",
                "text": f"append-adaptive-footer {page_path}",
                "expect": "ADAPTIVE-FOOTER-WROTE",
                "label": "turn2 append footer",
            }
        if label == "turn2 append footer":
            page_text = page_path.read_text(encoding="utf-8")
            assert 'data-adaptive="round2"' in page_text
            decisions.append(("footer verified", "write decision record"))
            return {
                "op": "send",
                "text": f"write-adaptive-decision {decision_path}",
                "expect": "ADAPTIVE-DECISION-WROTE",
                "label": "turn3 write decision",
            }
        if label == "turn3 write decision":
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            assert decision == {"basis": "footer verified", "next": "done"}
            decisions.append(("decision verified", "stop controller"))
            return {"op": "stop", "label": "stop after filesystem proof"}
        if label == "stop after filesystem proof":
            return {"op": "exit", "label": "exit filesystem adaptive controller"}
        return None

    stdout = io.StringIO()

    rc = run_adaptive_jsonl(
        first_request={"op": "start", "cmd": fake_command(), "label": "start adaptive child"},
        decide_next=decide_next,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    assert page_path.exists()
    assert decision_path.exists()
    assert decisions == [
        ("ready", "send writer command"),
        ("page exists", "send footer command"),
        ("footer verified", "write decision record"),
        ("decision verified", "stop controller"),
    ]

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["label"] for line in lines[:-1]] == [
        "start adaptive child",
        "turn1 write page",
        "turn2 append footer",
        "turn3 write decision",
        "stop after filesystem proof",
    ]


def test_run_adaptive_jsonl_can_recover_from_allowed_error(
    tmp_path: Path,
) -> None:
    run_adaptive_jsonl = getattr(repl_pty_session, "run_adaptive_jsonl", None)
    assert run_adaptive_jsonl is not None

    decisions: list[tuple[str, str, str | None]] = []

    def decide_next(payload: dict[str, object]) -> dict[str, object] | None:
        label = payload.get("label")
        event = str(payload.get("event"))
        error_kind = payload.get("error_kind")
        decisions.append((event, str(error_kind), label if isinstance(label, str) else None))

        if payload.get("event") == "ready":
            return {
                "op": "send",
                "text": "token-status",
                "expect": "Tokens used:",
                "timeout": 0.2,
                "allow_error": True,
                "label": "turn1 allowed timeout probe",
            }
        if label == "turn1 allowed timeout probe":
            assert payload.get("event") == "error"
            assert payload.get("error_kind") == "timeout"
            assert "Tokens: 0/inf" in str(payload.get("delta"))
            return {
                "op": "send",
                "text": "token-status",
                "expect": "Turns executed:",
                "label": "turn2 repair after allowed timeout",
            }
        if label == "turn2 repair after allowed timeout":
            return {"op": "stop", "label": "stop after allowed-error recovery"}
        if label == "stop after allowed-error recovery":
            return {"op": "exit", "label": "exit after allowed-error recovery"}
        return None

    stdout = io.StringIO()

    rc = run_adaptive_jsonl(
        first_request={"op": "start", "cmd": fake_command(), "label": "start adaptive child"},
        decide_next=decide_next,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["event"] == "error"
    assert lines[1]["ok"] is False
    assert lines[1]["allow_error"] is True
    assert lines[2]["event"] == "observed"
    assert lines[2]["ok"] is True
    assert [line["label"] for line in lines[:-1]] == [
        "start adaptive child",
        "turn1 allowed timeout probe",
        "turn2 repair after allowed timeout",
        "stop after allowed-error recovery",
    ]

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["error"] is None
    assert result["events"][1]["allow_error"] is True
    assert result["events"][1]["error_kind"] == "timeout"
    assert decisions == [
        ("ready", "None", "start adaptive child"),
        ("error", "timeout", "turn1 allowed timeout probe"),
        ("observed", "None", "turn2 repair after allowed timeout"),
        ("stopped", "None", "stop after allowed-error recovery"),
        ("controller_exit", "None", None),
    ]


def test_run_interactive_jsonl_send_accepts_text_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello\nworld\n", encoding="utf-8")
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text_file": str(prompt_file),
                        "expect": "echo:hello world",
                        "label": "text-file prompt",
                    }
                ),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path / "artifacts",
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["ok"] is True
    assert lines[1]["op"] == "send"
    assert lines[1]["label"] == "text-file prompt"
    assert lines[1]["input_source"] == "text_file"
    assert "echo:hello world" in lines[1]["delta"]

    result = json.loads((tmp_path / "artifacts" / "result.json").read_text(encoding="utf-8"))
    assert result["events"][1]["op"] == "send"
    assert result["events"][1]["label"] == "text-file prompt"
    assert result["events"][1]["input_source"] == "text_file"


def test_run_interactive_jsonl_send_folds_multiline_text(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text": "hello\nworld",
                        "expect": "echo:hello world",
                        "label": "multiline text prompt",
                    }
                ),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path / "artifacts",
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["ok"] is True
    assert lines[1]["input_source"] == "text"
    assert "echo:hello world" in lines[1]["delta"]


def test_run_interactive_jsonl_raw_preserves_literal_newlines(
    tmp_path: Path,
) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "raw",
                        "text": "raw-one\nraw-two\n",
                        "label": "literal newline raw input",
                        "timeout": 1.0,
                    }
                ),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path / "artifacts",
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["ok"] is True
    assert lines[1]["op"] == "raw"
    assert lines[1]["input_source"] == "raw"
    assert "echo:raw-one" in lines[1]["delta"]
    assert "echo:raw-two" in lines[1]["delta"]

    result = json.loads((tmp_path / "artifacts" / "result.json").read_text(encoding="utf-8"))
    assert result["events"][1]["op"] == "raw"
    assert result["events"][1]["label"] == "literal newline raw input"
    assert result["events"][1]["input_source"] == "raw"
    assert "echo:raw-one" in result["events"][1]["delta"]
    assert "echo:raw-two" in result["events"][1]["delta"]


def test_run_interactive_jsonl_can_limit_stdout_observation_text(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text": "hello",
                        "expect": "echo:hello",
                        "max_output_chars": 5,
                    }
                ),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=artifact_dir,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["ok"] is True
    assert "truncated_fields" in lines[1]
    assert lines[1]["truncated_fields"]["delta"] > 5
    assert "full transcript in artifacts" in lines[1]["delta"]
    assert lines[1]["delta"].endswith("> ")

    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert "echo:hello" in result["events"][1]["delta"]
    assert "truncated_fields" not in result["events"][1]


def test_text_file_truncates_stdout_but_keeps_artifacts_complete(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    prompt_file = tmp_path / "prompt.txt"
    long_token = "TEXTFILE-LONG-" + ("0123456789" * 24) + "-COMPLETE"
    prompt_file.write_text(long_token + "\n", encoding="utf-8")
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text_file": str(prompt_file),
                        "expect": "echo:TEXTFILE-LONG-",
                        "label": "text-file truncated stdout",
                        "max_output_chars": 80,
                    }
                ),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=artifact_dir,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[1]["ok"] is True
    assert lines[1]["input_source"] == "text_file"
    assert lines[1]["label"] == "text-file truncated stdout"
    assert "truncated_fields" in lines[1]
    assert lines[1]["truncated_fields"]["delta"] > 80
    assert "full transcript in artifacts" in lines[1]["delta"]

    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert "truncated_fields" not in result["events"][1]
    assert result["events"][1]["input_source"] == "text_file"
    assert f"echo:{long_token}" in result["events"][1]["delta"]
    assert long_token in (artifact_dir / "clean.txt").read_text(encoding="utf-8")
    assert long_token in (artifact_dir / "raw.log").read_text(encoding="utf-8")


def test_truncated_output_keeps_head_and_tail() -> None:
    truncated = repl_pty_session._truncate_output_text("0123456789abcdef", 6)

    assert truncated.startswith("012")
    assert truncated.endswith("def")
    assert "truncated 10 chars" in truncated


def test_run_interactive_jsonl_returns_zero_when_start_uses_prompt_fallback(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {
                        "op": "start",
                        "cmd": fake_command(),
                        "env": {"FAKE_REPL_READY_MARKER": "0"},
                        "timeout": 0.5,
                    }
                ),
                json.dumps({"op": "send", "text": "/goal verify", "expect": "Goal set:"}),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["event"] == "ready"
    assert lines[0]["ok"] is True
    assert "prompt_ready_fallback" in lines[0]["signals"]
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["ok"] is True
    assert payload["error"] is None


def test_run_interactive_jsonl_writes_partial_artifacts_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    class InterruptAfterStart(io.StringIO):
        def __init__(self, text: str) -> None:
            super().__init__(text)
            self._lines_read = 0

        def readline(self, size: int = -1) -> str:
            self._lines_read += 1
            if self._lines_read > 1:
                raise KeyboardInterrupt
            return super().readline(size)

    stdin = InterruptAfterStart(json.dumps({"op": "start", "cmd": fake_command()}) + "\n")
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 130
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["event"] == "ready"
    assert lines[1]["event"] == "error"
    assert lines[1]["error"] == "KeyboardInterrupt"
    assert (tmp_path / "raw.log").exists()
    assert (tmp_path / "clean.txt").exists()
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["ok"] is False
    assert payload["error"] == "KeyboardInterrupt"
    assert payload["events"][0]["event"] == "ready"
    assert payload["events"][-1]["event"] == "stopped"
    assert payload["events"][-1]["op"] == "interrupt_stop"
    assert payload["events"][-1]["label"] == "cleanup after KeyboardInterrupt"


def test_run_interactive_jsonl_exit_ends_controller_after_stop(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps({"op": "stop"}),
                json.dumps({"op": "exit"}),
                json.dumps({"op": "start", "cmd": fake_command()}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["event"] for line in lines] == ["ready", "stopped", "controller_exit"]


def test_run_interactive_jsonl_isolates_artifacts_between_starts(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text": "hello",
                        "expect": "not-in-output",
                        "timeout": 0.2,
                    }
                ),
                json.dumps({"op": "stop"}),
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps({"op": "send", "text": "hello", "expect": "echo:hello"}),
                json.dumps({"op": "stop"}),
                json.dumps({"op": "exit"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 1
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["artifact_dir"] == str(tmp_path)
    assert lines[3]["artifact_dir"] == str(tmp_path / "session-2")

    first = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "session-2" / "result.json").read_text(encoding="utf-8"))

    assert first["ok"] is False
    assert "not-in-output" in first["error"]
    assert second["ok"] is True
    assert second["error"] is None
    assert "echo:hello" in (tmp_path / "session-2" / "clean.txt").read_text(encoding="utf-8")


def test_run_interactive_jsonl_stops_active_session_before_restart(tmp_path: Path) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps(
                    {
                        "op": "send",
                        "text": "hello",
                        "expect": "echo:hello",
                    }
                ),
                json.dumps({"op": "start", "cmd": fake_command()}),
                json.dumps({"op": "send", "text": "hello-again", "expect": "echo:hello-again"}),
                json.dumps({"op": "stop"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    rc = run_interactive_jsonl(
        stdin=stdin,
        stdout=stdout,
        artifact_dir=tmp_path,
        timeout=5.0,
    )

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line["event"] for line in lines] == [
        "ready",
        "observed",
        "stopped",
        "ready",
        "observed",
        "stopped",
    ]
    assert lines[2]["artifact_dir"] == str(tmp_path)
    assert lines[3]["artifact_dir"] == str(tmp_path / "session-2")

    first = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "session-2" / "result.json").read_text(encoding="utf-8"))
    assert first["ok"] is True
    assert "echo:hello" in first["events"][1]["delta"]
    assert second["ok"] is True
    assert "echo:hello-again" in second["events"][1]["delta"]


def _load_skill_script_module(name: str):
    root = Path(__file__).resolve().parents[2]
    path = root / "tests/skills/clawcodex-repl-pty-debug/scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decider_helpers_prefer_current_delta_over_cumulative_screen() -> None:
    helpers = _load_skill_script_module("decider_helpers.py")

    stale_screen_response = {
        "kind": "assistant_output",
        "state": "idle",
        "signals": ["prompt"],
        "delta": 'Tool result:\n{"stdout":"done"}\n\n❯ ',
        "screen": 'Permission Required\n1. Yes\n2. No\nTool result:\n{"stdout":"done"}\n\n❯ ',
    }
    active_prompt_response = {
        "kind": "permission_prompt",
        "state": "awaiting_permission",
        "signals": ["permission_prompt"],
        "delta": "Permission Required\n1. Yes\n2. No",
        "screen": "Permission Required\n1. Yes\n2. No",
    }

    assert helpers.has_cumulative_text(stale_screen_response, "Permission Required")
    assert helpers.has_current_permission_prompt(stale_screen_response) is False
    assert helpers.has_current_permission_prompt(active_prompt_response) is True


def test_decider_helpers_parse_current_bash_exit_code() -> None:
    helpers = _load_skill_script_module("decider_helpers.py")

    ok_response = {
        "delta": 'Tool result:\n{"stdout":"ok","stderr":"","exit_code":0}',
        "screen": 'older {"exit_code": 1}',
    }
    failed_response = {
        "delta": 'Tool result:\n{"stdout":"","stderr":"bad","exit_code":2}',
    }

    assert helpers.bash_exit_code(ok_response) == 0
    assert helpers.bash_succeeded(ok_response) is True
    assert helpers.bash_exit_code(failed_response) == 2
    assert helpers.bash_succeeded(failed_response) is False


def test_skill_adaptive_driver_runs_callback_decisions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "adaptive-driver.jsonl").write_text(
        json.dumps({"event": "stale", "label": "old run"}) + "\n",
        encoding="utf-8",
    )
    decider.write_text(
        "\n".join(
            [
                "from decider_helpers import has_current_text",
                "",
                "FAKE_COMMAND = " + repr(fake_command()),
                "",
                "def first_request():",
                '    return {"op": "start", "cmd": FAKE_COMMAND, "label": "start fake"}',
                "",
                "def decide_next(response):",
                '    label = response.get("label")',
                '    event = response.get("event")',
                '    if label == "start fake" and event == "ready":',
                '        return {"op": "send", "text": "hello", "expect": "echo:hello", "label": "send hello"}',
                '    if label == "send hello" and event == "observed" and has_current_text(response, "echo:hello"):',
                '        return {"op": "stop", "label": "stop fake"}',
                '    if label == "stop fake" and event == "stopped":',
                '        return {"op": "exit", "label": "exit"}',
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [line["label"] for line in lines[:-1]] == [
        "start fake",
        "send hello",
        "stop fake",
    ]
    assert lines[-1]["event"] == "controller_exit"
    assert (artifact_dir / "result.json").exists()
    progress_lines = [
        json.loads(line) for line in (artifact_dir / "adaptive-driver.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_lines[0]["event"] == "decider_request"
    assert progress_lines[0]["request"]["label"] == "start fake"
    assert all(line.get("event") != "stale" for line in progress_lines)
    controller_lines = [line for line in progress_lines if line.get("event") != "decider_request"]
    decision_lines = [line for line in progress_lines if line.get("event") == "decider_request"]
    assert [line["label"] for line in controller_lines[:-1]] == [
        "start fake",
        "send hello",
        "stop fake",
    ]
    assert [line["request"]["label"] for line in decision_lines] == [
        "start fake",
        "send hello",
        "stop fake",
        "exit",
    ]
    assert decision_lines[1]["basis"]["label"] == "start fake"
    assert decision_lines[1]["basis"]["event"] == "ready"
    assert decision_lines[1]["basis"]["current_permission_prompt"] is False
    assert decision_lines[1]["basis"]["screen_mentions_permission"] is False
    assert decision_lines[1]["request"]["op"] == "send"


def test_skill_adaptive_driver_ignores_duplicate_stop_after_stopped(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "FAKE_COMMAND = " + repr(fake_command()),
                "",
                "def first_request():",
                '    return {"op": "start", "cmd": FAKE_COMMAND, "label": "start fake"}',
                "",
                "def decide_next(response):",
                '    if response.get("event") == "ready":',
                '        return {"op": "stop", "label": "stop fake"}',
                '    if response.get("event") == "stopped":',
                '        return {"op": "stop", "label": "stop fake again"}',
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "session has not been started" not in completed.stdout
    progress_lines = [
        json.loads(line) for line in (artifact_dir / "adaptive-driver.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    warnings = [line for line in progress_lines if line.get("event") == "decider_warning"]
    assert len(warnings) == 1
    assert warnings[0]["basis"]["event"] == "stopped"
    assert warnings[0]["basis"]["label"] == "stop fake"
    assert warnings[0]["basis"]["artifact_dir"] == str(artifact_dir)
    assert warnings[0]["ignored_request"] == {"label": "stop fake again", "op": "stop"}
    assert warnings[0]["message"] == "ignored duplicate stop after stopped; return None after a stopped response"
    assert warnings[0]["source"] == "decide_next(response)"


def test_skill_adaptive_driver_rejects_missing_contract_with_skeleton(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "def main():",
                '    return {"op": "start"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert "decider must define callable first_request()" in completed.stderr
    assert "def first_request():" in completed.stderr
    assert "def decide_next(response):" in completed.stderr
    error_payload = json.loads((artifact_dir / "driver-error.json").read_text(encoding="utf-8"))
    assert error_payload["event"] == "driver_error"
    assert error_payload["stage"] == "preflight"
    assert "decider must define callable first_request()" in error_payload["error"]
    progress_lines = [
        json.loads(line) for line in (artifact_dir / "adaptive-driver.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_lines == [error_payload]


def test_skill_adaptive_driver_shell_wrapper_reports_driver_error(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.sh"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "def decide_next(response):",
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PTY_ADAPTIVE_PYTHON": sys.executable},
        check=False,
    )

    assert completed.returncode != 0
    assert "PTY_ADAPTIVE_DRIVER_EXIT=" in completed.stdout
    assert "PTY_ADAPTIVE_DRIVER_ERROR_JSON_BEGIN" in completed.stdout
    assert "decider must define callable first_request()" in completed.stdout
    assert (artifact_dir / "driver-error.json").exists()
    assert (artifact_dir / "adaptive-driver.jsonl").exists()


def test_skill_adaptive_driver_requires_first_request_start(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "def first_request():",
                '    return {"op": "send", "text": "hello"}',
                "",
                "def decide_next(response):",
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert 'first_request() must return {"op": "start", ...}' in completed.stderr
    assert "put the first user turn in decide_next(response) after ready" in completed.stderr


def test_skill_adaptive_driver_rejects_action_key_with_clear_message(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.py"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "def first_request():",
                '    return {"action": "start"}',
                "",
                "def decide_next(response):",
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert 'must use "op", not "action"' in completed.stderr


def test_skill_adaptive_driver_shell_wrapper_reports_runtime_decider_error(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = root / "tests/skills/clawcodex-repl-pty-debug/scripts/pty_adaptive_driver.sh"
    decider = tmp_path / "decider.py"
    artifact_dir = tmp_path / "artifacts"
    decider.write_text(
        "\n".join(
            [
                "FAKE_COMMAND = " + repr(fake_command()),
                "",
                "def first_request():",
                '    return {"op": "start", "cmd": FAKE_COMMAND, "label": "start fake"}',
                "",
                "def decide_next(response):",
                '    if response.get("event") == "ready":',
                '        return {"action": "send", "text": "hello", "label": "bad action"}',
                "    return None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            str(driver),
            "--repo-root",
            str(root),
            "--artifact-dir",
            str(artifact_dir),
            "--decider",
            str(decider),
            "--timeout",
            "5",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PTY_ADAPTIVE_PYTHON": sys.executable},
        check=False,
    )

    assert completed.returncode != 0
    assert "PTY_ADAPTIVE_DRIVER_EXIT=" in completed.stdout
    assert "PTY_ADAPTIVE_DRIVER_ERROR_JSON_BEGIN" in completed.stdout
    assert "decide_next(response) must use" in completed.stdout
    error_payload = json.loads((artifact_dir / "driver-error.json").read_text(encoding="utf-8"))
    assert error_payload["event"] == "driver_error"
    assert error_payload["stage"] == "run"
    assert 'decide_next(response) must use "op", not "action"' in error_payload["error"]
    progress_lines = [
        json.loads(line) for line in (artifact_dir / "adaptive-driver.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress_lines[-1] == error_payload


def test_docs_use_current_goal_expectations() -> None:
    root = Path(__file__).resolve().parents[2]
    skill_root = root / "tests/skills/clawcodex-repl-pty-debug"
    main_skill = skill_root / "SKILL.md"
    reference_paths = [
        skill_root / "references/controller-protocol.md",
        skill_root / "references/failure-classification.md",
        skill_root / "references/live-provider-and-goals.md",
        skill_root / "references/nested-skill-validation.md",
    ]
    docs = [main_skill.read_text(encoding="utf-8")]
    docs.extend(path.read_text(encoding="utf-8") for path in reference_paths)
    bundle = "\n".join(docs)
    main_text = docs[0]
    normalized_bundle = " ".join(bundle.split())
    normalized_main = " ".join(main_text.split())

    assert len(docs[0].split()) <= 1200
    for path in reference_paths:
        assert path.exists()
        assert path.name in docs[0]

    for expected in [
        "## Permission Mode",
        "Every run plan must state whether permission prompts are in scope.",
        "Default child PTY starts use `--permission-mode bypassPermissions`.",
        "`--permission-mode bypassPermissions` only when permission prompts are out of scope",
        "Do not use `bypassPermissions` when permission prompts are in scope.",
        "Adaptive multi-turn runs must use the persistent `interactive` controller.",
        "Sandbox or filesystem approval is not live-provider data approval.",
        'An instruction such as "allow unsandboxed execution" is still insufficient',
        "only if the user explicitly allows sending workspace, skill, and prompt context to the external model provider",
    ]:
        assert " ".join(expected.split()) in normalized_main

    for expected in [
        '"expect":"Available Commands:"',
        '"expect":"Available tools:"',
        '"expect":"No goal set"',
        "Total units:",
        "CLAW_HEADLESS_BACKEND=stub",
        "CLAWCODEX_HOME",
        "fake_repl_child.py",
        "provider_error",
        "network_error",
        "persistent PTY session API",
        "temporary JSONL driver",
        "pty_jsonl_driver.py",
        "pty_jsonl_driver: controller exited before writing a response",
        "input_source",
        "label",
        "Do not import `scripts/debug/repl_pty_session.py`",
        "CLAWCODEX_MANAGED_SKILLS_DIR",
        ".claude/skills",
        '"loadedFrom": "managed"',
        "text_file",
        "PTY line discipline",
        "max_output_chars",
        "default to reporting only",
        "Existing code is unreasonable",
        "Optimize the narrow code path",
        "Do not edit the skill",
        "nested ClawCodex",
        "actual `Skill` tool call",
        "Tokens used:",
        "external model provider",
        "explicit user approval",
        "workspace, skill, or prompt context",
        "Default permission mode keeps `Permission Required` prompts visible",
        "permission-probe child",
        "choose the next `send`, `observe`, or `key` only after reading the previous JSON response",
        "current response `delta`, `kind`, `state`, `error_kind`, and `signals`",
        "`screen` is cumulative terminal state",
        "independently inspect that filesystem or SQLite evidence before returning the next operation",
        "do not reread large repo files just to rediscover the workflow",
        '`/tool Skill {"skill":"clawcodex-repl-pty-debug"}`',
        "The shorthand `/tool Skill clawcodex-repl-pty-debug` is invalid",
        "Do not stop and restart B between turns unless restart behavior is the explicit target.",
        "Multiple inner artifact directories for ordinary B turns means the run restarted B",
        "response or artifact read, decision, then next send/key/observe",
        "A prewritten `ops.jsonl` is fixed-script evidence, not adaptive multi-turn proof.",
        "`run_adaptive_jsonl`",
        "Return `None` only when no more controller operations are needed",
        'return `{"op":"observe","timeout":...}`',
        "After `key` or `raw`, follow with `observe` unless that response already contains the expected tool result",
        "key` and `raw` send bytes; they do not prove the resulting tool action has finished",
        "After a `stopped` response, return `None`",
        "Do not stop B just because the final file exists.",
        '"allow_error":true',
        "the error remains in artifacts but does not fail the run if later turns recover",
        "send a repair prompt based on the missing evidence",
        "decider_warning",
        "No child directories under that artifact root can mean one child session",
        "`session-*` child directories mean the child was restarted",
        'After `/tool Skill {"skill":"..."}`, treat the returned prompt as already loaded guidance',
        "Do not read `tests/skills`, `.claude/skills`, bundled scripts, or `clawcodex_ext/debug/repl_pty_session.py` merely to rediscover the workflow.",
        "Use the `skillRoot` path returned by `/tool Skill`; do not Glob/search for skill copies.",
        "Mirror directories are not authoritative.",
        "If the task already provides an exact helper command, run that command directly",
        "Treat any `ls`, `find`, Glob, Read, or Grep under `.agents/skills`, `.claude/skills`, `tests/skills`",
        "No-discovery applies to the outer A agent's own tool calls",
        "Checking only `adaptive-driver.jsonl` is insufficient",
        "audit the outer transcript, outer `result.json`, outer `clean.txt`, outer `raw.log`, or the controlling agent",
        "do not mark it passed",
        "Do not read helper implementation files unless the run has already been classified as a helper-layer failure and no-discovery failure.",
        "Prefer direct `start`/`send`/`observe`/`stop` controller operations before inspecting helper implementation.",
        "pty_adaptive_driver.py",
        "Use `pty_adaptive_driver.py` when a shell-only agent needs adaptive decisions without rewriting controller plumbing.",
        "Decider files can `import decider_helpers`",
        "has_current_permission_prompt(response)",
        "bash_succeeded(response)",
        'Returned requests must use `"op"`, not `"action"`.',
        '`first_request()` must return `{"op":"start",...}`',
        "put the first B user prompt in `decide_next(response)` after the `ready` response",
        "`send` text is ClawCodex user input, not a shell command",
        "Multiline `send.text` and `send.text_file` are folded into one REPL input line",
        "bounded chunks",
        "write timeout",
        "literal newlines are the behavior under test",
        "adaptive-driver.jsonl",
        "decider_request",
        "def first_request():",
        "def decide_next(response):",
        'response.get("event")',
        "Useful `response` fields include `event`, `op`, `label`, `kind`, `state`",
        "PTY_ADAPTIVE_DRIVER_EXIT=0",
        '"exit_code": 0',
        "if the outer agent times out before `result.json` exists",
    ]:
        assert " ".join(expected.split()) in normalized_bundle
    assert "Reply with exactly" not in bundle
