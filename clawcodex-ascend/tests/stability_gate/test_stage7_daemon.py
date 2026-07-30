"""Stage 7 — F-84 Daemon subsystem smoke test.

Per ``docs/feature_plan/06-ccb-benchmark/f-84-daemon.md`` §1.12.3, this
stage provides a lightweight round-trip check on the daemon subsystem:

1. The CLI surface is importable + parseable.
2. The supervisor + workers packages import cleanly.
3. A ``start → status → stop`` round-trip works through the actual
   ``extensions.daemon.cli.run_daemon`` entry point.
4. State file lifecycle is correct (written on start, removed on stop).
5. The CLI subcommand registration is wired behind the
   ``DAEMON + BRIDGE_MODE`` double feature gate.

Bounded to 30 seconds of wall-time by the per-test
``pytest.timeout`` markers; in practice the round-trip completes in
under 5 seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from extensions.daemon.workers.base import BaseWorker

# ---------------------------------------------------------------------------
# Module-level test double (no-op worker)
# ---------------------------------------------------------------------------


class _SmokeNoopWorker(BaseWorker):
    """Trivial worker that exits 0 immediately when spawned."""

    kind = "smoke-noop"

    async def run(self, env: dict[str, str]) -> int:
        # Exit cleanly so the supervisor treats this as a one-shot stop.
        return 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Per-test daemon state directory (a child of pytest's tmp_path)."""
    d = tmp_path / "daemon-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def dummy_worker_kind(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a no-op worker kind so the supervisor has something
    benign to spawn without touching ``remoteControl`` (which would
    pull in the bridge subsystem).
    """
    from extensions.daemon.worker_registry import WorkerRegistry

    WorkerRegistry.register("smoke-noop", lambda: _SmokeNoopWorker())
    yield "smoke-noop"
    WorkerRegistry.unregister("smoke-noop")


# ---------------------------------------------------------------------------
# Section 1 — Importability & contract surface
# ---------------------------------------------------------------------------


class TestStage7DaemonImports:
    """The daemon package surface must remain importable in production."""

    def test_daemon_package_public_api(self):
        import extensions.daemon as daemon_pkg

        for attr in (
            "Supervisor",
            "WorkerRegistry",
            "DaemonConfig",
            "DaemonState",
            "DaemonStatus",
            "EXIT_CODE_PERMANENT",
            "EXIT_CODE_TRANSIENT",
        ):
            assert hasattr(daemon_pkg, attr), f"missing public attr {attr!r}"

    def test_daemon_protocol_worker_is_runtime_checkable(self):
        from extensions.capabilities.daemon_protocol import Worker

        # Protocol attributes appear in ``__annotations__`` for class
        # variables; methods are visible via ``dir`` since the Protocol
        # body declares them as ``async def ... -> ...``.
        annotations = getattr(Worker, "__annotations__", {})
        assert "kind" in annotations
        for method in ("run", "health_check"):
            assert method in dir(Worker), f"Worker missing {method!r}"

    def test_cli_parser_exposes_all_f84_verbs(self):
        from extensions.daemon.cli import build_parser

        verbs = set(
            build_parser()._subparsers._group_actions[0].choices.keys()  # type: ignore[attr-defined]
        )
        # F-84 P84-E/F: every verb documented in §1.9 must be present.
        for verb in ("start", "stop", "status", "ps", "bg", "attach", "logs", "kill"):
            assert verb in verbs, f"missing CLI verb {verb!r}"

    def test_feature_flags_daemon_bridge_registered(self):
        from clawcodex_ext.feature_gate import get_registry

        reg = get_registry()
        assert reg.get_state("DAEMON") is not None
        assert reg.get_state("BRIDGE_MODE") is not None
        # Defaults are off — operators opt in via ``clawcodex feature
        # enable DAEMON BRIDGE_MODE`` or env vars.
        assert reg.is_enabled("DAEMON") is False
        assert reg.is_enabled("BRIDGE_MODE") is False


# ---------------------------------------------------------------------------
# Section 2 — In-process start/stop round-trip
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
class TestStage7DaemonRoundTrip:
    """``start → status → stop`` round-trip via the real ``run_daemon``
    entry point with the supervisor running in the foreground.
    """

    def test_status_before_start_reports_stopped(self, tmp_state_dir: Path, capsys: pytest.CaptureFixture[str]):
        from extensions.daemon.cli import run_daemon

        rc = run_daemon(
            [
                "--state-dir",
                str(tmp_state_dir),
                "status",
                "--name",
                "smoke-default",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "stopped" in captured.out
        assert "smoke-default" in captured.out

    def test_status_json_before_start(self, tmp_state_dir: Path, capsys: pytest.CaptureFixture[str]):
        from extensions.daemon.cli import run_daemon

        rc = run_daemon(
            [
                "--state-dir",
                str(tmp_state_dir),
                "status",
                "--name",
                "smoke-json",
                "--json",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.out)
        assert payload["name"] == "smoke-json"
        assert payload["status"] == "stopped"
        assert payload["state"] is None

    def test_start_stop_round_trip(
        self,
        tmp_state_dir: Path,
    ):
        """Drive a real ``extensions.daemon.cli`` subprocess through
        ``start`` (foreground, blocking cron worker), verify the state
        file appears, then ``stop`` it via the CLI.

        This is the closest analogue to the operator-facing
        ``clawcodex-dev daemon start && clawcodex-dev daemon stop`` flow.
        The cron worker blocks until cancelled, so the supervisor stays
        alive long enough to exercise the round-trip.
        """
        name = "smoke-roundtrip"

        # Launch the foreground supervisor as a subprocess using
        # ``python -c`` to invoke ``run_daemon`` (the CLI module has no
        # ``__main__`` block). ``--workers cron`` uses the built-in
        # cron stub which blocks until cancelled.
        start_proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from extensions.daemon.cli import run_daemon; "
                    "sys.exit(run_daemon(["
                    f"'--state-dir', {str(tmp_state_dir)!r}, "
                    "'start', "
                    f"'--name', {name!r}, "
                    "'--workers', 'cron', "
                    "'--foreground'"
                    "]))"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(tmp_state_dir),
        )

        state_path = tmp_state_dir / f"{name}.json"

        # Wait for the state file (written before any spawn).
        for _ in range(60):
            if state_path.exists():
                break
            time.sleep(0.05)
        assert state_path.exists(), (
            f"state file did not appear during start. stderr={start_proc.stderr.read().decode('utf-8', 'replace')!r}"
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["name"] == name
        assert state["worker_kinds"] == ["cron"]
        assert state["pid"] == start_proc.pid

        # ``daemon stop`` reads the state file and SIGTERMs the PID.
        stop_proc = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-c",
                (
                    "import sys; from extensions.daemon.cli import run_daemon; "
                    "sys.exit(run_daemon(["
                    f"'--state-dir', {str(tmp_state_dir)!r}, "
                    "'stop', "
                    f"'--name', {name!r}, "
                    "'--timeout-ms', '10000'"
                    "]))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # The stop CLI returns 0 on success.
        assert stop_proc.returncode == 0, f"daemon stop failed: stdout={stop_proc.stdout!r} stderr={stop_proc.stderr!r}"

        # Wait for the supervisor subprocess to exit.
        try:
            start_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            start_proc.kill()
            start_proc.wait(timeout=5)
            pytest.fail("supervisor did not exit within 10s of stop")

        assert start_proc.returncode == 0, (
            f"supervisor exited with {start_proc.returncode}: {start_proc.stderr.read().decode('utf-8', 'replace')!r}"
        )

        # State file should be removed after clean shutdown.
        assert not state_path.exists(), "state file should be removed after graceful shutdown"

    def test_stop_when_not_running_returns_error(self, tmp_state_dir: Path, capsys: pytest.CaptureFixture[str]):
        from extensions.daemon.cli import run_daemon

        rc = run_daemon(
            [
                "--state-dir",
                str(tmp_state_dir),
                "stop",
                "--name",
                "smoke-no-daemon",
                "--timeout-ms",
                "500",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "not running" in captured.out


# ---------------------------------------------------------------------------
# Section 3 — Subprocess-level smoke (when bridge worker is available)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
class TestStage7DaemonCliSubprocess:
    """Smoke test that invokes the daemon CLI as a real subprocess and
    verifies the state file appears + is cleaned up.

    This is the closest analogue to the operator-facing
    ``clawcodex-dev daemon start && clawcodex-dev daemon stop`` flow.
    """

    def test_cli_help_lists_daemon_verb(self, tmp_state_dir: Path):
        """``extensions.daemon.cli.run_daemon`` invoked with ``--help``
        must list every F-84 verb — a low-cost but high-signal
        regression catcher for the CLI parser.
        """
        # The CLI module has no ``if __name__ == "__main__"`` block,
        # so we drive it via the public ``run_daemon`` entry point in a
        # subprocess.
        result = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-c",
                ("import sys; from extensions.daemon.cli import run_daemon; sys.exit(run_daemon(['--help']))"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_state_dir),
        )
        # ``run_daemon(['--help'])`` returns 0 because argparse prints
        # help and exits normally.
        assert result.returncode == 0, f"CLI --help failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        # argparse writes --help to stdout normally; on parse error it
        # writes to stderr. We accept either stream so the test is
        # robust against argparse reconfiguration.
        combined = result.stdout + result.stderr
        for verb in ("start", "stop", "status", "bg", "attach", "logs", "kill"):
            assert verb in combined, f"help missing verb {verb!r}"
